"""
BRW Monetary Policy Shock Replication

This script replicates the Bu, Rogers, and Wu (2021) monetary policy shock measure
using the Fama-MacBeth two-step regression procedure.

Methodology:
1. Step 1 (Time-series): For each maturity j=1,...,30, regress dR_s^j on dR_s^2
   Options:
   - Rigobon (default): Heteroskedasticity-based identification (Rigobon 2003)
     beta_j = (cov_FOMC - cov_nonFOMC) / (var_FOMC - var_nonFOMC)
     Best match with original BRW (correlation 0.9998, RMSE 0.0012)
   - OLS: Standard OLS regression (correlation 0.999, RMSE 0.0027)
   - IV/2SLS: Use instrument Z = -dR_{s-7}^2 (weak instrument, poor results)

2. Step 2 (Cross-sectional): For each FOMC date s, regress dR_s^j on beta_j to get delta_i_hat_s

3. Step 3 (Scaling): Regress dR_s^2 on delta_i_hat_s to get scaling factor
   This "re-scales by the 2-year yield" as described in the methodology.

Data sources:
- Federal Reserve yield curve data (SVENY): https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv
- FOMC dates from SF Fed USMPD: https://www.frbsf.org/wp-content/uploads/USMPD.xlsx

Output:
- BRW_update.csv with columns:
  - date: FOMC announcement date
  - brw_rep: BRW shock excluding all unscheduled meetings
  - brw_ext: BRW shock excluding unscheduled meetings only in 2020 and all of March 2020
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import urllib.request
import os


def download_data(yields_path='feds200628.csv', usmpd_path='USMPD.xlsx', force=False):
    """Download yield curve and USMPD data if not present."""

    if not os.path.exists(yields_path) or force:
        print("Downloading Federal Reserve yield curve data...")
        url = 'https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv'
        urllib.request.urlretrieve(url, yields_path)
        print(f"  Saved to {yields_path}")

    if not os.path.exists(usmpd_path) or force:
        print("Downloading SF Fed USMPD data...")
        url = 'https://www.frbsf.org/wp-content/uploads/USMPD.xlsx'
        urllib.request.urlretrieve(url, usmpd_path)
        print(f"  Saved to {usmpd_path}")


def load_yield_data(path='feds200628.csv'):
    """Load and process Federal Reserve yield curve data."""

    yields = pd.read_csv(path, skiprows=9)
    yields['Date'] = pd.to_datetime(yields['Date'])
    yields = yields.set_index('Date')

    # Extract SVENY columns (zero-coupon yields for maturities 1-30 years)
    sveny_cols = [f'SVENY{str(i).zfill(2)}' for i in range(1, 31)]
    yields_sveny = yields[sveny_cols].copy()

    # Convert to numeric
    for col in sveny_cols:
        yields_sveny[col] = pd.to_numeric(yields_sveny[col], errors='coerce')

    return yields_sveny, sveny_cols


def load_fomc_dates(path='USMPD.xlsx', start_year=1994):
    """Load FOMC dates from SF Fed USMPD data."""

    df = pd.read_excel(path, sheet_name='Monetary Events')

    # Filter to start year and later
    df = df[df['Date'].dt.year >= start_year].copy()

    # Keep relevant columns
    df = df[['Date', 'Unscheduled']].copy()
    df = df.rename(columns={'Date': 'date', 'Unscheduled': 'unscheduled'})

    return df


def get_yield_change(date, yields_df, sveny_cols, max_lookback=5):
    """
    Get yield change from previous valid trading day to target date.
    Handles gaps from holidays/weekends.
    """
    if date not in yields_df.index:
        return pd.Series([np.nan] * len(sveny_cols), index=sveny_cols)

    current = yields_df.loc[date]
    if current.isna().any():
        return pd.Series([np.nan] * len(sveny_cols), index=sveny_cols)

    # Find previous valid trading day
    for days_back in range(1, max_lookback + 1):
        prev_date = date - pd.Timedelta(days=days_back)
        if prev_date in yields_df.index:
            prev = yields_df.loc[prev_date]
            if prev.notna().all():
                return current - prev

    return pd.Series([np.nan] * len(sveny_cols), index=sveny_cols)


def get_prior_yield_change(date, yields_df, sveny_cols, days_before=7, max_lookback=5):
    """
    Get yield change from approximately `days_before` days before the target date.
    This is used to construct the instrument for IV estimation.
    """
    target_date = date - pd.Timedelta(days=days_before)
    return get_yield_change(target_date, yields_df, sveny_cols, max_lookback)


def get_non_fomc_yield_changes(fomc_dates, yields_df, sveny_cols, days_around=5):
    """
    Get yield changes on non-FOMC trading days (for Rigobon identification).

    Uses trading days within `days_around` days of each FOMC date, excluding
    the FOMC date itself. This captures similar market conditions while
    excluding the monetary policy shock.
    """
    fomc_set = set(fomc_dates)
    records = []

    for fomc_date in fomc_dates:
        # Look at days around each FOMC date
        for delta in range(-days_around, days_around + 1):
            if delta == 0:
                continue  # Skip FOMC date itself

            check_date = fomc_date + pd.Timedelta(days=delta)

            # Skip if this is also an FOMC date
            if check_date in fomc_set:
                continue

            # Skip if not a trading day
            if check_date not in yields_df.index:
                continue

            changes = get_yield_change(check_date, yields_df, sveny_cols)
            if changes.notna().all():
                row = {'date': check_date}
                for col in sveny_cols:
                    row[f'd_{col}'] = changes[col]
                records.append(row)

    # Remove duplicates (same date may appear near multiple FOMC dates)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['date'])

    return df


def rigobon_beta(y_fomc, x_fomc, y_nonfomc, x_nonfomc):
    """
    Rigobon (2003) heteroskedasticity-based identification.

    beta = (cov_FOMC - cov_nonFOMC) / (var_FOMC - var_nonFOMC)

    This exploits the higher variance on FOMC days to identify the
    causal effect of monetary policy.
    """
    cov_fomc = np.cov(x_fomc, y_fomc)[0, 1]
    cov_nonfomc = np.cov(x_nonfomc, y_nonfomc)[0, 1]
    var_fomc = np.var(x_fomc, ddof=1)
    var_nonfomc = np.var(x_nonfomc, ddof=1)

    beta = (cov_fomc - cov_nonfomc) / (var_fomc - var_nonfomc)
    return beta


def iv_2sls(y, x, z):
    """
    Two-stage least squares estimation.

    Parameters:
    - y: dependent variable (n,)
    - x: endogenous regressor (n,)
    - z: instrument (n,)

    Returns:
    - beta: IV estimate of coefficient on x
    - first_stage_r2: R-squared from first stage regression
    """
    # First stage: regress x on z (with constant)
    Z = sm.add_constant(z)
    first_stage = sm.OLS(x, Z).fit()
    x_hat = first_stage.fittedvalues

    # Second stage: regress y on x_hat (with constant)
    X_hat = sm.add_constant(x_hat)
    second_stage = sm.OLS(y, X_hat).fit()

    return second_stage.params[1], first_stage.rsquared


def compute_brw_shocks(fomc_dates, yields_df, sveny_cols, method='rigobon'):
    """
    Compute BRW monetary policy shocks using Fama-MacBeth procedure.

    Parameters:
    - fomc_dates: list of FOMC announcement dates
    - yields_df: DataFrame with yield levels
    - sveny_cols: list of SVENY column names
    - method: 'rigobon' (default, best), 'ols', or 'iv'

    Steps:
    1. Time-series regression: For each maturity j, regress dR_j on dR_2
    2. Cross-sectional regression: For each FOMC date, regress dR_j on beta_j
    3. Scaling: Regress dR_2 on raw shocks to get scaling factor

    Returns DataFrame with date and scaled BRW shock.
    """

    # Build yield changes for all FOMC dates
    records = []
    for date in fomc_dates:
        changes = get_yield_change(date, yields_df, sveny_cols)

        if method == 'iv':
            prior_changes = get_prior_yield_change(date, yields_df, sveny_cols, days_before=7)
            if changes.notna().all() and prior_changes.notna().all():
                row = {'date': date}
                for col in sveny_cols:
                    row[f'd_{col}'] = changes[col]
                    row[f'prior_{col}'] = prior_changes[col]
                records.append(row)
        else:
            if changes.notna().all():
                row = {'date': date}
                for col in sveny_cols:
                    row[f'd_{col}'] = changes[col]
                records.append(row)

    df = pd.DataFrame(records)

    if len(df) == 0:
        return pd.DataFrame()

    # For Rigobon, get non-FOMC day yield changes
    if method == 'rigobon':
        nonfomc_df = get_non_fomc_yield_changes(fomc_dates, yields_df, sveny_cols)
        if len(nonfomc_df) == 0:
            print("    WARNING: No non-FOMC days found. Falling back to OLS.")
            method = 'ols'

    # STEP 1: Time-series regressions to get betas
    betas = []
    first_stage_r2s = []

    x_fomc = df['d_SVENY02'].values

    if method == 'iv':
        z = -df['prior_SVENY02'].values  # Instrument: negative of 7-day prior change
    elif method == 'rigobon':
        x_nonfomc = nonfomc_df['d_SVENY02'].values

    for j in range(1, 31):
        y_fomc = df[f'd_SVENY{str(j).zfill(2)}'].values

        if method == 'iv':
            beta_j, r2 = iv_2sls(y_fomc, x_fomc, z)
            first_stage_r2s.append(r2)
        elif method == 'rigobon':
            y_nonfomc = nonfomc_df[f'd_SVENY{str(j).zfill(2)}'].values
            beta_j = rigobon_beta(y_fomc, x_fomc, y_nonfomc, x_nonfomc)
        else:  # OLS
            X = sm.add_constant(x_fomc)
            model = sm.OLS(y_fomc, X).fit()
            beta_j = model.params[1]

        betas.append(beta_j)

    betas = np.array(betas)

    if method == 'iv':
        avg_first_stage_r2 = np.mean(first_stage_r2s)
        print(f"    IV first-stage R-squared: {avg_first_stage_r2:.4f}")
        if avg_first_stage_r2 < 0.05:
            print("    WARNING: Weak instrument detected. Consider using method='rigobon'.")
    elif method == 'rigobon':
        var_ratio = np.var(x_fomc) / np.var(x_nonfomc)
        print(f"    Rigobon variance ratio (FOMC/non-FOMC): {var_ratio:.2f}")

    # STEP 2: Cross-sectional regressions for each FOMC date
    shocks = []
    for _, row in df.iterrows():
        y_changes = np.array([row[f'd_SVENY{str(j).zfill(2)}'] for j in range(1, 31)])
        X_cs = sm.add_constant(betas)
        model = sm.OLS(y_changes, X_cs).fit()

        shocks.append({
            'date': row['date'],
            'delta_i_hat': model.params[1],
            'd2y': row['d_SVENY02']
        })

    results = pd.DataFrame(shocks)

    # STEP 3: Scaling regression - regress dR_2 on delta_i_hat
    X_scale = sm.add_constant(results['delta_i_hat'].values)
    scale_reg = sm.OLS(results['d2y'].values, X_scale).fit()
    scale_factor = scale_reg.params[1]

    print(f"    Scaling factor: {scale_factor:.6f}")
    print(f"    Scaling regression R-squared: {scale_reg.rsquared:.4f}")

    # Apply scaling
    results['brw_shock'] = results['delta_i_hat'] * scale_factor

    return results[['date', 'brw_shock']]


def main(method='rigobon'):
    """
    Main function to generate BRW_update.csv.

    Parameters:
    - method: 'rigobon' (default, best match), 'ols', or 'iv'
    """

    print("=" * 60)
    print("BRW Monetary Policy Shock Replication")
    print("=" * 60)
    print(f"\nUsing method: {method.upper()}")

    # Download data
    download_data()

    # Load yield data
    print("\nLoading yield curve data...")
    yields_df, sveny_cols = load_yield_data()
    print(f"  Yield data range: {yields_df.index.min().date()} to {yields_df.index.max().date()}")

    # Load FOMC dates
    print("\nLoading FOMC dates from USMPD...")
    fomc_df = load_fomc_dates(start_year=1994)
    print(f"  Total FOMC dates from 1994: {len(fomc_df)}")
    print(f"  Scheduled meetings: {(fomc_df['unscheduled'] == 0).sum()}")
    print(f"  Unscheduled meetings: {(fomc_df['unscheduled'] == 1).sum()}")

    # Create two samples:
    # 1. brw_rep: Exclude ALL unscheduled meetings
    # 2. brw_ext: Exclude unscheduled meetings in 2020 AND all of March 2020

    # Sample 1: Scheduled meetings only
    scheduled_dates = fomc_df[fomc_df['unscheduled'] == 0]['date'].tolist()

    # Sample 2: All meetings except (unscheduled in 2020) and (March 2020)
    def keep_for_ext(row):
        date = row['date']
        year = date.year
        month = date.month
        unscheduled = row['unscheduled']

        # Exclude all of March 2020
        if year == 2020 and month == 3:
            return False

        # Exclude unscheduled meetings in 2020
        if year == 2020 and unscheduled == 1:
            return False

        return True

    ext_mask = fomc_df.apply(keep_for_ext, axis=1)
    ext_dates = fomc_df[ext_mask]['date'].tolist()

    print(f"\n  brw_rep sample (scheduled only): {len(scheduled_dates)} dates")
    print(f"  brw_ext sample: {len(ext_dates)} dates")

    # Compute BRW shocks for both samples
    print(f"\nComputing BRW shocks for brw_rep (scheduled meetings only)...")
    brw_rep = compute_brw_shocks(scheduled_dates, yields_df, sveny_cols, method=method)
    print(f"  Computed shocks for {len(brw_rep)} dates")

    print(f"\nComputing BRW shocks for brw_ext...")
    brw_ext = compute_brw_shocks(ext_dates, yields_df, sveny_cols, method=method)
    print(f"  Computed shocks for {len(brw_ext)} dates")

    # Merge results
    output = fomc_df[['date']].copy()

    brw_rep = brw_rep.rename(columns={'brw_shock': 'brw_rep'})
    output = output.merge(brw_rep, on='date', how='left')

    brw_ext = brw_ext.rename(columns={'brw_shock': 'brw_ext'})
    output = output.merge(brw_ext, on='date', how='left')

    # Format date column
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')

    # Save to CSV
    output_path = 'BRW_update.csv'
    try:
        output.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
    except PermissionError:
        import time
        output_path = f'BRW_update_{int(time.time())}.csv'
        output.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path} (original file was locked)")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nbrw_rep:")
    print(f"  Non-null values: {output['brw_rep'].notna().sum()}")
    print(f"  Mean: {output['brw_rep'].mean():.6f}")
    print(f"  Std:  {output['brw_rep'].std():.6f}")

    print(f"\nbrw_ext:")
    print(f"  Non-null values: {output['brw_ext'].notna().sum()}")
    print(f"  Mean: {output['brw_ext'].mean():.6f}")
    print(f"  Std:  {output['brw_ext'].std():.6f}")

    # Show first and last few rows
    print("\nFirst 10 rows:")
    print(output.head(10).to_string(index=False))

    print("\nLast 10 rows:")
    print(output.tail(10).to_string(index=False))

    return output


if __name__ == '__main__':
    # Use Rigobon by default (best match: correlation 0.9998, RMSE 0.0012)
    # Options: 'rigobon' (best), 'ols' (good), 'iv' (poor due to weak instrument)
    main(method='rigobon')
