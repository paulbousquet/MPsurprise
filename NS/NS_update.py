## Construct NS (Nakamura-Steinsson style) monetary policy surprises
## Filename: ns_updated.py
## Outputs: NS_updated.csv with ns_rep and ns_ext columns

import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##################################################
### CONFIGURATION
##################################################

USMPD_URL = "https://www.frbsf.org/wp-content/uploads/USMPD.xlsx"
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

##################################################


def download_usmpd() -> pd.ExcelFile:
    """Download USMPD from SF Fed."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(USMPD_URL, headers=headers)
    response.raise_for_status()
    return pd.ExcelFile(BytesIO(response.content))


def load_y1() -> pd.DataFrame:
    """Load daily one-year GSW Treasury yield for normalization."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(GSW_URL, headers=headers)
    response.raise_for_status()
    y1 = pd.read_csv(StringIO(response.text), skiprows=9)
    y1["Date"] = pd.to_datetime(y1["Date"])
    y1 = y1[["Date", "SVENY01"]].dropna()
    y1["dy1"] = y1["SVENY01"].diff()
    return y1


def calc_mps(data: pd.DataFrame, cols: list[str], y1: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MPS from intraday rate changes.

    Args:
        data: DataFrame with Date column and rate changes
        cols: Column names to use for PCA
        y1: DataFrame with Date and dy1 (daily yield changes)

    Returns:
        DataFrame with Date and MPS columns
    """
    df = data[["Date"] + cols].dropna().copy()

    # Standardize and run PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])

    pca = PCA(n_components=1)
    df["PC1"] = pca.fit_transform(X_scaled).flatten()

    # Merge with yield data
    df = df.merge(y1[["Date", "dy1"]], on="Date", how="left")

    # Regress dy1 on PC1 to get normalization coefficient (drop NaN for regression)
    df_reg = df[["PC1", "dy1"]].dropna()
    X = np.column_stack([np.ones(len(df_reg)), df_reg["PC1"].values])
    y = df_reg["dy1"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    coef_pc1 = beta[1]

    # MPS = coefficient * PC1
    df["MPS"] = coef_pc1 * df["PC1"]

    return df[["Date", "MPS"]]


def main():
    print("Downloading USMPD from SF Fed...")
    xl = download_usmpd()

    print("Loading GSW yields...")
    y1 = load_y1()

    cols = ["MP1", "MP2", "ED2", "ED3", "ED4"]

    # ============================================================
    # ns_rep: Statements, scheduled meetings only, 1995 to present
    # ============================================================
    print("\nComputing ns_rep (Statements, scheduled, 1995-present)...")
    stmt = pd.read_excel(xl, sheet_name="Statements")
    stmt["Date"] = pd.to_datetime(stmt["Date"])

    # Filter: 1995+, scheduled only
    stmt = stmt[stmt["Date"].dt.year >= 1995]
    stmt = stmt[stmt["Unscheduled"] != 1]

    print(f"  {len(stmt)} observations: {stmt['Date'].min().date()} to {stmt['Date'].max().date()}")

    ns_rep = calc_mps(stmt, cols, y1)
    ns_rep = ns_rep.rename(columns={"MPS": "ns_rep"})

    # ============================================================
    # ns_ext: Monetary Events, 1995 to present
    #         Exclude 9/17/2001 and unscheduled COVID meetings
    # ============================================================
    print("\nComputing ns_ext (Monetary Events, 1995-present, excl. 9/2001 & COVID unscheduled)...")
    me = pd.read_excel(xl, sheet_name="Monetary Events")
    me["Date"] = pd.to_datetime(me["Date"])

    # Filter: 1995+
    me = me[me["Date"].dt.year >= 1995]

    # Exclude September 2001 (9/17/2001)
    me = me[~((me["Date"].dt.year == 2001) & (me["Date"].dt.month == 9))]

    # Exclude unscheduled meetings during COVID (2020)
    covid_unscheduled = (me["Date"].dt.year == 2020) & (me["Unscheduled"] == 1)
    me = me[~covid_unscheduled]

    print(f"  {len(me)} observations: {me['Date'].min().date()} to {me['Date'].max().date()}")

    ns_ext = calc_mps(me, cols, y1)
    ns_ext = ns_ext.rename(columns={"MPS": "ns_ext"})

    # ============================================================
    # Combine and output
    # ============================================================
    print("\nMerging...")
    out = ns_rep.merge(ns_ext, on="Date", how="outer").sort_values("Date")

    out.to_csv("NS_updated.csv", index=False)
    print(f"\nWrote NS_updated.csv with {len(out)} observations")
    print(out.head(10))
    print("...")
    print(out.tail(10))

    return out


if __name__ == "__main__":
    main()
