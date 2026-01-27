import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.decomposition import PCA
import requests
from io import StringIO

dates_df = pd.read_csv('dates.csv', parse_dates=['Date'])
target_dates = dates_df['Date'].tolist()

nfp_surp = pd.read_csv('nfp_surp.csv', parse_dates=['Date']).set_index('Date')
base = pd.read_csv('base.csv', parse_dates=['Date']).set_index('Date')
rel_date_df = pd.read_csv('rel_date.csv', parse_dates=[0])
rel_dates = rel_date_df.iloc[:, 0].sort_values().tolist()
bcom = pd.read_csv('bcomsp.csv', parse_dates=[0], index_col=0)
skew = pd.read_csv('skew.csv', parse_dates=[0], index_col=0)

nfp_level = pdr.DataReader('PAYEMS', 'fred', '1980-01-01', '2025-12-31')['PAYEMS']

sp500_df = yf.download('^GSPC', start='1980-01-01', end='2025-12-31', progress=False)
sp500 = sp500_df['Close'].squeeze()
sp500.index = pd.to_datetime(sp500.index).tz_localize(None)

yield_url = 'https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv'
resp = requests.get(yield_url, headers={'User-Agent': 'Mozilla/5.0'})
yield_df = pd.read_csv(StringIO(resp.text), skiprows=9, parse_dates=['Date']).set_index('Date')
yield_cols = [f'SVENY{i:02d}' for i in range(1, 11)]
yields = yield_df[yield_cols].dropna()

pca = PCA(n_components=4)
pca_result = pca.fit_transform(yields)
pc2 = pd.Series(pca_result[:, 1], index=yields.index, name='PC2')

def get_latest_release(target, rel_list):
    valid = [r for r in rel_list if r <= target]
    return valid[-1] if valid else None

def get_latest_up_to(series, date, inclusive=True):
    valid = series[series.index <= date] if inclusive else series[series.index < date]
    return valid.iloc[-1] if len(valid) > 0 else np.nan

def get_latest_idx(series, date, inclusive=True):
    valid = series[series.index <= date] if inclusive else series[series.index < date]
    return valid.index[-1] if len(valid) > 0 else None

results, raw_surprises = [], []
missing = {k: [] for k in ['NFP_SURP', 'NFP_12M', 'SP500_3M', 'SLOPE_3M', 'BCOM_3M', 'TR_SKEW']}
june_1999, may_2025 = pd.Timestamp('1999-06-01'), pd.Timestamp('2025-05-01')

for dt in target_dates:
    row = {'Date': dt}
    nfp_rel = get_latest_release(dt, rel_dates)
    
    # NFP_SURP
    try:
        src = nfp_surp.iloc[:, 0] if dt >= june_1999 else base['NFP_SURP']
        raw_surp = get_latest_up_to(src, dt)
        surp_idx = get_latest_idx(src, dt)
        nfp_val = get_latest_up_to(nfp_level, surp_idx) if surp_idx else np.nan
        scaled = raw_surp / nfp_val if pd.notna(raw_surp) and pd.notna(nfp_val) else np.nan
        raw_surprises.append({'Date': dt, 'surp_date': surp_idx, 'raw_surp': raw_surp, 
                              'nfp_level': nfp_val, 'scaled': scaled})
    except:
        raw_surprises.append({'Date': dt, 'surp_date': None, 'raw_surp': np.nan, 
                              'nfp_level': np.nan, 'scaled': np.nan})
        missing['NFP_SURP'].append(dt)
    
    # NFP_12M
    try:
        idx = get_latest_idx(nfp_level, nfp_rel) if nfp_rel else None
        if idx:
            nfp_now = nfp_level.loc[idx]
            nfp_1y = get_latest_up_to(nfp_level, idx - pd.DateOffset(years=1))
            row['NFP_12M'] = np.log(nfp_now / nfp_1y) * 100
        else:
            row['NFP_12M'] = np.nan
            missing['NFP_12M'].append(dt)
    except:
        row['NFP_12M'] = np.nan
        missing['NFP_12M'].append(dt)
    
    # SP500_3M (65 trading days)
    try:
        sp_idx = get_latest_idx(sp500, dt, inclusive=False)
        sp_now = sp500.loc[sp_idx]
        sp_past = sp500[sp500.index <= sp_idx].iloc[-66]
        row['SP500_3M'] = np.log(sp_now / sp_past) * 100
    except:
        row['SP500_3M'] = np.nan
        missing['SP500_3M'].append(dt)
    
    # SLOPE_3M (63 trading days - consistent with ~3 months)
    try:
        pc2_before = pc2[pc2.index < dt]
        if len(pc2_before) >= 64:
            pc2_now = pc2_before.iloc[-1]
            pc2_3m = pc2_before.iloc[-64]
            row['SLOPE_3M'] = -pc2_now + pc2_3m
        else:
            row['SLOPE_3M'] = np.nan
            missing['SLOPE_3M'].append(dt)
    except:
        row['SLOPE_3M'] = np.nan
        missing['SLOPE_3M'].append(dt)
    
    # BCOM_3M (63 trading days for consistency)
    try:
        bcom_before = bcom.iloc[:, 0][bcom.index < dt]
        if len(bcom_before) >= 64:
            row['BCOM_3M'] = np.log(bcom_before.iloc[-1] / bcom_before.iloc[-64]) * 100
        else:
            row['BCOM_3M'] = np.nan
            missing['BCOM_3M'].append(dt)
    except:
        row['BCOM_3M'] = np.nan
        missing['BCOM_3M'].append(dt)
    
    # TR_SKEW (average of most recent 30 days before target)
    try:
        skew_before = skew.iloc[:, 0][skew.index < dt]
        if len(skew_before) >= 30:
            row['TR_SKEW'] = skew_before.iloc[-30:].mean()
        else:
            row['TR_SKEW'] = skew_before.mean() if len(skew_before) > 0 else np.nan
            if dt < may_2025:
                missing['TR_SKEW'].append(dt)
    except:
        row['TR_SKEW'] = np.nan
        if dt < may_2025:
            missing['TR_SKEW'].append(dt)
    
    results.append(row)

raw_df = pd.DataFrame(raw_surprises)
scaled_series = raw_df['scaled']
#raw_df['NFP_SURP_std'] = (scaled_series - scaled_series.mean()) / scaled_series.std()
trimmed = scaled_series.sort_values().iloc[1:-1]
raw_df['NFP_SURP_std'] = (scaled_series - trimmed.mean()) / trimmed.std()

out_df = pd.DataFrame(results)
out_df['NFP_SURP'] = raw_df['NFP_SURP_std'].values
out_df = out_df[['Date', 'NFP_SURP', 'NFP_12M', 'SP500_3M', 'SLOPE_3M', 'BCOM_3M', 'TR_SKEW']]

out_df.to_csv('output_variables.csv', index=False)
raw_df[['Date', 'surp_date', 'raw_surp', 'nfp_level', 'scaled']].to_csv('nfp_raw_data.csv', index=False)

with open('missing_dates.txt', 'w') as f:
    for var, dates in missing.items():
        f.write(f"{var}:\n")
        for d in dates:
            f.write(f"  {d.strftime('%Y-%m-%d')}\n")
        f.write("\n")

print("Done.")
