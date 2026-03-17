"""Diagnostic: check raw Month float values and parsed month correctness."""
import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def parse_month_float(val):
    try:
        val = float(val)
        year = int(val)
        decimal_part = val - year
        month = round(decimal_part * 100)
        if month < 1 or month > 12:
            return None, None, val
        return year, month, val
    except:
        return None, None, val

for fname in ['2021.xlsx','2022.xlsx','2023.xlsx','2024.xlsx','2025.xlsx']:
    df = pd.read_excel(f'data/{fname}', usecols=['Month', 'Delivery qty.', 'Value of supply(KRW)'])
    print(f'\n===== {fname} (shape={df.shape}) =====')
    
    raw_months = sorted(df['Month'].dropna().unique())
    print(f'Raw Month float values: {raw_months}')
    
    for rm in raw_months:
        yr, mn, raw = parse_month_float(rm)
        status = 'OK' if yr and mn else 'FAIL'
        print(f'  {raw} -> year={yr}, month={mn} [{status}]')
    
    df['_yr'] = df['Month'].apply(lambda x: parse_month_float(x)[0])
    df['_mn'] = df['Month'].apply(lambda x: parse_month_float(x)[1])
    valid = df.dropna(subset=['_yr','_mn'])
    invalid = df[df['_yr'].isna() | df['_mn'].isna()]
    
    if len(invalid) > 0:
        print(f'  WARNING: {len(invalid)} rows failed parsing!')
        print(f'  Failed Month values: {invalid["Month"].unique()}')
    
    grp = valid.groupby(['_yr','_mn']).agg(
        rows=('Delivery qty.','count'),
        total_qty=('Delivery qty.','sum'),
        total_rev=('Value of supply(KRW)','sum')
    )
    print(f'Monthly summary:')
    for (yr, mn), row in grp.iterrows():
        print(f'  {int(yr)}-{int(mn):02d}: {int(row["rows"]):4d} rows | qty={row["total_qty"]:>15,.0f} | rev={row["total_rev"]:>18,.0f}')
