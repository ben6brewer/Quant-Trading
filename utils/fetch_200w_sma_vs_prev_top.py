# utils/fetch_200w_sma_vs_prev_top.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from coinmetrics.api_client import CoinMetricsClient
from utils.pretty_print_df import pretty_print_df

def fetch_200w_sma_vs_prev_top(filepath="data/sma_200w_vs_prev_top.parquet"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    client = CoinMetricsClient()

    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        latest = df['date'].max().date()
        if latest >= datetime.utcnow().date() - timedelta(days=1):
            print(f"Cached 200w SMA data up-to-date (latest: {latest}).")
            return df

    print("Fetching PriceUSD from Coin Metrics...")
    result = client.get_asset_metrics(
        assets="btc",
        metrics="PriceUSD",
        start_time="2010-01-01",
        end_time=datetime.utcnow().strftime("%Y-%m-%d"),
        frequency="1d",
        page_size=10000
    )

    df = result.to_dataframe()
    df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.rename(columns={"PriceUSD": "close"}).sort_values("date").reset_index(drop=True)
    df = df[['date', 'close']]

    # Moving averages
    df['sma_200w'] = df['close'].rolling(window=1400).mean()
    df['sma_50w'] = df['close'].rolling(window=350).mean()

    # Hardcoded tops and bottoms
    top_2013 = df.loc[df['date'] == pd.Timestamp("2013-12-05"), 'close'].values[0]
    top_2017 = df.loc[df['date'] == pd.Timestamp("2017-12-16"), 'close'].values[0]
    top_2021 = df.loc[df['date'] == pd.Timestamp("2021-11-08"), 'close'].values[0]

    bottom_2018 = pd.Timestamp("2018-12-16")
    bottom_2022 = pd.Timestamp("2022-11-21")
    bottom_dates = [bottom_2018, bottom_2022]

    # Assign prev_top
    prev_top_vals = []
    for date in df['date']:
        if date < bottom_2018:
            prev_top_vals.append(top_2013)
        elif date < bottom_2022:
            prev_top_vals.append(top_2017)
        else:
            prev_top_vals.append(top_2021)
    df['prev_top'] = prev_top_vals

    # Pure ratio
    df['sma_vs_top_ratio'] = df['sma_200w'] / df['prev_top']

    # Compute fancy risk logic
    risk_series = []
    breached = False
    current_bottom_idx = 0
    next_bottom = bottom_dates[current_bottom_idx] if current_bottom_idx < len(bottom_dates) else pd.Timestamp.max

    for idx, row in df.iterrows():
        date = row['date']
        close = row['close']
        sma_50w = row['sma_50w']
        sma_200w = row['sma_200w']
        prev_top = row['prev_top']

        if pd.isna(close) or pd.isna(sma_50w) or pd.isna(sma_200w) or pd.isna(prev_top):
            risk_series.append(np.nan)
            continue

        ratio = sma_200w / prev_top

        # First breach
        if not breached and ratio >= 1:
            breached = True
            risk_series.append(1.0)
            continue

        # After breach
        if breached:
            if close < sma_50w:
                if date >= next_bottom:
                    breached = False
                    current_bottom_idx += 1
                    next_bottom = bottom_dates[current_bottom_idx] if current_bottom_idx < len(bottom_dates) else pd.Timestamp.max
                    risk_series.append(ratio)
                else:
                    risk_series.append(np.nan)
            else:
                risk_series.append(1.0)
        else:
            risk_series.append(ratio)

    df['sma_cycle_risk'] = risk_series

    df.to_parquet(filepath)
    print(f"Saved 200w SMA vs previous top dataset to {filepath}")
    pretty_print_df(df.tail(10), title="200W SMA vs Previous BTC Top with Risk")
    return df
