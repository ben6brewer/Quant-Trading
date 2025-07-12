# utils/fetch_mvrv_historical.py

import os
import pandas as pd
from datetime import datetime
from coinmetrics.api_client import CoinMetricsClient
from utils.fetch_btc_historical import fetch_btc_historical_data

def fetch_mvrv_historical_data(filepath="data/mvrv_coinmetrics.parquet", rolling_window=365):
    btc_df = fetch_btc_historical_data()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Ensure proper datetime format (UTC naive)
    btc_df['date'] = pd.to_datetime(btc_df['date']).dt.tz_localize(None)

    if os.path.exists(filepath):
        try:
            cached_df = pd.read_parquet(filepath)
            cached_df['date'] = pd.to_datetime(cached_df['date']).dt.tz_localize(None)

            latest_date = cached_df['date'].max().date()
            today = datetime.utcnow().date()

            if latest_date >= today:
                print(f"Cached MVRV data is up-to-date (latest: {latest_date}). Using cached data.")
                return cached_df

            else:
                print(f"Cached MVRV data outdated (latest: {latest_date}). Fetching fresh data...")

        except Exception as e:
            print(f"Error loading cached MVRV data: {e}. Refetching...")

    # Fetch fresh MVRV data
    client = CoinMetricsClient()
    print("Fetching CapMrktCurUSD and CapRealUSD from CoinMetrics...")

    result = client.get_asset_metrics(
        assets=["btc"],
        metrics=["CapMrktCurUSD", "CapRealUSD"],
        frequency="1d"
    )
    cm_df = result.to_dataframe()

    # Preprocess
    cm_df['date'] = pd.to_datetime(cm_df['time']).dt.tz_localize(None)
    cm_df = cm_df.drop(columns=['time'])
    cm_df = cm_df.sort_values('date').reset_index(drop=True)

    # MVRV calculations
    cm_df['mvrv_ratio'] = cm_df['CapMrktCurUSD'] / cm_df['CapRealUSD']
    cm_df['mvrv_mean'] = cm_df['mvrv_ratio'].rolling(window=rolling_window).mean()
    cm_df['mvrv_std'] = cm_df['mvrv_ratio'].rolling(window=rolling_window).std()
    cm_df['mvrv_z_score'] = (cm_df['mvrv_ratio'] - cm_df['mvrv_mean']) / cm_df['mvrv_std']

    # Merge with BTC OHLCV data
    merged_df = pd.merge(btc_df, cm_df, on='date', how='left')

    if 'asset' in merged_df.columns:
        merged_df = merged_df.drop(columns=['asset'])

    merged_df = merged_df.rename(columns={
        'CapMrktCurUSD': 'market_cap',
        'CapRealUSD': 'realized_market_cap'
    })

    # Forward fill MVRV-related values
    for col in ['mvrv_ratio', 'mvrv_mean', 'mvrv_std', 'mvrv_z_score']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill()

    # Normalize z-score into 0–1 risk score
    z_min = merged_df['mvrv_z_score'].min()
    z_max = merged_df['mvrv_z_score'].max()
    merged_df['mvrv_risk'] = (merged_df['mvrv_z_score'] - z_min) / (z_max - z_min)
    print(f"MVRV Z-Score Range: min = {z_min:.3f}, max = {z_max:.3f}")

    # Save full enriched dataset to cache (includes BTC + MVRV)
    merged_df.to_parquet(filepath)
    print(f"Saved merged MVRV and BTC price data to {filepath}")

    return merged_df
