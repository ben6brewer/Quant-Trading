# utils/fetch_mvrv_historical.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from coinmetrics.api_client import CoinMetricsClient
from utils.fetch_btc_historical import fetch_btc_historical_data
from utils.pretty_print_df import pretty_print_df

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

    # --- Normalizations ---
    z = merged_df['mvrv_z_score']
    z_min, z_max = z.min(), z.max()
    z_mean, z_std = z.mean(), z.std()

    # Min-Max normalization
    merged_df['min_max_norm'] = (z - z_min) / (z_max - z_min)

    # Z-score norm (same as mvrv_z_score but renaming for consistency)
    merged_df['z_score_norm'] = (z - z_mean) / z_std

    # Sigmoid normalization on z-score
    merged_df['sigmoid_norm'] = 1 / (1 + np.exp(-merged_df['z_score_norm']))

    # Log normalization on shifted z-score (to keep positive values)
    epsilon = 1e-9
    shifted_z = z - z_min + epsilon
    log_vals = np.log(shifted_z)
    log_min, log_max = log_vals.min(), log_vals.max()
    merged_df['log_norm'] = (log_vals - log_min) / (log_max - log_min)

    # Hardcoded choice for risk metric (set to log_norm for now)
    merged_df['mvrv_risk'] = merged_df['log_norm']

    # Forward fill any NaNs that may appear
    for col in ['min_max_norm', 'z_score_norm', 'sigmoid_norm', 'log_norm', 'mvrv_risk']:
        merged_df[col] = merged_df[col].ffill()

    # Save full enriched dataset to cache (includes BTC + MVRV)
    merged_df.to_parquet(filepath)
    print(f"Saved merged MVRV and BTC price data to {filepath}")

    return merged_df
