# utils/fetch_sopr_historical_data.py

import os
import pandas as pd
from datetime import datetime, timedelta
from coinmetrics.api_client import CoinMetricsClient
from utils.fetch_btc_historical import fetch_btc_historical_data

def fetch_sopr_historical_data(filepath="data/sopr_coinmetrics.parquet", rolling_window=365):
    btc_df = fetch_btc_historical_data()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        try:
            cached_df = pd.read_parquet(filepath)
            cached_df['date'] = pd.to_datetime(cached_df['date']).dt.tz_localize(None)
            btc_df['date'] = pd.to_datetime(btc_df['date']).dt.tz_localize(None)

            latest_date = cached_df['date'].max().date()
            today_date = datetime.utcnow().date()

            if latest_date >= (today_date - timedelta(days=1)):
                print(f"Cached SOPR data is up-to-date (latest: {latest_date}). Using cached data.")
                merged_df = pd.merge(btc_df, cached_df, on='date', how='left')

                if 'asset' in merged_df.columns:
                    merged_df = merged_df.drop(columns=['asset'])

                # Fill SOPR rolling metrics forward
                for col in ['sopr', 'sopr_mean', 'sopr_std', 'sopr_z_score']:
                    if col in merged_df.columns:
                        merged_df[col] = merged_df[col].ffill()

                # Normalize risk
                z_min = merged_df['sopr_z_score'].min()
                z_max = merged_df['sopr_z_score'].max()
                merged_df['sopr_risk'] = (merged_df['sopr_z_score'] - z_min) / (z_max - z_min)

                cols = [
                    'date', 'open', 'high', 'low', 'close', 'volume',
                    'sopr', 'sopr_mean', 'sopr_std', 'sopr_z_score', 'sopr_risk'
                ]
                return merged_df[cols]

            else:
                print(f"Cached SOPR data outdated (latest: {latest_date}). Fetching fresh data...")

        except Exception as e:
            print(f"Error loading cached SOPR data: {e}. Refetching...")

    # Fetch fresh SOPR data
    client = CoinMetricsClient()
    print("Fetching SOPR from CoinMetrics...")

    result = client.get_asset_metrics(
        assets=["btc"],
        metrics=["SOPR"],
        frequency="1d"
    )
    df = result.to_dataframe()

    # Preprocess
    df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.drop(columns=['time'])
    df = df.sort_values('date').reset_index(drop=True)

    df = df.rename(columns={"SOPR": "sopr"})

    # Calculate rolling stats
    df['sopr_mean'] = df['sopr'].rolling(window=rolling_window).mean()
    df['sopr_std'] = df['sopr'].rolling(window=rolling_window).std()
    df['sopr_z_score'] = (df['sopr'] - df['sopr_mean']) / df['sopr_std']

    # Merge with BTC price data
    btc_df['date'] = pd.to_datetime(btc_df['date']).dt.tz_localize(None)
    merged_df = pd.merge(btc_df, df, on='date', how='left')

    # Forward fill missing SOPR values
    for col in ['sopr', 'sopr_mean', 'sopr_std', 'sopr_z_score']:
        merged_df[col] = merged_df[col].ffill()

    # Normalize risk
    z_min = merged_df['sopr_z_score'].min()
    z_max = merged_df['sopr_z_score'].max()
    merged_df['sopr_risk'] = (merged_df['sopr_z_score'] - z_min) / (z_max - z_min)
    print(f"SOPR Z-Score Range: min = {z_min:.3f}, max = {z_max:.3f}")

    # Final columns
    cols = [
        'date', 'open', 'high', 'low', 'close', 'volume',
        'sopr', 'sopr_mean', 'sopr_std', 'sopr_z_score', 'sopr_risk'
    ]
    merged_df = merged_df[cols]

    # Save
    merged_df.to_parquet(filepath)
    print(f"Saved SOPR metrics to {filepath}")
    return merged_df
