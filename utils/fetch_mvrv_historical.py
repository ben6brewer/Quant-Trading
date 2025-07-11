# utils/fetch_mvrv_historical.py

import os
import pandas as pd
from datetime import datetime, timedelta
from coinmetrics.api_client import CoinMetricsClient
from utils.fetch_btc_historical import fetch_btc_historical_data

def fetch_mvrv_historical_data(filepath="data/mvrv_coinmetrics.parquet", rolling_window=365):
    # Fetch BTC price data (cached or fresh inside that function)
    btc_df = fetch_btc_historical_data()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        try:
            cached_df = pd.read_parquet(filepath)

            # Clean cached date column: remove tz info and format date string without time & zeros
            cached_df['date'] = pd.to_datetime(cached_df['date']).dt.tz_localize(None)
            cached_df['date'] = cached_df['date'].dt.strftime('%Y-%-m-%-d')

            latest_date = pd.to_datetime(cached_df['date']).max().date()
            today_date = datetime.utcnow().date()

            if latest_date >= (today_date - timedelta(days=1)):
                print(f"Cached MVRV data is up-to-date (latest: {latest_date}). Using cached data.")

                # Also clean btc_df date column for merging
                btc_df['date'] = pd.to_datetime(btc_df['date']).dt.tz_localize(None)
                btc_df['date'] = btc_df['date'].dt.strftime('%Y-%-m-%-d')

                # Merge cached MVRV data with fresh btc_df before returning
                merged_df = pd.merge(btc_df, cached_df, on='date', how='inner')

                # Drop 'asset' column if present
                if 'asset' in merged_df.columns:
                    merged_df = merged_df.drop(columns=['asset'])

                # Rename columns if still present (should be from cached)
                if 'CapMrktCurUSD' in merged_df.columns:
                    merged_df = merged_df.rename(columns={
                        'CapMrktCurUSD': 'market_cap',
                        'CapRealUSD': 'realized_market_cap'
                    })

                # Normalize mvrv_z_score to 0–1 scale
                z_min = merged_df['mvrv_z_score'].min()
                z_max = merged_df['mvrv_z_score'].max()
                merged_df['mvrv_risk'] = (merged_df['mvrv_z_score'] - z_min) / (z_max - z_min)

                # Final column ordering
                cols = [
                    'date', 'open', 'high', 'low', 'close', 'volume',
                    'market_cap', 'realized_market_cap',
                    'mvrv_ratio', 'mvrv_mean', 'mvrv_std', 'mvrv_z_score', 'mvrv_risk'
                ]
                merged_df = merged_df[cols]
                return merged_df
            else:
                print(f"Cached MVRV data outdated (latest: {latest_date}). Fetching fresh data...")
        except Exception as e:
            print(f"Error loading cached data: {e}. Refetching fresh data...")
    else:
        print("No cached MVRV data found. Fetching fresh data...")

    # Fetch Coin Metrics data fresh
    client = CoinMetricsClient()
    print("Fetching CapMrktCurUSD and CapRealUSD from CoinMetrics...")

    result = client.get_asset_metrics(
        assets=["btc"],
        metrics=["CapMrktCurUSD", "CapRealUSD"],
        frequency="1d"
    )

    coinmetrics_df = result.to_dataframe()

    # Parse dates and sort
    coinmetrics_df['time'] = pd.to_datetime(coinmetrics_df['time'])
    coinmetrics_df = coinmetrics_df.rename(columns={'time': 'date'})
    coinmetrics_df = coinmetrics_df.sort_values('date').reset_index(drop=True)

    # Calculate MVRV ratio and stats
    coinmetrics_df['mvrv_ratio'] = coinmetrics_df['CapMrktCurUSD'] / coinmetrics_df['CapRealUSD']
    coinmetrics_df['mvrv_mean'] = coinmetrics_df['mvrv_ratio'].rolling(window=rolling_window).mean()
    coinmetrics_df['mvrv_std'] = coinmetrics_df['mvrv_ratio'].rolling(window=rolling_window).std()
    coinmetrics_df['mvrv_z_score'] = (coinmetrics_df['mvrv_ratio'] - coinmetrics_df['mvrv_mean']) / coinmetrics_df['mvrv_std']

    # Clean and format date columns for merging
    btc_df['date'] = pd.to_datetime(btc_df['date']).dt.tz_localize(None)
    coinmetrics_df['date'] = coinmetrics_df['date'].dt.tz_localize(None)

    btc_df['date'] = btc_df['date'].dt.strftime('%Y-%-m-%-d')
    coinmetrics_df['date'] = coinmetrics_df['date'].dt.strftime('%Y-%-m-%-d')

    # Merge BTC price data with Coin Metrics data on 'date'
    merged_df = pd.merge(btc_df, coinmetrics_df, on='date', how='inner')

    # Drop 'asset' column
    if 'asset' in merged_df.columns:
        merged_df = merged_df.drop(columns=['asset'])

    # Rename columns
    merged_df = merged_df.rename(columns={
        'CapMrktCurUSD': 'market_cap',
        'CapRealUSD': 'realized_market_cap'
    })

    # Normalize mvrv_z_score to 0–1 scale to get mvrv_risk
    z_min = merged_df['mvrv_z_score'].min()
    z_max = merged_df['mvrv_z_score'].max()
    merged_df['mvrv_risk'] = (merged_df['mvrv_z_score'] - z_min) / (z_max - z_min)
    print(f"MVRV Z-Score Range: min = {z_min:.3f}, max = {z_max:.3f}")

    # Select desired columns in order
    cols = [
        'date', 'open', 'high', 'low', 'close', 'volume',
        'market_cap', 'realized_market_cap',
        'mvrv_ratio', 'mvrv_mean', 'mvrv_std', 'mvrv_z_score', 'mvrv_risk'
    ]
    merged_df = merged_df[cols]

    # Save merged dataframe to parquet cache
    merged_df.to_parquet(filepath)
    print(f"Saved merged MVRV and BTC price data to {filepath}")
    return merged_df

