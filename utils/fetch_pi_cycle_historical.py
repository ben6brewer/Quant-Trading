# utils/fetch_pi_cycle_historical.py

import os
import pandas as pd
from datetime import datetime, timedelta
from utils.fetch_btc_historical import fetch_btc_historical_data
from utils.pretty_print_df import pretty_print_df

def fetch_pi_cycle_historical_data(filepath="data/pi_cycle_btc.parquet"):
    btc_df = fetch_btc_historical_data()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Prepare date columns without timezone
    btc_df['date'] = pd.to_datetime(btc_df['date']).dt.tz_localize(None)

    if os.path.exists(filepath):
        try:
            cached_df = pd.read_parquet(filepath)
            cached_df['date'] = pd.to_datetime(cached_df['date']).dt.tz_localize(None)

            latest_date = cached_df['date'].max().date()
            today_date = datetime.utcnow().date()

            if latest_date >= (today_date - timedelta(days=1)):
                print(f"Cached Pi Cycle data is up-to-date (latest: {latest_date}). Using cached data.")

                merged_df = pd.merge(
                    btc_df,
                    cached_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore'),
                    on='date',
                    how='left'
                )

                # Forward fill SMAs and Pi Cycle ratio
                for col in ['sma_350d', '2(sma_111d)', 'pi_cycle_risk']:
                    if col in merged_df.columns:
                        merged_df[col] = merged_df[col].ffill()

                # Select relevant columns
                cols = [
                    'date', 'open', 'high', 'low', 'close', 'volume',
                    'sma_350d', '2(sma_111d)', 'pi_cycle_risk'
                ]
                return merged_df[cols]

            else:
                print(f"Cached Pi Cycle data outdated (latest: {latest_date}). Fetching fresh data...")

        except Exception as e:
            print(f"Error loading cached Pi Cycle data: {e}. Refetching...")

    btc_df = btc_df.sort_values('date').reset_index(drop=True)

    # Calculate SMAs
    btc_df['sma_350d'] = btc_df['close'].rolling(window=350).mean()
    btc_df['2(sma_111d)'] = btc_df['close'].rolling(window=111).mean()

    # Multiply the 111-day SMA by 2 as requested
    btc_df['2(sma_111d))'] = btc_df['2(sma_111d)'] * 2

    # Calculate pi_cycle_risk as (2 * sma_111d)) / sma_350d ratio
    btc_df['pi_cycle_risk'] = btc_df['2(sma_111d)'] / btc_df['sma_350d']

    # Forward fill SMAs and ratio for missing values
    btc_df['sma_350d'] = btc_df['sma_350d'].ffill()
    btc_df['2(sma_111d)'] = btc_df['2(sma_111d)'].ffill()
    btc_df['pi_cycle_risk'] = btc_df['pi_cycle_risk'].ffill()

    # Save to cache
    btc_df.to_parquet(filepath)
    print(f"Saved Pi Cycle data with SMAs to {filepath}")

    cols = [
        'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_350d', '2(sma_111d)', 'pi_cycle_risk'
    ]
        # Filter and print rows starting from 2021-04-01
    filtered_df = btc_df[btc_df['date'] >= '2021-06-01']
    pretty_print_df(filtered_df.head(20))
    return btc_df[cols]
