# utils/fetch_pi_cycle_historical.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.fetch_btc_historical import fetch_btc_historical_data
from utils.pretty_print_df import pretty_print_df

def fetch_pi_cycle_historical_data(filepath="data/pi_cycle_btc.parquet"):
    btc_df = fetch_btc_historical_data()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

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

                for col in [
                    'sma_350d', 'sma_111d_doubled', 'pi_cycle_ratio',
                    'min_max_norm', 'z_score_norm', 'sigmoid_norm', 'log_norm'
                ]:
                    if col in merged_df.columns:
                        merged_df[col] = merged_df[col].ffill()

                # ✅ Hardcoded choice of risk metric
                merged_df['pi_cycle_risk'] = merged_df['log_norm']

                pretty_print_df(merged_df.tail(10), title="Pi Cycle BTC Data with Risk Metrics")

                return merged_df[[
                    'date', 'open', 'high', 'low', 'close', 'volume',
                    'sma_350d', 'sma_111d_doubled', 'pi_cycle_ratio',
                    'min_max_norm', 'z_score_norm', 'sigmoid_norm', 'log_norm', 'pi_cycle_risk'
                ]]

        except Exception as e:
            print(f"Error loading cached Pi Cycle data: {e}. Refetching...")

    # Sort and calculate SMAs
    btc_df = btc_df.sort_values('date').reset_index(drop=True)
    sma_111 = btc_df['close'].rolling(window=111).mean()
    sma_350 = btc_df['close'].rolling(window=350).mean()

    btc_df['sma_111d_doubled'] = sma_111 * 2
    btc_df['sma_350d'] = sma_350

    # Pi Cycle Ratio
    btc_df['pi_cycle_ratio'] = btc_df['sma_111d_doubled'] / btc_df['sma_350d']

    # --- Normalize multiple ways ---
    ratio_min = btc_df['pi_cycle_ratio'].min()
    ratio_max = btc_df['pi_cycle_ratio'].max()
    ratio_mean = btc_df['pi_cycle_ratio'].mean()
    ratio_std = btc_df['pi_cycle_ratio'].std()

    btc_df['min_max_norm'] = (btc_df['pi_cycle_ratio'] - ratio_min) / (ratio_max - ratio_min)
    btc_df['z_score_norm'] = (btc_df['pi_cycle_ratio'] - ratio_mean) / ratio_std
    btc_df['sigmoid_norm'] = 1 / (1 + np.exp(-btc_df['z_score_norm']))

    # ✅ Log Normalization (scaled to 0-1)
    epsilon = 1e-9
    log_values = np.log(btc_df['pi_cycle_ratio'] + epsilon)
    log_min = log_values.min()
    log_max = log_values.max()
    btc_df['log_norm'] = (log_values - log_min) / (log_max - log_min)

    # ✅ Hardcoded choice of risk metric
    btc_df['pi_cycle_risk'] = btc_df['min_max_norm']  # Using min-max norm as risk metric

    # Forward fill any missing values
    for col in [
        'sma_350d', 'sma_111d_doubled', 'pi_cycle_ratio',
        'min_max_norm', 'z_score_norm', 'sigmoid_norm', 'log_norm', 'pi_cycle_risk'
    ]:
        btc_df[col] = btc_df[col].ffill()

    # Save to cache
    btc_df.to_parquet(filepath)
    print(f"Saved Pi Cycle data with SMAs and normalized risk to {filepath}")

    return btc_df[[
        'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_350d', 'sma_111d_doubled', 'pi_cycle_ratio',
        'min_max_norm', 'z_score_norm', 'sigmoid_norm', 'log_norm', 'pi_cycle_risk'
    ]]
