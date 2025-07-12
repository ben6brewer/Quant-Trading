# utils/cmc_data_fetch.py

from utils.pretty_print_df import *
from utils.fetch_btc_historical import *

import yfinance as yf
import os
import pandas as pd
import requests
import numpy as np
from datetime import datetime

DATA_DIR = "data"

def get_cmc_fear_greed_data():
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        response = requests.get(url)
        data = response.json()
        if "data" in data:
            fng_data = data['data']
            fng_df = pd.DataFrame(fng_data)
            fng_df['date'] = pd.to_datetime(fng_df['timestamp'].astype(int), unit='s').dt.date
            fng_df['F&G'] = fng_df['value'].astype(int)
            return fng_df[['date', 'F&G']]
        else:
            print("No data found in response.")
            return pd.DataFrame()
    except Exception as e:
        print("Error fetching Fear and Greed data:", e)
        return pd.DataFrame()


def fetch_fear_and_greed_index_data(period="max", interval="1d"):
    btc_filepath = os.path.join(DATA_DIR, "BTC_USD.parquet")
    combined_filepath = os.path.join(DATA_DIR, "F&G_BTC_USD.parquet")

    # Fetch BTC historical price data (already cached in BTC_USD.parquet)
    df = fetch_btc_historical_data(period=period, interval=interval, filepath=btc_filepath)
    fng_df = get_cmc_fear_greed_data()

    if df.empty or fng_df.empty:
        print("One or both data sources are empty, cannot merge.")
        return df

    # Ensure datetime dtype
    df['date'] = pd.to_datetime(df['date'])
    fng_df['date'] = pd.to_datetime(fng_df['date'])

    # Merge
    merged_df = pd.merge(df, fng_df, on='date', how='left')
    merged_df = merged_df.dropna(subset=['F&G'])

    # Normalize F&G Index
    fg = merged_df['F&G'].astype(float)
    r_min, r_max = fg.min(), fg.max()
    r_mean, r_std = fg.mean(), fg.std()
    epsilon = 1e-9
    shifted = fg - r_min + epsilon
    log_vals = np.log(shifted)
    lmin, lmax = log_vals.min(), log_vals.max()

    merged_df['min_max_norm'] = (fg - r_min) / (r_max - r_min)
    merged_df['z_score_norm'] = (fg - r_mean) / r_std
    merged_df['sigmoid_norm'] = 1 / (1 + np.exp(-merged_df['z_score_norm']))
    merged_df['log_norm'] = (log_vals - lmin) / (lmax - lmin)

    # Set F&G risk to be log_norm (override or redefine if desired)
    merged_df['F&G_risk'] = merged_df['sigmoid_norm']  # or 'sigmoid_norm' or 'log_norm'

    # Set datetime index
    merged_df.set_index('date', inplace=True)

    # Save to file
    merged_df.to_parquet(combined_filepath)

    # Return with date reset as column
    return merged_df.reset_index()
