# utils.cmc_data_fetch.py

from utils.pretty_print_df import *
from utils.fetch_btc_historical import *

import yfinance as yf
import os
import pandas as pd
import requests
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


def fetch_fear_and_greed_index(period="max", interval="1d"):
    btc_filepath = os.path.join(DATA_DIR, "BTC_USD.parquet")
    combined_filepath = os.path.join(DATA_DIR, "F&G_BTC_USD.parquet")

    # Fetch BTC historical price data (already cached in BTC_USD.parquet)
    df = fetch_btc_historical_data(period=period, interval=interval, filepath=btc_filepath)
    fng_df = get_cmc_fear_greed_data()

    if df.empty or fng_df.empty:
        print("One or both data sources are empty, cannot merge.")
        return df

    # Convert and merge
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    if not pd.api.types.is_datetime64_any_dtype(fng_df['date']):
        fng_df['date'] = pd.to_datetime(fng_df['date'])

    merged_df = pd.merge(df, fng_df, on='date', how='left')
    merged_df = merged_df.dropna(subset=['F&G'])

    # Set datetime index
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df.set_index('date', inplace=True)

    merged_df.to_parquet(combined_filepath)

    return merged_df
