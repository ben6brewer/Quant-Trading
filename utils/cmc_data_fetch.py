# utils.cmc_data_fetch.py

from utils.pretty_print_df import *

import yfinance as yf
import os
import pandas as pd
import requests
from datetime import datetime

DATA_DIR = "data"


def fetch_btc_historical_data(period="max", interval="1d", filepath=os.path.join(DATA_DIR, "F&G_BTC_USD.parquet")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            latest_date = df['date'].max()
            today = pd.to_datetime('today').normalize().date()

            if latest_date >= today:
                print(f"Cached BTC data is up-to-date (latest: {latest_date}).")
                return df
            else:
                print(f"Cached BTC data outdated (latest: {latest_date}). Fetching new data...")
        except Exception as e:
            print(f"Error loading cached BTC data: {e}. Refetching...")
    else:
        print("No cached BTC data found. Fetching fresh data...")

    df = yf.download("BTC-USD", period=period, interval=interval)
    if df.empty:
        print("Warning: No BTC data fetched")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df = df.reset_index()
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.date

    df.to_parquet(filepath)
    print(f"Saved fresh BTC data to {filepath}")

    return df



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


def fetch_fear_and_greed_index():
    df = fetch_btc_historical_data()

    fng_df = get_cmc_fear_greed_data()

    if df.empty or fng_df.empty:
        print("One or both data sources are empty, cannot merge.")
        return df

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date']).dt.date
    if not pd.api.types.is_datetime64_any_dtype(fng_df['date']):
        fng_df['date'] = pd.to_datetime(fng_df['date']).dt.date

    merged_df = pd.merge(df, fng_df, on='date', how='left')

    merged_df = merged_df.dropna(subset=['F&G'])

    return merged_df


pretty_print_df(fetch_fear_and_greed_index().head())