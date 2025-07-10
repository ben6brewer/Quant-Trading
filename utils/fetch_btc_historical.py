#utils/fetch_btc_historical.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = "data"

def fetch_btc_historical_data(period="max", interval="1d", filepath=os.path.join(DATA_DIR, "BTC_USD.parquet")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            latest_date = df['date'].max()
            today = pd.to_datetime('today').normalize()

            if latest_date >= today:
                print(f"Cached BTC data is up-to-date (latest: {latest_date}).")
                return df
            else:
                print(f"Cached BTC data outdated (latest: {latest_date}). Fetching new data...")
        except Exception as e:
            print(f"Error loading cached BTC data: {e}. Refetching...")
    else:
        print("No cached BTC data found. Fetching fresh data...")

    df = yf.download("BTC-USD", period=period, interval=interval, auto_adjust=True)
    if df.empty:
        print("Warning: No BTC data fetched")
        return pd.DataFrame()

    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    df = df.reset_index()
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])

    df.to_parquet(filepath)
    print(f"Saved fresh BTC data to {filepath}")

    return df
