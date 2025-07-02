import os
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = "data"

def fetch_spy_historical_data(period="max", interval="1d", filepath=os.path.join(DATA_DIR, "SPY.parquet")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            latest_date = df['date'].max()
            today = pd.to_datetime('today').normalize().date()

            if latest_date >= today:
                print(f"Cached SPY data is up-to-date (latest: {latest_date}).")
                return df
            else:
                print(f"Cached SPY data outdated (latest: {latest_date}). Fetching new data...")
        except Exception as e:
            print(f"Error loading cached SPY data: {e}. Refetching...")
    else:
        print("No cached SPY data found. Fetching fresh data...")

    df = yf.download("SPY", period=period, interval=interval)
    if df.empty:
        print("Warning: No SPY data fetched")
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
    print(f"Saved fresh SPY data to {filepath}")

    return df
