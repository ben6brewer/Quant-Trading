import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = "data"

def get_last_market_day(today=None):
    """Return the most recent weekday before or equal to today."""
    today = today or datetime.utcnow().date()
    while today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        today -= timedelta(days=1)
    return today

def fetch_spy_historical_data(period="max", interval="1d", filepath=os.path.join(DATA_DIR, "SPY.parquet")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)

            if 'date' not in df.columns and df.index.name == 'date':
                df = df.reset_index()

            if 'date' not in df.columns:
                raise ValueError("Cached SPY data missing 'date' column")

            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max().date()
            last_market_day = get_last_market_day()

            if latest_date >= last_market_day:
                print(f"Cached SPY data is up-to-date (latest: {latest_date}).")
                return df
            else:
                print(f"Cached SPY data outdated (latest: {latest_date} < {last_market_day}). Fetching new data...")
        except Exception as e:
            print(f"Error loading cached SPY data: {e}. Refetching...")
    else:
        print("No cached SPY data found. Fetching fresh data...")

    df = yf.download("SPY", period=period, interval=interval, auto_adjust=True)
    if df.empty:
        print("Warning: No SPY data fetched")
        return pd.DataFrame()

    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    df = df.reset_index()

    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    elif 'date' not in df.columns:
        df['date'] = df.index

    df['date'] = pd.to_datetime(df['date'])
    df = df[['date'] + [col for col in df.columns if col != 'date']]
    df.to_parquet(filepath, index=False)
    print(f"Saved fresh SPY data to {filepath}")

    return df
