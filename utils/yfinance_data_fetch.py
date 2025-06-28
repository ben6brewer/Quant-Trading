# utils/data_fetch.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils.data_utils import *

DATA_DIR = "data"
REQUIRED_PARAMS = ["ticker", "period", "interval"]

def fetch_btc_historical_data(period, interval, filepath):
    """
    Fetch BTC-USD historical data, add date column,
    save to parquet if needed, and return the DataFrame.
    """

    # Check if cached exists
    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            latest_date = df['date'].max()
            today = pd.to_datetime('today').normalize()

            if latest_date >= today:
                print(f"Cached BTC-USD data up-to-date (latest: {latest_date.date()}).")
                return df
            else:
                print(f"Cached BTC-USD data outdated (latest: {latest_date.date()}). Refetching...")
        except Exception as e:
            print(f"Error loading BTC cached data: {e}. Refetching...")

    # Fetch fresh data
    df = yf.download("BTC-USD", period=period, interval=interval)
    if df.empty:
        print("Warning: No BTC-USD data fetched")
        return pd.DataFrame()

    # Normalize column names to lowercase
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    # Reset index to add date column
    df = df.reset_index()
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Save to parquet
    df.to_parquet(filepath)
    print(f"Saved fresh BTC-USD data to {filepath}")

    return df


def fetch_data_for_strategy(strategy_settings):
    """
    Fetch historical data for a single ticker in strategy_settings.
    If the data already exists and is up-to-date, it won't be fetched again.

    Returns:
        pd.DataFrame: A DataFrame containing the ticker's data,
                      with custom attributes 'title' and 'ticker_symbol'.
    """

    missing_params = [param for param in REQUIRED_PARAMS if param not in strategy_settings]
    if missing_params:
        print(f"ERROR: Missing required strategy_settings parameters: {missing_params}")
        return pd.DataFrame()

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist. Please create it before running.")

    ticker = strategy_settings["ticker"]
    period = strategy_settings["period"]
    interval = strategy_settings["interval"]
    strategy_title = strategy_settings.get("title", "Untitled Strategy")

    print(f"Checking cached data for {ticker}...")
    filename = f"{ticker.replace('-', '_')}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    should_fetch = True

    if ticker == "BTC-USD":
        # Use BTC helper to fetch data
        df = fetch_btc_historical_data(period, interval, filepath)
        if df.empty:
            return df
        should_fetch = False  # Already handled in helper

    else:
        if os.path.exists(filepath):
            try:
                df = pd.read_parquet(filepath)
                latest_date = df.index.max().date()
                today = datetime.utcnow().date()

                if latest_date >= today:
                    print(f"Cached data for {ticker} is up-to-date (latest: {latest_date}). Skipping fetch.")
                    should_fetch = False
                else:
                    print(f"Cached data for {ticker} is outdated (latest: {latest_date}). Fetching new data...")
            except Exception as e:
                print(f"Error reading existing data for {ticker}: {e}. Refetching...")

        if should_fetch:
            print(f"Fetching {ticker} data (period={period}, interval={interval})...")
            df = yf.download(ticker, period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data fetched for {ticker}")
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]

            df.to_parquet(filepath)
            print(f"Saved fresh {ticker} data to {filepath}")

        # Add date column by resetting index if not done already
        if not should_fetch:
            # df already from helper with date col, skip this
            pass
        else:
            df = df.reset_index()
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.date

    # Set custom attributes
    df.ticker = ticker
    df.title = strategy_title

    return df
