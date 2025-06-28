# utils/data_fetch.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils.data_utils import *

DATA_DIR = "data"
REQUIRED_PARAMS = ["ticker", "period", "interval"]

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

    if os.path.exists(filepath):
        try:
            df = load_parquet(filepath)
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

        save_parquet(df, filepath)
        print(f"Saved fresh {ticker} data to {filepath}")

    # Set custom attributes (no ticker column)
    df.ticker = ticker
    df.title = strategy_title

    return df
