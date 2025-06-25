# utils/data_fetch.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils.data_utils import *

DATA_DIR = "data"
REQUIRED_PARAMS = ["tickers", "period", "interval"]


def fetch_data_for_strategy(strategy_settings):
    """
    Fetch historical data for tickers in strategy_settings.
    If the data already exists and is up-to-date, it won't be fetched again.
    
    Returns:
        dict: A dictionary of DataFrames keyed by ticker symbol.
    """
    missing_params = [param for param in REQUIRED_PARAMS if param not in strategy_settings]
    if missing_params:
        print(f"ERROR: Missing required strategy_settings parameters: {missing_params}")
        return {}

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist. Please create it before running.")

    tickers = strategy_settings["tickers"]
    period = strategy_settings["period"]
    interval = strategy_settings["interval"]

    if not tickers or not isinstance(tickers, (list, tuple)):
        raise ValueError(f"'tickers' must be a non-empty list or tuple, got: {tickers}")

    data_dict = {}

    for ticker in tickers:
        print(f"Checking cached data for {ticker}...")

        filename = f"{ticker.replace('-', '_')}.parquet"
        filepath = os.path.join(DATA_DIR, filename)

        should_fetch = True

        if os.path.exists(filepath):
            try:
                existing_df = load_parquet(filepath)
                latest_date = existing_df.index.max().date()
                today = datetime.utcnow().date()

                if latest_date >= today:
                    print(f"Cached data for {ticker} is up-to-date (latest: {latest_date}). Skipping fetch.")
                    data_dict[ticker] = existing_df
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
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]

            save_parquet(df, filepath)
            print(f"Saved fresh {ticker} data to {filepath}")
            data_dict[ticker] = df

    return data_dict