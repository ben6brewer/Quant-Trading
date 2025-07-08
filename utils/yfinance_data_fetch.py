# utils/data_fetch.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from utils.data_utils import *
from utils.cmc_data_fetch import *
from utils.fetch_btc_historical import *
from utils.fetch_spy_historical import *
from utils.fetch_vix_historical import *

DATA_DIR = "data"
REQUIRED_PARAMS = ["ticker", "period", "interval"]

def fetch_data_for_strategy(strategy_settings):
    """
    Fetch historical data for a single ticker in strategy_settings.
    Returns a DataFrame with custom attributes 'title' and 'ticker' stored in .attrs.
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

    # Handle custom behavior for Crypto Sentiment Strategy
    if strategy_title == "Crypto Sentiment Strategy":
        df = fetch_fear_and_greed_index(period=period, interval=interval)
        if df.empty:
            return df
        df.attrs['ticker'] = ticker
        df.attrs['title'] = strategy_title
        return df

    elif strategy_title == "VIX Strategy":
        vix_df = fetch_vix_historical_data()
        if ticker == "SPY":
            spy_df = fetch_spy_historical_data()
            merged_df = pd.merge(spy_df, vix_df, on="date", how="inner")
            merged_df.attrs['title'] = strategy_title
            merged_df.attrs['ticker'] = ticker
            return merged_df

        if ticker == "BTC-USD":
            btc_df = fetch_btc_historical_data()
            merged_df = pd.merge(btc_df, vix_df, on="date", how="inner")
            merged_df.attrs['title'] = strategy_title
            merged_df.attrs['ticker'] = ticker
            return merged_df

    # Standard logic for other strategies
    filename = f"{ticker.replace('-', '_')}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    should_fetch = True

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

        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df.to_parquet(filepath)
        print(f"Saved fresh {ticker} data to {filepath}")

    else:
        # When loading cached data:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    print(f"Warning: Could not convert index to datetime for {ticker}: {e}")

    # Set metadata via .attrs
    df.attrs['ticker'] = ticker
    df.attrs['title'] = strategy_title

    return df
