# utils/data_fetch.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

from utils.data_utils import *
from utils.fetch_fear_and_greed_index_data import *
from utils.fetch_btc_historical import *
from utils.fetch_spy_historical import *
from utils.fetch_vix_historical import *

DATA_DIR = "data"
REQUIRED_PARAMS = ["ticker", "interval"]

def fetch_data_for_strategy(strategy_settings):
    """
    Fetch historical data for a single ticker in strategy_settings.
    Returns a DataFrame with custom attributes 'title' and 'ticker' stored in .attrs.
    """
    # Ensure required parameters are present
    missing_params = [param for param in REQUIRED_PARAMS if param not in strategy_settings]
    if missing_params:
        print(f"ERROR: Missing required strategy_settings parameters: {missing_params}")
        return pd.DataFrame()

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist. Please create it before running.")

    ticker = strategy_settings["ticker"]
    interval = strategy_settings["interval"]
    strategy_title = strategy_settings.get("title", "Untitled Strategy")

    # Optional params
    period = strategy_settings.get("period")
    start = strategy_settings.get("start")
    end = strategy_settings.get("end")  # ✅ new optional param

    # Custom behavior: Crypto Sentiment Strategy
    if strategy_title == "Crypto Sentiment Strategy":
        df = fetch_fear_and_greed_index_data(period=period or "max", interval=interval)
        if df.empty:
            return df
        df.attrs['ticker'] = ticker
        df.attrs['title'] = strategy_title
        return df

    # Custom behavior: VIX Strategy (merge with VIX)
    elif strategy_title == "VIX Strategy":
        vix_df = fetch_vix_historical_data()

        if ticker == "SPY":
            spy_df = fetch_spy_historical_data()
            merged_df = pd.merge(spy_df, vix_df, on="date", how="inner")
            merged_df.attrs['title'] = strategy_title
            merged_df.attrs['ticker'] = ticker
            return merged_df  # ✅ Prevent fall-through

        if ticker == "BTC-USD":
            btc_df = fetch_btc_historical_data()
            merged_df = pd.merge(btc_df, vix_df, on="date", how="inner")
            merged_df.attrs['title'] = strategy_title
            merged_df.attrs['ticker'] = ticker
            return merged_df  # ✅ Prevent fall-through

    # Standard data fetching logic for other strategies
    filename = f"{ticker.replace('-', '_')}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    should_fetch = True

    if os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)

            # Ensure index is datetime or recover from 'date' column
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)

            latest_date = df.index.max().date()
            today = datetime.utcnow().date()

            if end:
                end_date = pd.to_datetime(end).date()
                # only re-fetch if existing data doesn't reach `end`
                if latest_date >= end_date:
                    should_fetch = False
            elif latest_date >= today:
                should_fetch = False

        except Exception as e:
            print(f"Error reading existing data for {ticker}: {e}. Refetching...")

    if should_fetch:
        # Build kwargs dynamically: prefer (start, end) over period
        download_kwargs = {
            "tickers": ticker,
            "interval": interval,
            "auto_adjust": True,
        }
        if start:
            download_kwargs["start"] = start
        if end:
            download_kwargs["end"] = end
        if not start and not end and period:
            download_kwargs["period"] = period
        if not start and not end and not period:
            download_kwargs["period"] = "max"

        df = yf.download(**download_kwargs)

        if df.empty:
            print(f"Warning: No data fetched for {ticker}")
            return pd.DataFrame()

        # Normalize column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]

        # Ensure date column and index
        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df.to_parquet(filepath)
        print(f"Saved fresh {ticker} data to {filepath}")

    # Final validation of datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

    # Attach metadata
    df.attrs['ticker'] = ticker
    df.attrs['title'] = strategy_title

    return df
