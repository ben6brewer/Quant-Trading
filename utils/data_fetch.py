# utils/data_fetch.py

import os
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
REQUIRED_PARAMS = ["tickers", "period", "interval"]

def fetch_data_for_strategy(strategy_settings):
    """
    Fetch historical data for tickers in strategy_settings.
    Validates required parameters and handles common issues.
    
    Returns:
        dict: A dictionary of DataFrames keyed by ticker symbol.
    """
    missing_params = [param for param in REQUIRED_PARAMS if param not in strategy_settings]
    if missing_params:
        print(f"ERROR: Missing required strategy_settings parameters: {missing_params}")
        return {}

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist. Please create it before running.")

    try:
        tickers = strategy_settings["tickers"]
        period = strategy_settings["period"]
        interval = strategy_settings["interval"]

        if not tickers or not isinstance(tickers, (list, tuple)):
            raise ValueError(f"'tickers' must be a non-empty list or tuple, got: {tickers}")

        data_dict = {}

        for ticker in tickers:
            print(f"Fetching {ticker} data (period={period}, interval={interval})...")
            df = yf.download(ticker, period=period, interval=interval)

            if df.empty:
                print(f"Warning: No data fetched for {ticker}")
                continue

            # Flatten MultiIndex columns and lowercase all column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]

            filepath = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.parquet")
            df.to_parquet(filepath)
            print(f"Saved {ticker} data to {filepath}")

            data_dict[ticker] = df

        return data_dict

    except Exception as e:
        print(f"Error in fetch_data_for_strategy: {e}")
        return {}
