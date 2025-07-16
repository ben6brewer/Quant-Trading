# utils/fetch_sp500_historical_tickers.py
import os
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta, timezone
import yfinance as yf

from utils.pretty_print_df import *

# List of fundamental metrics to fetch from ticker.info
FUNDAMENTAL_METRICS = [
    'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseValue', 
    'marketCap', 'beta', 'dividendYield', 'dividendRate', 'earningsQuarterlyGrowth', 'revenueGrowth', 
    'profitMargins', 'grossMargins', 'operatingMargins', 'returnOnAssets', 'returnOnEquity', 
    'sharesOutstanding', 'floatShares', 'averageVolume', '52WeekHigh', '52WeekLow', 'shortRatio', 
    'shortPercentOfFloat', 'earningsGrowth', 'pegRatio', 'priceHint', 'exchange', 'sector', 
    'industry', 'fullTimeEmployees', 'longBusinessSummary'
]

DATA_DIR = "data/sp500"
DAILY_LISTS_PATH = "data/sp500_daily_lists.parquet"

def fetch_sp500_historical_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text

    tables = pd.read_html(StringIO(html))
    
    # Current constituents
    current_table = tables[0]
    current_symbols = set(current_table["Symbol"].tolist())

    # Historical changes (flatten multi-index columns)
    changes_table = tables[1]
    changes_table.columns = ['_'.join(col).strip() for col in changes_table.columns.values]
    changes_table = changes_table.rename(columns={
        'Date_Date': 'date',
        'Added_Ticker': 'added',
        'Removed_Ticker': 'removed'
    })
    changes_table['date'] = pd.to_datetime(changes_table['date'], errors='coerce')
    changes_table = changes_table.sort_values('date')

    # Rewind ticker set to earliest date by reversing changes
    ticker_set = current_symbols.copy()
    for _, row in changes_table[::-1].iterrows():
        if pd.notna(row['added']):
            ticker_set.discard(row['added'])
        if pd.notna(row['removed']):
            ticker_set.add(row['removed'])

    # Forward apply daily changes
    start_date = changes_table['date'].min()
    end_date = pd.Timestamp.today()
    date = start_date

    change_map = changes_table.groupby('date').apply(lambda df: df.to_dict('records')).to_dict()

    daily_membership = {}
    current_set = ticker_set.copy()

    while date <= end_date:
        if date in change_map:
            for change in change_map[date]:
                if pd.notna(change['added']):
                    current_set.add(change['added'])
                if pd.notna(change['removed']):
                    current_set.discard(change['removed'])
        daily_membership[date.strftime('%Y-%m-%d')] = sorted(current_set)
        date += timedelta(days=1)

    df = pd.DataFrame(list(daily_membership.items()), columns=['date', 'tickers'])
    df['date'] = pd.to_datetime(df['date'])

    df.to_parquet(DAILY_LISTS_PATH, index=False)
    print(f"✅ Saved daily ticker membership to {DAILY_LISTS_PATH}")
    return df

def fetch_and_cache_ticker_data(ticker):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{ticker}.parquet")

    if os.path.exists(filepath):
        df_existing = pd.read_parquet(filepath)
        last_date = df_existing['date'].max()
        today = datetime.now(timezone.utc).date()
        if last_date.date() >= today:
            print(f"{ticker}: Data up-to-date up to {last_date.date()}")
            return df_existing
        start_date = last_date + timedelta(days=1)
        print(f"{ticker}: Updating data from {start_date} to today")
    else:
        df_existing = None
        start_date = None

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yf_ticker = yf.Ticker(ticker)
    try:
        if start_date:
            hist = yf_ticker.history(start=start_date.strftime("%Y-%m-%d"), end=today_str, auto_adjust=True)
        else:
            hist = yf_ticker.history(start="1900-01-01", end=today_str, auto_adjust=True)
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return None

    if hist.empty:
        print(f"{ticker}: No historical price data fetched — skipping save")
        return None

    hist = hist.reset_index()
    hist['date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
    price_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    hist = hist[price_cols]

    info = yf_ticker.info
    for metric in FUNDAMENTAL_METRICS:
        hist[metric] = info.get(metric, None)

    if df_existing is not None:
        df_existing = df_existing[df_existing['date'] < hist['date'].min()]
        df_combined = pd.concat([df_existing, hist], ignore_index=True)
    else:
        df_combined = hist

    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    df_combined.to_parquet(filepath, index=False)
    print(f"{ticker}: Saved data with {len(df_combined)} rows to {filepath}")
    return df_combined

def update_all_sp500_tickers():
    df_daily = fetch_sp500_historical_tickers()
    df_daily['year'] = df_daily['date'].dt.year

    print(f"Updating data for years {df_daily['year'].min()} to {df_daily['year'].max()}")

    # Keep track of tickers to remove if fetching fails
    tickers_to_remove = set()

    for year, group in df_daily.groupby('year'):
        print(f"\n=== Processing year: {year} ===")
        tickers_in_year = set()
        for tickers_list in group['tickers']:
            tickers_in_year.update(tickers_list)

        print(f"Found {len(tickers_in_year)} unique tickers in {year}")

        for ticker in tickers_in_year:
            try:
                result = fetch_and_cache_ticker_data(ticker)
                if result is None:
                    print(f"Warning: Removing {ticker} due to missing data")
                    tickers_to_remove.add(ticker)
            except Exception as e:
                print(f"Error processing {ticker} in year {year}: {e}")
                tickers_to_remove.add(ticker)

    if tickers_to_remove:
        print(f"\nRemoving {len(tickers_to_remove)} tickers from daily membership list due to missing data.")
        # Remove these tickers from all daily membership entries
        def filter_tickers(tickers):
            return [t for t in tickers if t not in tickers_to_remove]

        df_daily['tickers'] = df_daily['tickers'].apply(filter_tickers)
        df_daily.to_parquet(DAILY_LISTS_PATH, index=False)
        print(f"✅ Updated daily membership list saved to {DAILY_LISTS_PATH}")

