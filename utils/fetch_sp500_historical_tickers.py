import os
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

# FMP_API_KEY = "aAVTkNEGAeplhCtKxzZdKDyjabzDwBE0"
FMP_API_KEY = "U6orNhvNRMN7i4mANJe4clvzg6tjQtJa"
DATA_DIR = "data/sp500"
DAILY_LISTS_PATH = "data/sp500_daily_lists.parquet"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# List of TTM metrics to fetch from the /ratios-ttm endpoint
TTM_METRICS = [
    "peRatioTTM", "priceToSalesRatioTTM", "pbRatioTTM",
    "operatingProfitMarginTTM", "netProfitMarginTTM", "returnOnEquityTTM",
    "returnOnAssetsTTM", "debtEquityRatioTTM", "currentRatioTTM",
    "quickRatioTTM", "grossProfitMarginTTM", "epsTTM", "revenuePerShareTTM"
]

def fetch_sp500_historical_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    tables = pd.read_html(html)

    current_table = tables[0]
    current_symbols = set(current_table["Symbol"].tolist())

    changes_table = tables[1]
    changes_table.columns = ['_'.join(col).strip() for col in changes_table.columns.values]
    changes_table = changes_table.rename(columns={
        'Date_Date': 'date',
        'Added_Ticker': 'added',
        'Removed_Ticker': 'removed'
    })
    changes_table['date'] = pd.to_datetime(changes_table['date'], errors='coerce')
    changes_table = changes_table.sort_values('date')

    ticker_set = current_symbols.copy()
    for _, row in changes_table[::-1].iterrows():
        if pd.notna(row['added']):
            ticker_set.discard(row['added'])
        if pd.notna(row['removed']):
            ticker_set.add(row['removed'])

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
        date += pd.Timedelta(days=1)

    df = pd.DataFrame(list(daily_membership.items()), columns=['date', 'tickers'])
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(DAILY_LISTS_PATH, index=False)
    print(f"✅ Saved daily ticker membership to {DAILY_LISTS_PATH}")
    return df

def fetch_and_cache_ticker_data(ticker):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{ticker}.parquet")

    # Try FMP price data first
    price_url = f"{FMP_BASE_URL}/historical-price-full/{ticker}?serietype=line&timeseries=365&apikey={FMP_API_KEY}"
    price_resp = requests.get(price_url)
    df_prices = None

    if price_resp.status_code == 200:
        price_json = price_resp.json()
        if 'historical' in price_json and price_json['historical']:
            df_prices = pd.DataFrame(price_json['historical'])
            df_prices['date'] = pd.to_datetime(df_prices['date'])

            # Check expected columns exist before selecting
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if all(col in df_prices.columns for col in required_cols):
                df_prices = df_prices[required_cols]
            else:
                print(f"{ticker}: Missing expected price columns from FMP, falling back to yfinance")
                df_prices = None
        else:
            print(f"{ticker}: No historical price data from FMP, falling back to yfinance")
    else:
        print(f"{ticker}: Failed to fetch price data from FMP, status {price_resp.status_code}, falling back to yfinance")

    # If FMP failed or missing columns, fallback to yfinance
    if df_prices is None:
        try:
            yf_data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
            if yf_data.empty:
                print(f"{ticker}: yfinance returned empty data")
                return None

            print(f"{ticker}: Raw yfinance columns before processing: {yf_data.columns}")

            # Handle MultiIndex columns by flattening to first level (price fields)
            if isinstance(yf_data.columns, pd.MultiIndex):
                yf_data.columns = yf_data.columns.get_level_values(0)
                print(f"{ticker}: Flattened MultiIndex columns in yfinance data")

            yf_data.reset_index(inplace=True)

            # Convert columns to strings and lowercase to normalize
            yf_data.columns = [str(col).lower() for col in yf_data.columns]

            print(f"{ticker}: Processed yfinance columns after lowering: {yf_data.columns}")
            print(f"{ticker}: Sample yfinance data rows:\n{yf_data.head()}")

            expected_yf_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = set(expected_yf_cols) - set(yf_data.columns)
            if missing_cols:
                print(f"{ticker}: Missing columns in yfinance data: {missing_cols}")
                return None

            df_prices = yf_data[expected_yf_cols]

        except Exception as e:
            print(f"{ticker}: Failed to fetch price data from yfinance: {e}")
            return None


    # Fetch TTM ratios from FMP
    ttm_url = f"{FMP_BASE_URL}/ratios-ttm/{ticker}?apikey={FMP_API_KEY}"
    ttm_resp = requests.get(ttm_url)
    if ttm_resp.status_code != 200:
        print(f"{ticker}: Failed to fetch TTM ratios, status {ttm_resp.status_code}")
        return None

    ttm_json = ttm_resp.json()
    if not ttm_json or not isinstance(ttm_json, dict):
        print(f"{ticker}: No TTM data")
        return None

    # Add TTM metrics to price df, using None if missing
    for metric in TTM_METRICS:
        df_prices[metric] = ttm_json.get(metric, None)

    df_prices.to_parquet(filepath, index=False)
    print(f"{ticker}: Saved {len(df_prices)} rows to {filepath}")
    return df_prices



def update_all_sp500_tickers(start_year=2024):
    df_daily = fetch_sp500_historical_tickers()
    
    # Filter df_daily to only dates from start_year onward
    df_daily = df_daily[df_daily['date'].dt.year >= start_year]
    
    df_daily['year'] = df_daily['date'].dt.year
    print(f"📆 Processing years {df_daily['year'].min()} to {df_daily['year'].max()}")

    tickers_to_remove = set()

    for year, group in df_daily.groupby('year'):
        print(f"\n=== Year {year} ===")
        tickers = set(t for tlist in group['tickers'] for t in tlist)
        print(f"Found {len(tickers)} tickers")

        for ticker in tickers:
            try:
                result = fetch_and_cache_ticker_data(ticker)
                if result is None:
                    tickers_to_remove.add(ticker)
            except Exception as e:
                print(f"Error with {ticker}: {e}")
                tickers_to_remove.add(ticker)

    if tickers_to_remove:
        print(f"🚫 Removing {len(tickers_to_remove)} tickers due to data errors")
        df_daily['tickers'] = df_daily['tickers'].apply(lambda lst: [t for t in lst if t not in tickers_to_remove])
        df_daily.to_parquet(DAILY_LISTS_PATH, index=False)
        print(f"✅ Updated daily membership saved to {DAILY_LISTS_PATH}")

def fetch_and_cache_aapl_data():
    ticker = "NVDA"
    filepath = os.path.join(DATA_DIR, f"{ticker}.parquet")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Fetch price data from yfinance
    try:
        yf_data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if yf_data.empty:
            print(f"{ticker}: yfinance returned empty data")
            return None

        print(f"{ticker}: Raw yfinance columns before processing: {yf_data.columns}")

        # Flatten MultiIndex columns if present
        if isinstance(yf_data.columns, pd.MultiIndex):
            print(f"{ticker}: Flattening MultiIndex columns in yfinance data")
            yf_data.columns = ['_'.join(col).strip().lower() for col in yf_data.columns.values]
        else:
            yf_data.columns = [str(col).lower() for col in yf_data.columns]

        yf_data.reset_index(inplace=True)

        # Rename columns to remove ticker suffixes and normalize 'date'
        new_cols = {}
        for col in yf_data.columns:
            if '_' in col:
                base_col = col.split('_')[0]
                new_cols[col] = base_col
            elif col.lower() == 'date':
                new_cols[col] = 'date'
            else:
                new_cols[col] = col.lower()
        yf_data.rename(columns=new_cols, inplace=True)

        print(f"{ticker}: Columns after removing suffixes and normalizing: {yf_data.columns}")
        print(f"{ticker}: Sample yfinance data rows:\n{yf_data.head()}")

        expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(expected_cols) - set(yf_data.columns)
        if missing_cols:
            print(f"{ticker}: Missing columns in yfinance data after suffix removal: {missing_cols}")
            return None

        df_prices = yf_data[expected_cols]

    except Exception as e:
        print(f"{ticker}: Failed to fetch price data from yfinance: {e}")
        return None

    # Fetch TTM ratios from FMP
    ttm_url = f"{FMP_BASE_URL}/ratios-ttm/{ticker}?apikey={FMP_API_KEY}"
    ttm_resp = requests.get(ttm_url)
    if ttm_resp.status_code != 200:
        print(f"{ticker}: Failed to fetch TTM ratios, status {ttm_resp.status_code}")
        return None

    ttm_json = ttm_resp.json()
    if not ttm_json or not isinstance(ttm_json, dict):
        print(f"{ticker}: No TTM data")
        return None

    # Add TTM metrics to price df, using None if missing
    for metric in TTM_METRICS:
        df_prices[metric] = ttm_json.get(metric, None)

    df_prices.to_parquet(filepath, index=False)
    print(f"{ticker}: Saved {len(df_prices)} rows to {filepath}")
    return df_prices