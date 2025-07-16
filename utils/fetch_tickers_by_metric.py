import os
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

SP500_DAILY_LIST_PATH = "data/sp500_daily_lists.parquet"
SP500_DATA_DIR = "data/sp500"

def fetch_top_tickers_by_metric(num_tickers_to_fetch, metric, time_period):
    return _fetch_extreme_tickers_by_metric(num_tickers_to_fetch, metric, time_period, top=True)

def fetch_bottom_tickers_by_metric(num_tickers_to_fetch, metric, time_period):
    return _fetch_extreme_tickers_by_metric(num_tickers_to_fetch, metric, time_period, top=False)

def _fetch_extreme_tickers_by_metric(num, metric, period, top=True):
    df_daily = pd.read_parquet(SP500_DAILY_LIST_PATH)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)

    grouped = df_daily.groupby(pd.Grouper(freq=_get_pandas_freq(period)))
    result_rows = []

    for period_start, group in grouped:
        if group.empty:
            continue
        print(f"\n🔍 Processing period starting {period_start.date()} with {len(group)} day(s)")
        tickers = set()
        for tlist in group['tickers']:
            tickers.update(tlist)

        period_metric_data = []
        for ticker in tickers:
            ticker_path = os.path.join(SP500_DATA_DIR, f"{ticker}.parquet")
            if not os.path.exists(ticker_path):
                continue

            try:
                df_ticker = pd.read_parquet(ticker_path)
                df_ticker = df_ticker[df_ticker['date'] >= period_start]
                df_ticker = df_ticker[df_ticker['date'] < _get_next_period_start(period_start, period)]
                if df_ticker.empty:
                    continue

                latest_row = df_ticker.sort_values('date').iloc[-1]
                val = latest_row.get(metric, None)
                if pd.notna(val):
                    period_metric_data.append({
                        'period_start': period_start,
                        'ticker': ticker,
                        metric: val
                    })
            except Exception as e:
                print(f"⚠️ Skipping {ticker} due to error: {e}")
                continue

        if period_metric_data:
            df_metric = pd.DataFrame(period_metric_data)
            df_metric.sort_values(metric, ascending=not top, inplace=True)
            top_rows = df_metric.head(num)

            # 🖨️ Print top or bottom N tickers for this period
            print(f"{'Top' if top else 'Bottom'} {num} tickers for '{metric}' in {period_start.date()}:")
            print(top_rows[['ticker', metric]].to_string(index=False))

            result_rows.extend(top_rows.to_dict('records'))

    return pd.DataFrame(result_rows)

def _get_pandas_freq(period):
    return {
        "day": "D",
        "week": "W",
        "month": "M",
        "quarter": "QS",  # Use quarter start
        "year": "YS"      # Use year start
    }.get(period.lower(), "QS")

def _get_next_period_start(period_start, period):
    if period == "day":
        return period_start + pd.Timedelta(days=1)
    elif period == "week":
        return period_start + pd.DateOffset(weeks=1)
    elif period == "month":
        return period_start + pd.DateOffset(months=1)
    elif period == "quarter":
        return period_start + pd.DateOffset(months=3)
    elif period == "year":
        return period_start + pd.DateOffset(years=1)
    else:
        return period_start + pd.DateOffset(months=3)
