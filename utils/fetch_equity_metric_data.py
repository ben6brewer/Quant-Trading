import yfinance as yf
import pandas as pd

def fetch_equity_metric_data_list(equity_ticker_list, metric='trailingPE'):
    """
    Fetches historical close prices and the most recent specified fundamental metric for each equity.
    Groups all results by date, with tickers sorted alphabetically within each date group.

    Args:
        equity_ticker_list (list of str): List of ticker symbols.
        metric (str): The fundamental metric to fetch (e.g. 'trailingPE', 'forwardPE').

    Returns:
        pd.core.groupby.DataFrameGroupBy: Grouped DataFrame by 'date' with columns ['ticker', 'date', 'close', metric]
    """
    all_data = []

    for ticker in equity_ticker_list:
        try:
            yf_ticker = yf.Ticker(ticker)

            # Get historical daily close prices (default 1 year)
            hist = yf_ticker.history(period="1y")
            if hist.empty:
                continue

            hist = hist.reset_index()
            hist['ticker'] = ticker
            hist['date'] = pd.to_datetime(hist['Date'])
            hist = hist[['ticker', 'date', 'Close']].rename(columns={'Close': 'close'})

            # Fetch the static fundamental metric for all dates
            metric_value = yf_ticker.info.get(metric, None)
            hist[metric] = metric_value

            all_data.append(hist)

        except Exception as e:
            print(f"⚠️ Failed to fetch data for {ticker}: {e}")

    if not all_data:
        # Return empty grouped DataFrame with correct columns
        empty_df = pd.DataFrame(columns=['ticker', 'date', 'close', metric])
        return empty_df.groupby('date')

    full_df = pd.concat(all_data, ignore_index=True)

    # Sort by date then ticker alphabetically to ensure consistent order per date
    full_df.sort_values(['date', 'ticker'], inplace=True)

    # Return grouped by date
    return full_df.groupby('date')
