# backtest/performance_metrics.py

import numpy as np
import pandas as pd
import yfinance as yf  # kept for compatibility if other modules import/use it
from config.universe_config import BACKTEST_CONFIG

CRYPTO_TRADING_DAYS_PER_YEAR = 365
EQUITY_TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_YEAR = EQUITY_TRADING_DAYS_PER_YEAR  # Default to equity unless specified


# ------------------------- Core daily/annualized calcs -------------------------

def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute daily returns for a price or equity curve series.
    Assumes df has a column 'total_equity'.
    """
    returns = df['total_equity'].pct_change().dropna()
    return returns


def calculate_cumulative_returns(df: pd.DataFrame) -> float:
    returns = calculate_daily_returns(df)
    return (1 + returns).prod() - 1


def calculate_cagr(df: pd.DataFrame) -> float:
    """
    CAGR computed over the span of the DataFrame index, using TRADING_DAYS_PER_YEAR
    to convert days to years (consistent with the annualization used elsewhere).
    """
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"❌ Error converting index to datetime in CAGR: {e}")
            return 0

    try:
        start_date = df.index[0]
        end_date = df.index[-1]
        days = (end_date - start_date).days
    except Exception as e:
        print(f"❌ Error calculating duration in CAGR: {e}")
        return 0

    start_value = df['total_equity'].iloc[0]
    end_value = df['total_equity'].iloc[-1]

    if days == 0 or start_value <= 0 or end_value <= 0:
        return 0

    years = days / TRADING_DAYS_PER_YEAR
    total_return = (end_value / start_value) - 1

    if total_return >= 0:
        cagr = (end_value / start_value) ** (1 / years) - 1
    else:
        cagr = -((start_value / end_value) ** (1 / years) - 1)

    return cagr


def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
    """
    Annualized Sharpe using TRADING_DAYS_PER_YEAR and a flat annual risk-free rate.
    """
    returns = calculate_daily_returns(df)
    excess_returns = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    std_dev = returns.std()
    if std_dev == 0:
        return 0
    return (excess_returns.mean() / std_dev) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
    """
    Annualized Sortino using downside deviation (returns < 0) and TRADING_DAYS_PER_YEAR.
    """
    returns = calculate_daily_returns(df)
    downside_returns = returns[returns < 0]
    if downside_returns.std() == 0:
        return 0
    excess_return = returns.mean() - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    return (excess_return / downside_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_max_drawdown(df: pd.DataFrame) -> float:
    equity_curve = df['total_equity'].cummax()
    drawdown = df['total_equity'] / equity_curve - 1
    return drawdown.min()


def calculate_volatility(df: pd.DataFrame) -> float:
    """
    Calculate annualized volatility (std dev) of returns.
    Excludes weekends and annualizes with TRADING_DAYS_PER_YEAR (252).
    """
    returns = calculate_daily_returns(df)
    # Keep only weekdays (Mon–Fri)
    returns = returns[returns.index.dayofweek < 5]
    # Annualized volatility
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


# ------------------------- Trade extraction (generic) -------------------------

def extract_trades(df: pd.DataFrame) -> list:
    trades = []
    position = 0.0
    entry_price = None
    entry_date = None

    for date, row in df.iterrows():
        signal = row['signal']
        price = row['close']

        if signal != position:
            if position != 0 and entry_price is not None:
                if position > 0:
                    trade_return = (price - entry_price) / entry_price
                else:
                    trade_return = (entry_price - price) / entry_price

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'direction': 'long' if position > 0 else 'short',
                    'return': trade_return
                })

            if signal != 0:
                entry_price = price
                entry_date = date
                position = signal
            else:
                position = 0
                entry_price = None
                entry_date = None

    if position != 0 and entry_price is not None:
        final_price = df['close'].iloc[-1]
        final_date = df.index[-1]
        if position > 0:
            trade_return = (final_price - entry_price) / entry_price
        else:
            trade_return = (entry_price - final_price) / entry_price

        trades.append({
            'entry_date': entry_date,
            'exit_date': final_date,
            'direction': 'long' if position > 0 else 'short',
            'return': trade_return
        })

    return trades


# ------------------------- Default metrics dict (generic) -------------------------

def extract_performance_metrics_dict(df: pd.DataFrame) -> dict:
    """
    Generic full metrics set used for most strategies (incl. trade stats).
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"❌ Failed to convert index to datetime in performance metrics: {e}")
                return {}

    cagr = calculate_cagr(df)
    cum_return = calculate_cumulative_returns(df)
    sharpe = calculate_sharpe_ratio(df)
    sortino = calculate_sortino_ratio(df)
    max_dd = calculate_max_drawdown(df)
    vol = calculate_volatility(df)

    trades = extract_trades(df)
    num_trades = len(trades)
    wins = [t for t in trades if t['return'] > 0]
    losses = [t for t in trades if t['return'] <= 0]
    win_rate = len(wins) / num_trades if num_trades else 0
    avg_win = sum(t['return'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['return'] for t in losses) / len(losses) if losses else 0

    return {
        "Title": df.attrs.get('title', 'Unknown'),
        "Ticker": df.attrs.get('ticker', 'Unknown'),
        "CAGR (%)": cagr * 100,
        "Cumulative Return (%)": cum_return * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd * 100,
        "Volatility (%)": vol * 100,
        "Number of Trades": num_trades,
        "Win Rate (%)": win_rate * 100,
        "Avg Win (%)": avg_win * 100,
        "Avg Loss (%)": avg_loss * 100
    }


# ------------------------- IWV/AGG EI mix–specific metrics -------------------------

def _weekday_returns(df: pd.DataFrame) -> pd.Series:
    """
    Helper: weekday-only daily returns, for consistent annualization.
    """
    r = calculate_daily_returns(df)
    return r[r.index.dayofweek < 5]


def calculate_variance_annualized(df: pd.DataFrame) -> float:
    """
    Annualized variance of weekday returns.
    """
    r = _weekday_returns(df)
    return float(r.var() * TRADING_DAYS_PER_YEAR)


def calculate_stddev_annualized(df: pd.DataFrame) -> float:
    """
    Annualized standard deviation of weekday returns.
    """
    r = _weekday_returns(df)
    return float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def extract_eif_metrics_dict(df: pd.DataFrame) -> dict:
    """
    Metrics tailored for static IWV/AGG equity-vs-fixed-income mixes.

    Fields:
      - Ticker
      - CAGR (%)
      - Cumulative Return (%)
      - Sharpe
      - Sortino
      - Variance (ann.)
      - Std Dev (ann. %)
      - Volatility (ann. %)   # identical to Std Dev here; included for presentation
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return {}

    ticker = df.attrs.get('ticker', 'Unknown')
    cagr = calculate_cagr(df)
    cum_return = calculate_cumulative_returns(df)
    sharpe = calculate_sharpe_ratio(df)
    sortino = calculate_sortino_ratio(df)
    var_ann = calculate_variance_annualized(df)
    std_ann = calculate_stddev_annualized(df)
    vol_ann = std_ann  # synonymous here

    return {
        "Ticker": ticker,
        "CAGR (%)": cagr * 100,
        "Cumulative Return (%)": cum_return * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Variance (ann.)": var_ann,
        "Std Dev (ann. %)": std_ann * 100,
        "Max Drawdown (%)": calculate_max_drawdown(df) * 100,
    }
