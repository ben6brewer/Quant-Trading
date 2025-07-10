# backtest/performance_metrics.py

import numpy as np
import pandas as pd
import yfinance as yf

CRYPTO_TRADING_DAYS_PER_YEAR = 365

def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    return df['total_equity'].pct_change().fillna(0)

def calculate_cumulative_returns(df: pd.DataFrame) -> float:
    returns = calculate_daily_returns(df)
    return (1 + returns).prod() - 1

def calculate_cagr(df: pd.DataFrame) -> float:
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

    years = days / CRYPTO_TRADING_DAYS_PER_YEAR
    total_return = (end_value / start_value) - 1

    if total_return >= 0:
        cagr = (end_value / start_value) ** (1 / years) - 1
    else:
        cagr = -((start_value / end_value) ** (1 / years) - 1)

    return cagr

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    returns = calculate_daily_returns(df)
    excess_returns = returns - (risk_free_rate / CRYPTO_TRADING_DAYS_PER_YEAR)
    std_dev = returns.std()
    if std_dev == 0:
        return 0
    return (excess_returns.mean() / std_dev) * np.sqrt(CRYPTO_TRADING_DAYS_PER_YEAR)

def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    returns = calculate_daily_returns(df)
    downside_returns = returns[returns < 0]
    if downside_returns.std() == 0:
        return 0
    excess_return = returns.mean() - (risk_free_rate / CRYPTO_TRADING_DAYS_PER_YEAR)
    return (excess_return / downside_returns.std()) * np.sqrt(CRYPTO_TRADING_DAYS_PER_YEAR)

def calculate_max_drawdown(df: pd.DataFrame) -> float:
    equity_curve = df['total_equity'].cummax()
    drawdown = df['total_equity'] / equity_curve - 1
    return drawdown.min()

def calculate_volatility(df: pd.DataFrame) -> float:
    returns = calculate_daily_returns(df)
    return returns.std() * np.sqrt(CRYPTO_TRADING_DAYS_PER_YEAR)

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

def extract_performance_metrics_dict(df: pd.DataFrame) -> dict:
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
