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
    start_value = df['total_equity'].iloc[0]
    end_value = df['total_equity'].iloc[-1]
    days = (df.index[-1] - df.index[0]).days
    if days == 0:
        return 0
    years = days / CRYPTO_TRADING_DAYS_PER_YEAR
    return (end_value / start_value) ** (1 / years) - 1

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
    position = None
    entry_price = None
    entry_date = None
    signal_prev = 0

    for date, row in df.iterrows():
        signal = row['signal']
        price = row['close']

        if signal != signal_prev and signal != 0:
            if position is not None:
                trade_return = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'direction': 'long' if position == 1 else 'short',
                    'return': trade_return
                })
            position = signal
            entry_price = price
            entry_date = date

        elif signal == 0 and position is not None:
            trade_return = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'direction': 'long' if position == 1 else 'short',
                'return': trade_return
            })
            position = None
            entry_price = None
            entry_date = None

        signal_prev = signal

    return trades

def print_performance_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0429):
    trades = extract_trades(df)
    num_trades = len(trades)
    wins = [t for t in trades if t['return'] > 0]
    losses = [t for t in trades if t['return'] <= 0]
    win_rate = len(wins) / num_trades if num_trades else 0
    avg_win = np.mean([t['return'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['return'] for t in losses]) if losses else 0

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Calculate number of years from index
    start_date = df.index.min()
    end_date = df.index.max()
    num_years = (end_date - start_date).days / 365.25
    print()
    print("📈 Performance Metrics")
    print(f"CAGR: {calculate_cagr(df) * 100:.2f}%")
    print(f"Cumulative Return over {num_years:.2f} years: {calculate_cumulative_returns(df) * 100:.2f}%")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(df, risk_free_rate):.4f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(df, risk_free_rate):.4f}")
    print(f"Max Drawdown: {calculate_max_drawdown(df) * 100:.2f}%")
    print(f"Volatility: {calculate_volatility(df) * 100:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Number of Wins: {len(wins)}")
    print(f"Number of Losses: {len(losses)}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Average Win (%): {avg_win * 100:.2f}%")
    print(f"Average Loss (%): {avg_loss * 100:.2f}%")
    print()