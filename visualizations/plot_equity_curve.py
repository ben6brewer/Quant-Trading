# utils/plot_equity_curve.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_equity_curve(results_df):
    """
    Plots the total equity curve from a backtest results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame with a 'total_equity' column and datetime index.
        save_path (str or None): Optional path to save the plot image.
    """
    if 'total_equity' not in results_df.columns:
        raise ValueError("DataFrame must contain a 'total_equity' column.")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['total_equity'], label='Total Equity', color='blue', linewidth=2)
    
    plt.title(f"{results_df.ticker} {results_df.title} Equity Curve", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Equity ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    # Improve date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()

def plot_equity_vs_benchmark(results_df):
    """
    Plots the strategy's equity curve against a Buy & Hold equity curve using the 'close' and 'total_equity' columns.

    Args:
        results_df (pd.DataFrame): DataFrame with 'total_equity' and 'close' columns, indexed by datetime.
        save_path (str or None): Optional path to save the plot image.
    """
    if 'total_equity' not in results_df.columns or 'close' not in results_df.columns:
        raise ValueError("DataFrame must contain both 'total_equity' and 'close' columns.")

    # Convert 'close' column to float if it's in string format (e.g., with commas)
    close_prices = results_df['close'].replace(',', '', regex=True).astype(float)

    # Calculate Buy & Hold equity curve
    initial_equity = results_df['total_equity'].iloc[0]
    initial_price = close_prices.iloc[0]
    buy_and_hold_equity = close_prices / initial_price * initial_equity

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['total_equity'], label='Strategy Equity', color='blue', linewidth=2)
    plt.plot(results_df.index, buy_and_hold_equity, label='Buy & Hold', color='orange', linewidth=2)

    plt.title(f"{results_df.ticker} vs {results_df.title}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Equity ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()