# utils/plot_signals.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_signals(df):
    """
    Plots the closing price, 50-week moving average, and buy/sell signal changes.

    Only plots signal points when the signal value changes from the previous row.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', alpha=0.6)
    plt.plot(df.index, df['50_week_ma'], label='50 Week MA', alpha=0.75)

    # Find signal change points
    signal_changes = df['signal'].diff().fillna(0) != 0
    change_points = df[signal_changes]

    # Plot buy/sell signals
    buy_signals = change_points[change_points['signal'] == 1]
    sell_signals = change_points[change_points['signal'] == -1]

    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)

    plt.title(f'{df.ticker} {df.title} Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()