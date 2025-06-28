# utils/plot_signals.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_signals(data_frame):
    """
    Plots the closing price and signal changes.
    Optionally plots moving averages if present.
    Displays actual date values on the x-axis.
    Also plots 'exit to cash' signals when positions are closed.
    """
    if 'close' not in data_frame.columns or 'signal' not in data_frame.columns:
        print("DataFrame must contain 'close' and 'signal' columns.")
        return

    df = data_frame.copy()
    df.index = pd.to_datetime(df['date']) if 'date' in df.columns else pd.to_datetime(df.index)
    ticker = getattr(data_frame, 'ticker', '')
    title = getattr(data_frame, 'title', '')

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', alpha=0.6)

    if '50_week_ma' in df.columns:
        plt.plot(df.index, df['50_week_ma'], label='50 Week MA', alpha=0.75)

    # Identify signal change points
    signal_diff = df['signal'].diff().fillna(0)
    change_points = df[signal_diff != 0]

    # Buy = signal went to 1
    buy_signals = change_points[change_points['signal'] == 1]

    # Sell/Short = signal went to -1
    sell_signals = change_points[change_points['signal'] == -1]

    # Exit to cash = signal went to 0 from 1 or -1
    exit_signals = change_points[(change_points['signal'] == 0) & (signal_diff.abs() == 1)]

    # Plot markers
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Short Selling Signal', s=100)
    plt.scatter(exit_signals.index, exit_signals['close'], marker='o', color='blue', label='Exit to Cash', s=80)

    plt.title(f"{ticker} {title} Signals".strip())
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
