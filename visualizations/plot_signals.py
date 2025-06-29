# utils/plot_signals.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_fng(ax, df):
    """
    Plots the closing price with a color gradient based on the Fear & Greed index.
    """
    x = mdates.date2num(df.index.to_numpy())
    y = df['close'].to_numpy()
    fng = df['F&G'].to_numpy()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=df['F&G'].min(), vmax=df['F&G'].max())
    cmap = cm.get_cmap('turbo')  # Change colormap here

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(fng)
    lc.set_linewidth(2)
    lc.set_zorder(1)
    line = ax.add_collection(lc)

    return line

def plot_moving_averages(ax, df):
    """
    Plots optional moving averages.
    """
    if '50_week_ma' in df.columns:
        ax.plot(df.index, df['50_week_ma'], label='50 Week MA', alpha=0.75, zorder=2)

def plot_signals(dataframe):
    """
    Master plotting function that renders close prices, moving averages, and signals.
    """
    if 'close' not in dataframe.columns or 'signal' not in dataframe.columns:
        print("DataFrame must contain 'close' and 'signal' columns.")
        return

    df = dataframe.copy()
    df.title = dataframe.title
    df.ticker = dataframe.ticker
    df.index = pd.to_datetime(df['date']) if 'date' in df.columns else pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(14, 7))

    if 'F&G' in df.columns:
        line = plot_fng(ax, df)
        plt.colorbar(line, ax=ax, label='Fear & Greed Index')
    else:
        ax.plot(df.index, df['close'], label='Close Price', alpha=0.6)

    plot_moving_averages(ax, df)

    signal_diff = df['signal'].diff().fillna(0)
    change_points = df[signal_diff != 0]

    buy_signals = change_points[change_points['signal'] == 1]
    sell_signals = change_points[change_points['signal'] == -1]
    exit_signals = change_points[(change_points['signal'] == 0) & (signal_diff.abs() == 1)]

    ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100, zorder=5)
    ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Short Selling Signal', s=100, zorder=5)
    ax.scatter(exit_signals.index, exit_signals['close'], marker='o', color='blue', label='Exit to Cash', s=80, zorder=5)

    ax.set_title(f"{df.ticker} {df.title} Signals".strip())
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
