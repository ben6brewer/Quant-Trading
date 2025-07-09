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
    cmap = cm.get_cmap('turbo')

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(fng)
    lc.set_linewidth(2)
    lc.set_zorder(1)
    line = ax.add_collection(lc)

    return line

def plot_moving_averages(ax, df):
    for col in df.columns:
        if '_ma' in col.lower():
            # Use specific labels for fast/slow MAs if window info is available
            if col == 'fast_ma' and 'fast_ma_window' in df.attrs:
                label = f"Fast MA ({df.attrs['fast_ma_window']})"
            elif col == 'slow_ma' and 'slow_ma_window' in df.attrs:
                label = f"Slow MA ({df.attrs['slow_ma_window']})"
            else:
                digits = ''.join(filter(str.isdigit, col))
                label = f"MA ({digits})" if digits else col.replace('_', ' ').title()

            ax.plot(df.index, df[col], label=label, alpha=0.75, zorder=2)


def plot_vix(ax, df):
    """
    Plots the VIX on its own subplot.
    """
    if 'vix' not in df.columns:
        print("DataFrame does not contain 'vix' column; skipping VIX plot.")
        return

    ax.plot(df.index, df['vix'], color='orange', alpha=0.8, label='VIX')
    ax.set_ylabel('VIX')
    ax.tick_params(axis='y')
    ax.grid(True)
    ax.legend(loc='upper right')

def plot_signals(dataframe):
    if 'close' not in dataframe.columns or 'signal' not in dataframe.columns:
        print("DataFrame must contain 'close' and 'signal' columns.")
        return
    
    df = dataframe.copy()
    df.attrs.update(dataframe.attrs)
    df.index = pd.to_datetime(df['date']) if 'date' in df.columns else pd.to_datetime(df.index)

    has_vix = 'vix' in df.columns

    if has_vix:
        # Create two subplots: top is 3x taller than bottom
        fig, (ax_price, ax_vix) = plt.subplots(
            2, 1, 
            figsize=(14, 8), 
            gridspec_kw={'height_ratios': [3, 1]}, 
            sharex=True
        )
    else:
        # Create only one subplot if no VIX data
        fig, ax_price = plt.subplots(figsize=(14, 6))

    # Plot price or F&G colored line on top
    if 'F&G' in df.columns:
        line = plot_fng(ax_price, df)
        plt.colorbar(line, ax=ax_price, label='Fear & Greed Index')
    else:
        ax_price.plot(df.index, df['close'], label='Close Price', alpha=0.6)

    plot_moving_averages(ax_price, df)

    # Plot signals on price plot
    signal_diff = df['signal'].diff().fillna(0)
    change_points = df[signal_diff != 0]

    buy_signals = change_points[change_points['signal'] == 1.0]
    sell_signals = change_points[change_points['signal'] == -1.0]
    exit_signals = change_points[(change_points['signal'] == 0.0) & (signal_diff.abs() >= 0.5)]

    partial_exit_signals = change_points[
        (change_points['signal'] < 1.0) & 
        (change_points['signal'] > 0.0) & 
        (signal_diff < 0)
    ]

    ax_price.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100, zorder=5)
    ax_price.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Short Selling Signal', s=100, zorder=5)
    ax_price.scatter(exit_signals.index, exit_signals['close'], marker='o', color='blue', label='Exit to Cash', s=80, zorder=5)
    ax_price.scatter(partial_exit_signals.index, partial_exit_signals['close'], marker='o', color='purple', label='Partial Exit', s=80, zorder=5)

    ax_price.set_title(f"{df.attrs.get('ticker')} {df.attrs.get('title')} Signals")

    ax_price.set_ylabel('Price')
    ax_price.legend(loc='upper left')
    ax_price.grid(True)

    if has_vix:
        # Plot VIX on bottom subplot
        plot_vix(ax_vix, df)
        ax_vix.set_xlabel('Date')
        ax_vix.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_vix.xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.setp(ax_price.get_xticklabels(), visible=False)  # Hide top x-axis labels
    else:
        ax_price.set_xlabel('Date')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

