# utils/plot_signals.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec  

def plot_fng(ax, df):
    """
    Plots the closing price with a color gradient based on the Fear & Greed index.
    Returns the LineCollection for use in colorbar.
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
    ax.add_collection(lc)

    return lc  # ✅ Return this for the colorbar


def plot_signals(df, ax_price, cbar_ax=None, ax_secondary=None):
    df = df.copy()
    df = df.sort_index()

    # Plot close price differently if 'F&G' column exists
    if 'F&G' in df.columns:
        lc = plot_fng(ax_price, df)  # ✅ Capture LineCollection
        if cbar_ax is not None:
            cbar = plt.colorbar(lc, cax=cbar_ax)  # ✅ Link to LineCollection
            cbar.set_label('Fear & Greed Index')  # Optional label
    else:
        # Plot black close price line
        ax_price.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)

        # Color-coded risk metric scatter plot (if any)
        risk_cols = [col for col in df.columns if col.endswith('_risk') and df[col].between(0, 1).any()]
        if risk_cols:
            risk_col = risk_cols[0]
            norm = mcolors.Normalize(vmin=0, vmax=1)
            cmap = plt.cm.coolwarm_r
            sc = ax_price.scatter(
                df.index, df['close'], c=df[risk_col], cmap=cmap, norm=norm,
                s=10, alpha=0.7, label=f'{risk_col}'
            )

            if cbar_ax is not None:
                cbar = plt.colorbar(sc, cax=cbar_ax)
                cbar.set_label('Risk Metric')

    # Plot signal markers (buy/sell)
    if 'signal' in df.columns:
        entries = df[df['signal'].diff() > 0]
        exits = df[df['signal'].diff() < 0]

        ax_price.scatter(entries.index, entries['close'], color='green', label='Buy Signal', marker='^', s=75, zorder=5)
        ax_price.scatter(exits.index, exits['close'], color='red', label='Sell Signal', marker='v', s=75 ,zorder=5)

    # Plot secondary axis (VIX, F&G, etc.)
    if ax_secondary:
        if 'vix' in df.columns:
            ax_secondary.plot(df.index, df['vix'], label='VIX', color='blue', alpha=0.4)
            ax_secondary.set_ylabel('VIX')
        elif 'F&G' in df.columns:
            ax_secondary.plot(df.index, df['F&G'], label='Fear & Greed Index', color='orange', alpha=0.4)
            ax_secondary.set_ylabel('F&G Index')

    # Format x-axis
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_price.xaxis.get_major_locator()))

    ax_price.set_title(df.attrs.get('title', 'Price & Signals'), fontsize=12)
    ax_price.set_ylabel('Price')

    # Deduplicate legend entries
    handles, labels = ax_price.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_price.legend(unique.values(), unique.keys(), loc='upper left')

    if ax_secondary:
        handles2, labels2 = ax_secondary.get_legend_handles_labels()
        unique2 = dict(zip(labels2, handles2))
        ax_secondary.legend(unique2.values(), unique2.keys(), loc='upper right')

def plot_signal_tab(fig, signal_df, has_fng, has_vix):
    if has_vix or has_fng:
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[3, 1] if has_vix else [1, 0],
            width_ratios=[20, 1] if has_fng else [1, 0],
            hspace=0.1, wspace=0.05,
            figure=fig
        )
        ax_price = fig.add_subplot(gs[0, 0])
        ax_vix = fig.add_subplot(gs[1, 0], sharex=ax_price) if has_vix else None
        cbar_ax = fig.add_subplot(gs[:, 1]) if has_fng else None
    else:
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax_price = fig.add_subplot(gs[0])
        ax_vix = None
        cbar_ax = None

    plot_signals(signal_df, ax_price=ax_price, cbar_ax=cbar_ax, ax_secondary=ax_vix)

