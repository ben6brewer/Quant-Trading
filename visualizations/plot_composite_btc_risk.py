# visualizations/plot_composite_btc_risk.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd

def plot_btc_with_risk_metric(df, risk_col='mean_risk'):
    """
    Plot BTC close price and a chosen risk metric (0-1 scale) on twin y-axes,
    with black axis lines, ticks, and labels, and combined legend.
    Also shows most recent value of the risk metric as an annotation.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    # Set log scale for BTC price axis
    ax1.set_yscale('log')

    # BTC Close Price - orange line
    ax1.plot(df['date'], df['close'], color='orange', label='BTC Price')
    ax1.set_xlabel('Date', color='black')
    ax1.set_ylabel('BTC Price', color='black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    for spine in ax1.spines.values():
        spine.set_color('black')

    # X-axis formatting
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Risk metric line on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df[risk_col], color='crimson', label=f'{risk_col} (0–1)')
    ax2.set_ylabel(f'{risk_col} (0–1)', color='black')
    ax2.tick_params(axis='y', colors='black')
    for spine in ax2.spines.values():
        spine.set_color('black')

    # Add recent risk metric value in top-left
    latest_value = df[risk_col].dropna().iloc[-1] if not df[risk_col].dropna().empty else None
    if latest_value is not None:
        ax1.text(
            0.01, 0.97,
            f"Latest {risk_col}: {latest_value:.3f}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
        )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_btc_color_coded_risk_metric(df, risk_col='mean_risk'):
    """
    Plot BTC close price as a color-coded line by chosen risk metric (0-1),
    with a colorbar legend and annotation showing the most recent risk value.

    Args:
        df (pd.DataFrame): Dataframe with 'date', 'close', and risk columns.
        risk_col (str): Risk column to color-code by.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('turbo')
    
    # Plot BTC price segments colored by risk metric
    for i in range(len(df) - 1):
        x = [df['date'].iloc[i], df['date'].iloc[i + 1]]
        y = [df['close'].iloc[i], df['close'].iloc[i + 1]]
        risk_val = df[risk_col].iloc[i]
        ax.plot(x, y, color=cmap(norm(risk_val)), linewidth=2)

    # Add annotation for the most recent risk value
    latest_risk_val = df[risk_col].dropna().iloc[-1] if not df[risk_col].dropna().empty else None
    if latest_risk_val is not None:
        ax.text(
            0.01, 0.97,
            f"Latest {risk_col}: {latest_risk_val:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
        )
# Set log scale for BTC price axis
    ax.set_yscale('log')
    ax.set_title(f'BTC Price Color-Coded by {risk_col}', fontsize=14)
    ax.set_xlabel('Date', color='black')
    ax.set_ylabel('BTC Price', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(f'{risk_col} (0–1)', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    plt.tight_layout()
    plt.show()
