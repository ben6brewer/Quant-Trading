import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd

def plot_btc_with_risk_metric(df, risk_col='mean_risk'):
    """
    Plot BTC close price and a chosen risk metric (0-1 scale) on twin y-axes,
    with black axis lines, ticks, and labels, and combined legend.

    Args:
        df (pd.DataFrame): Dataframe containing 'date', 'close', and risk columns.
        risk_col (str): Name of the risk column to plot ('mvrv_risk', 'F&G_risk', 'mean_risk', etc).
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

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

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_btc_color_coded_risk(df, risk_col='mean_risk', cmap_name='turbo'):
    """
    Plot BTC close price as a color-coded line by chosen risk metric (0-1),
    with a colorbar legend.

    Args:
        df (pd.DataFrame): Dataframe with 'date', 'close', and risk columns.
        risk_col (str): Risk column to color-code by.
        cmap_name (str): Name of matplotlib colormap to use.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(cmap_name)

    # Plot BTC price segments colored by risk metric
    for i in range(len(df) - 1):
        x = [df['date'].iloc[i], df['date'].iloc[i + 1]]
        y = [df['close'].iloc[i], df['close'].iloc[i + 1]]
        risk_val = df[risk_col].iloc[i]
        ax.plot(x, y, color=cmap(norm(risk_val)), linewidth=2)

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
