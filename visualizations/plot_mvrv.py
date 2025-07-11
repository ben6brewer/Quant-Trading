import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.colors as mcolors

def plot_btc_mvrv_z_score(df):
    """
    Plot BTC close price (orange) and MVRV z-score (red) on twin y-axes,
    with all axis lines, ticks, and labels in black, legend colored.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')  # white background

    # BTC Close Price - orange line
    ax1.plot(df['date'], df['close'], color='orange', label='BTC Price')
    ax1.set_xlabel('Date', color='black')
    ax1.set_ylabel('BTC Price', color='black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')

    # Set all spines of ax1 to black
    for spine in ax1.spines.values():
        spine.set_color('black')

    # X-axis date formatting
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # MVRV z-score - red line on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['mvrv_z_score'], color='crimson', label='MVRV Z-Score')
    ax2.set_ylabel('MVRV Z-Score', color='black')
    ax2.tick_params(axis='y', colors='black')

    # Set all spines of ax2 to black
    for spine in ax2.spines.values():
        spine.set_color('black')

    # Combine legends (colored lines)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_btc_mvrv_z_score_risk(df):
    """
    Plot BTC close price (orange) and MVRV z-score risk (red, normalized 0–1)
    on twin y-axes, with black axis lines, ticks, and labels, and a legend.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')  # white background

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

    # MVRV Risk - red line on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['mvrv_risk'], color='crimson', label='MVRV Risk (0–1)')
    ax2.set_ylabel('MVRV Risk (0–1)', color='black')
    ax2.tick_params(axis='y', colors='black')
    for spine in ax2.spines.values():
        spine.set_color('black')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_btc_color_coded_mvrv_z_score_risk(df):
    """
    Plot BTC close price with color-coded line using mvrv_risk (0–1) as the gradient,
    using the 'turbo' colormap. Only the BTC price is shown.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Normalize mvrv_risk to [0, 1] for colormap
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('turbo')

    # Plot BTC close price as a colored line by segment
    for i in range(len(df) - 1):
        x = [df['date'].iloc[i], df['date'].iloc[i + 1]]
        y = [df['close'].iloc[i], df['close'].iloc[i + 1]]
        risk = df['mvrv_risk'].iloc[i]
        ax.plot(x, y, color=cmap(norm(risk)), linewidth=2)

    # Axis labels and formatting
    ax.set_title('BTC Price Color-Coded by MVRV Risk', fontsize=14)
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

    # Colorbar to show risk gradient
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('MVRV Risk (0–1)', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    plt.tight_layout()
    plt.show()