# visualizations/plot_composite_btc_risk.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import pandas as pd

def plot_btc_with_risk_metric(df):
    """
    Interactive plot of BTC close price and risk metrics (0-1 scale) on twin y-axes.
    Starts with mean_risk, use left/right arrows to switch between other risk metrics.
    Dynamically zooms x-axis to the available date range of the current risk metric.
    Legend placed just below the latest risk value annotation in top-left.
    Title styled similar to plot_btc_color_coded_risk_metric.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    risk_cols = [col for col in df.columns if col.endswith('_risk')]
    risk_cols = ['mean_risk'] + [col for col in risk_cols if col != 'mean_risk']
    idx = [0]  # Mutable index to track which risk is currently shown

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax2 = ax1.twinx()

    def draw_plot():
        ax1.clear()
        ax2.clear()

        risk_col = risk_cols[idx[0]]

        # Zoom x-axis to valid dates for current risk metric
        valid_dates = df.loc[df[risk_col].notna(), 'date']
        if not valid_dates.empty:
            x_min, x_max = valid_dates.min(), valid_dates.max()
        else:
            x_min, x_max = df['date'].min(), df['date'].max()

        # Title styled like plot_btc_color_coded_risk_metric
        ax1.set_title(f'BTC Close Price and Risk Metric: {risk_col}', fontsize=14)

        # BTC Price
        ax1.set_yscale('log')
        ax1.plot(df['date'], df['close'], color='orange', label='BTC Price')
        ax1.set_xlim(x_min, x_max)
        ax1.set_xlabel('Date', color='black')
        ax1.set_ylabel('BTC Price', color='black')
        ax1.tick_params(axis='x', colors='black')
        ax1.tick_params(axis='y', colors='black')
        for spine in ax1.spines.values():
            spine.set_color('black')
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Risk Metric
        ax2.plot(df['date'], df[risk_col], color='crimson', label=f'{risk_col} (0–1)')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylabel(f'{risk_col} (0–1)', color='black', rotation=270, labelpad=15)
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(axis='y', colors='black')
        for spine in ax2.spines.values():
            spine.set_color('black')

        # Latest value annotation on ax1 (top-left)
        latest_value = df[risk_col].dropna().iloc[-1] if not df[risk_col].dropna().empty else None
        if latest_value is not None:
            text_box = ax1.text(
                0.01, 0.97,
                f"Latest {risk_col}: {latest_value:.3f}",
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
            )
        else:
            text_box = None

        # Legend: positioned just below latest value annotation with more vertical padding
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2

        # Increased vertical spacing: y=0.90 instead of 0.93
        ax1.legend(all_lines, all_labels, loc='upper left', fontsize=9, bbox_to_anchor=(0.01, 0.90), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave some room for title
        plt.draw()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(risk_cols)
            draw_plot()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(risk_cols)
            draw_plot()

    draw_plot()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def plot_btc_color_coded_risk_metric(df):
    """
    Interactive color-coded plot of BTC close price by risk metric.
    Starts with mean_risk; use left/right arrow keys to cycle through.
    Dynamically zooms x-axis to available data range per risk metric.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    risk_cols = [col for col in df.columns if col.endswith('_risk')]
    risk_cols = ['mean_risk'] + [col for col in risk_cols if col != 'mean_risk']
    idx = [0]

    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    cbar_ax.set_facecolor('white')

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('turbo')

    def draw_plot():
        ax.clear()
        cbar_ax.clear()

        risk_col = risk_cols[idx[0]]

        for i in range(len(df) - 1):
            x = [df['date'].iloc[i], df['date'].iloc[i + 1]]
            y = [df['close'].iloc[i], df['close'].iloc[i + 1]]
            risk_val = df[risk_col].iloc[i]
            ax.plot(x, y, color=cmap(norm(risk_val)), linewidth=2)

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

        ax.set_yscale('log')
        ax.set_title(f'BTC Price Color-Coded by {risk_col}', fontsize=14)
        ax.set_xlabel('Date', color='black')
        ax.set_ylabel('BTC Price', color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')

        # Zoom x-axis to valid dates for current risk metric
        valid_dates = df.loc[df[risk_col].notna(), 'date']
        if not valid_dates.empty:
            ax.set_xlim(valid_dates.min(), valid_dates.max())
        else:
            ax.set_xlim(df['date'].min(), df['date'].max())

        ax.relim()
        ax.autoscale_view()

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f'{risk_col} (0–1)', color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(cbar.ax.get_yticklabels(), color='black')

        fig.subplots_adjust(bottom=0.18)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(risk_cols)
            draw_plot()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(risk_cols)
            draw_plot()

    draw_plot()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()