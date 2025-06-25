# utils/plot_equity_curve.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_equity_curve(results_df, title="Equity Curve", save_path=None):
    """
    Plots the total equity curve from a backtest results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame with a 'total_equity' column and datetime index.
        title (str): Title of the plot.
        save_path (str or None): Optional path to save the plot image.
    """
    if 'total_equity' not in results_df.columns:
        raise ValueError("DataFrame must contain a 'total_equity' column.")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['total_equity'], label='Total Equity', color='blue', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Equity ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    # Improve date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Equity curve saved to: {save_path}")
    else:
        plt.show()
