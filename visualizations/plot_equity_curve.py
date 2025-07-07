# utils/plot_equity_curve.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_equity_curve(results_df):
    """
    Plots the total equity curve from a backtest results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame with a 'total_equity' column and datetime index.
        save_path (str or None): Optional path to save the plot image.
    """
    if 'total_equity' not in results_df.columns:
        raise ValueError("DataFrame must contain a 'total_equity' column.")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['total_equity'], label='Total Equity', color='blue', linewidth=2)
    
    plt.title(f"{results_df.ticker} {results_df.title} Equity Curve", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Equity ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    # Improve date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()

def plot_equity_vs_benchmark(results_df):
    """
    Plots the strategy's equity curve against a Buy & Hold equity curve using the 'close' and 'total_equity' columns.

    Args:
        results_df (pd.DataFrame): DataFrame with 'total_equity' and 'close' columns, indexed by datetime.
        save_path (str or None): Optional path to save the plot image.
    """
    if 'total_equity' not in results_df.columns or 'close' not in results_df.columns:
        raise ValueError("DataFrame must contain both 'total_equity' and 'close' columns.")

    # Convert 'close' column to float if it's in string format (e.g., with commas)
    close_prices = results_df['close'].replace(',', '', regex=True).astype(float)

    # Calculate Buy & Hold equity curve
    initial_equity = results_df['total_equity'].iloc[0]
    initial_price = close_prices.iloc[0]
    buy_and_hold_equity = close_prices / initial_price * initial_equity

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['total_equity'], label='Strategy Equity', color='blue', linewidth=2)
    plt.plot(results_df.index, buy_and_hold_equity, label='Buy & Hold', color='orange', linewidth=2)

    plt.title(f"{results_df.ticker} vs {results_df.title}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Equity ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()


def plot_multiple_equity_curves(results_dfs, normalize=True):
    """
    Plots equity curves for multiple strategies starting from the latest common start date.
    Optionally normalizes equity curves to start at the same value.

    Args:
        results_dfs (list of pd.DataFrame): Each DataFrame must have datetime index and 'total_equity'.
                                            Optional: `.title` and `.ticker` attributes.
        normalize (bool): Whether to normalize all curves to start at the same value.
    """
    latest_start = max(df.index.min() for df in results_dfs)

    aligned_dfs = []
    for df in results_dfs:
        trimmed_df = df[df.index >= latest_start].copy()
        for attr in ['title', 'ticker']:
            if hasattr(df, attr):
                setattr(trimmed_df, attr, getattr(df, attr))

        if normalize:
            starting_equity = trimmed_df['total_equity'].iloc[0]
            trimmed_df['normalized_equity'] = trimmed_df['total_equity'] / starting_equity * 1_000_000
        else:
            trimmed_df['normalized_equity'] = trimmed_df['total_equity']

        aligned_dfs.append(trimmed_df)

    plt.figure(figsize=(14, 7))
    for df in aligned_dfs:
        title = getattr(df, 'title', 'Strategy')
        ticker = getattr(df, 'ticker', 'Asset')
        label = f"{ticker} - {title}"
        plt.plot(df.index, df['normalized_equity'], label=label, linewidth=2)

    plt.title("Equity Curve Comparison", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Total Equity ($)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_grid_search_equity_curves(results_dfs, best_params):
    latest_start = max(df.index.min() for df in results_dfs)

    aligned_dfs = []
    for df in results_dfs:
        trimmed_df = df[df.index >= latest_start].copy()
        trimmed_df.attrs.update(df.attrs)

        aligned_dfs.append(trimmed_df)

    plt.figure(figsize=(14, 7))
    for df in aligned_dfs:
        params = df.attrs.get('params')
        if params:
            label = f"VIX:{params['vix_threshold']} TP:{params['take_profit_pct']:.2f} Exit:{params['partial_exit_pct']:.2f}"
        else:
            title = df.attrs.get('title', 'Strategy')
            ticker = df.attrs.get('ticker', 'Asset')
            label = f"{ticker} - {title}"

        is_best = params == best_params if params is not None else False

        if is_best:
            plt.plot(df.index, df['total_equity'], label=label, linewidth=2, alpha=1.0, color='red')
        else:
            plt.plot(df.index, df['total_equity'], label=label, linewidth=1, alpha=0.85)

    plt.title("Equity Curve Comparison", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Total Equity ($)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()