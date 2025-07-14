# utils/plot_equity_curve.py

import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec  

def plot_equity_curve(results_df, ax):
    if 'total_equity' not in results_df.columns:
        raise ValueError("DataFrame must contain a 'total_equity' column.")

    ticker = results_df.attrs.get('ticker', 'Unknown')
    title = results_df.attrs.get('title', 'Untitled Strategy')

    ax.plot(results_df.index, results_df['total_equity'], label='Total Equity', color='blue', linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Equity ($)")
    ax.set_title(f"{ticker} {title} Equity Curve")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')



def plot_equity_vs_benchmark(results_df, ax):
    if 'total_equity' not in results_df.columns or 'close' not in results_df.columns:
        raise ValueError("DataFrame must contain both 'total_equity' and 'close' columns.")

    close_prices = results_df['close'].replace(',', '', regex=True).astype(float)
    initial_equity = results_df['total_equity'].iloc[0]
    initial_price = close_prices.iloc[0]
    buy_and_hold_equity = close_prices / initial_price * initial_equity

    ticker = results_df.attrs.get('ticker', 'Unknown')
    title = results_df.attrs.get('title', 'Untitled Strategy')

    ax.plot(results_df.index, results_df['total_equity'], label='Strategy Equity', color='blue', linewidth=2)
    ax.plot(results_df.index, buy_and_hold_equity, label='Buy & Hold', color='orange', linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.set_title(f"{ticker} vs {title}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')


def plot_multiple_equity_curves(results_dfs, normalize=True):
    """
    Plots equity curves for multiple strategies starting from the latest common start date.
    Optionally normalizes equity curves to start at the same value.

    Args:
        results_dfs (list of pd.DataFrame): Each DataFrame must have datetime index and 'total_equity'.
                                            Optional: .attrs['title'] and .attrs['ticker'].
        normalize (bool): Whether to normalize all curves to start at the same value.
    """
    latest_start = max(df.index.min() for df in results_dfs)

    aligned_dfs = []
    for df in results_dfs:
        trimmed_df = df[df.index >= latest_start].copy()
        trimmed_df.attrs.update(df.attrs)

        if normalize:
            starting_equity = trimmed_df['total_equity'].iloc[0]
            trimmed_df['normalized_equity'] = trimmed_df['total_equity'] / starting_equity * 1_000_000
        else:
            trimmed_df['normalized_equity'] = trimmed_df['total_equity']

        aligned_dfs.append(trimmed_df)

    plt.figure(figsize=(14, 7))
    for df in aligned_dfs:
        title = df.attrs.get('title', 'Strategy')
        ticker = df.attrs.get('ticker', 'Asset')
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

import matplotlib.dates as mdates

def plot_grid_search_equity_curves(results_dfs, best_params, benchmark_df=None):
    """
    Plots equity curves from a grid search, highlighting the best-performing strategy,
    and optionally overlays a buy-and-hold benchmark.

    Args:
        results_dfs (list of pd.DataFrame): Each with 'total_equity' and a .attrs['params'] dict.
        best_params (dict): The best parameter set to highlight.
        benchmark_df (pd.DataFrame, optional): A buy-and-hold benchmark DataFrame to overlay.
    """
    print("Loading grid search results")
    if len(results_dfs) == 0:
        print("⚠️ No results to plot.")
        return

    # Align all DataFrames to the latest common start date
    latest_start = max(df.index.min() for df in results_dfs)
    aligned_dfs = []

    for df in results_dfs:
        trimmed_df = df[df.index >= latest_start].copy()
        trimmed_df.attrs.update(df.attrs)
        aligned_dfs.append(trimmed_df)

    # Also align benchmark if provided
    if benchmark_df is not None:
        benchmark_df = benchmark_df[benchmark_df.index >= latest_start].copy()

    plt.figure(figsize=(14, 7))
    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    # Plot all non-best strategies
    for df in aligned_dfs:
        params = df.attrs.get('params')
        if params != best_params:
            plt.plot(
                df.index,
                df['total_equity'],
                linewidth=0.5,
                alpha=1,
                color=next(color_cycle),
                zorder=1
            )

    # Highlight best-performing strategy
    for df in aligned_dfs:
        params = df.attrs.get('params')
        if params == best_params:
            label_parts = [f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" for k, v in params.items()]
            plt.plot(
                df.index,
                df['total_equity'],
                label=", ".join(label_parts),
                linewidth=2.5,
                color='lawngreen',
                zorder=3
            )

    if benchmark_df is not None:
        plt.plot(
            benchmark_df.index,
            benchmark_df['total_equity'],
            label=f"{results_dfs[0].attrs.get('ticker')} Benchmark",
            linewidth=2,
            color='blue',
            zorder=4
        )

    # Plot formatting
    plt.title(f"Equity Curve Comparison - {results_dfs[0].attrs.get('title', 'Strategy')}", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Total Equity ($)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_equity_tab(fig, results_df, curve_type="equity"):
    gs = gridspec.GridSpec(
        1, 1,
        hspace=0.1, wspace=0.05,
        figure=fig
    )
    ax = fig.add_subplot(gs[0])

    if curve_type == "equity":
        plot_equity_curve(results_df, ax)
    elif curve_type == "vs_benchmark":
        plot_equity_vs_benchmark(results_df, ax)
