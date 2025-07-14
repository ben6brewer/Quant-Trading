from config.secrets_config import *
from config.universe_config import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import *
from utils.pretty_print_df import *
from backtest.performance_metrics import *
from utils.data_fetch import *
import pandas as pd

def compare_strategies(strategy_settings_list):
    """
    Runs, trims, collects metrics, and plots strategies from a list of strategy settings dicts.
    Each settings dict must include 'strategy_class', 'title', and 'ticker'.
    """
    backtester = BacktestEngine()
    raw_data = []

    # Step 1: Fetch and tag data
    for settings in strategy_settings_list:
        df = fetch_data_for_strategy(settings)

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    ticker = getattr(df.attrs, 'ticker', 'unknown ticker')
                    print(f"Warning: Could not convert index to datetime for {ticker}: {e}")

        df.attrs['title'] = settings.get('title', 'Untitled Strategy')
        df.attrs['ticker'] = settings.get('ticker', 'Unknown')

        raw_data.append((settings, df))

    # Step 2: Align start date
    latest_start = max(df.index.min() for _, df in raw_data)

    trimmed_data = []
    for settings, df in raw_data:
        trimmed = df[df.index >= latest_start].copy()
        trimmed.attrs['title'] = df.attrs['title']
        trimmed.attrs['ticker'] = df.attrs['ticker']
        trimmed_data.append((settings, trimmed))

    # Step 3: Run each strategy, backtest, and collect metrics
    metrics_list = []
    results = []
    for settings, df in trimmed_data:
        strategy_class = settings["strategy_class"]
        strategy_args = {k: v for k, v in settings.items() if k not in ['strategy_class', 'title', 'ticker']}
        strategy = strategy_class(**strategy_args)

        signal_df = strategy.generate_signals(df)
        result_df = backtester.run_backtest(signal_df)

        result_df.attrs['title'] = df.attrs['title']
        result_df.attrs['ticker'] = df.attrs['ticker']

        metrics = extract_performance_metrics_dict(result_df)
        metrics_list.append(metrics)
        results.append(result_df)

    # Step 4: Print metrics
    metrics_df = pd.DataFrame(metrics_list)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pretty_print_df(metrics_df)

    # Step 5: Plot equity curves
    plot_multiple_equity_curves(results)


def analyze_strategy(strategy_settings):
    df = fetch_data_for_strategy(strategy_settings)

    # ✅ Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"⚠️ Could not convert index to datetime: {e}")

    strategy = strategy_settings["strategy_class"](
        **{k: v for k, v in strategy_settings.items() if k not in ['strategy_class', 'title', 'ticker']}
    )

    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)

    has_fng = 'F&G' in signal_df.columns
    has_vix = 'vix' in signal_df.columns

    plot_funcs = [
        ("Signals", lambda fig: plot_signal_tab(fig, signal_df, has_fng, has_vix)),
        ("Equity Curve", lambda fig: plot_equity_tab(fig, results_df, curve_type="equity")),
        ("Equity vs Benchmark", lambda fig: plot_equity_tab(fig, results_df, curve_type="vs_benchmark")),
    ]

    idx = [0]

    # Create the figure once with constrained_layout enabled
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    def draw():
        fig.clear()
        _, plot_func = plot_funcs[idx[0]]
        plot_func(fig)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(plot_funcs)
            draw()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(plot_funcs)
            draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()

    pretty_print_df(pd.DataFrame([extract_performance_metrics_dict(results_df)]))