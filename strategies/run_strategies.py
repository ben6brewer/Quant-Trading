from datetime import date
from config.secrets_config import *
from config.universe_config import *
#from alpaca.alpaca_executor import *

from strategies.fifty_week_ma_strategy import *
from strategies.crypto_sentiment_strategy import *
from strategies.vix_spy_strategy import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import *
from backtest.performance_metrics import *
from utils.pretty_print_df import *
from utils.cmc_data_fetch import *
from utils.yfinance_data_fetch import *

import pandas as pd
pd.set_option('display.float_format', '{:,.2f}'.format)

def compare_strategies(strategy_class_and_settings_list):
    """
    Runs, trims, prints performance, and plots strategies from a list of (strategy_class, settings_dict) tuples.
    Each settings dict must include 'title' and 'ticker'.
    """
    backtester = BacktestEngine()

    # Step 1: Fetch and tag data
    raw_data = []
    for strategy_class, settings in strategy_class_and_settings_list:
        df = fetch_data_for_strategy(settings)
        df.title = settings.get('title', 'Untitled Strategy')
        df.ticker = settings.get('ticker', 'Unknown')
        raw_data.append((strategy_class, df))

    # Step 2: Align on common start date
    latest_start = max(df.index.min() for _, df in raw_data)

    trimmed_data = []
    for strategy_class, df in raw_data:
        trimmed = df[df.index >= latest_start].copy()
        trimmed.title = df.title
        trimmed.ticker = df.ticker
        trimmed_data.append((strategy_class, trimmed))

    # Step 3: Run each strategy, backtest, and print metrics
    results = []
    for strategy_class, df in trimmed_data:
        strategy = strategy_class()
        signal_df = strategy.generate_signals(df)
        result_df = backtester.run_backtest(signal_df)
        result_df.title = df.title
        result_df.ticker = df.ticker
        print(f"\n📊 Performance for {result_df.ticker} - {result_df.title}:")
        print_performance_metrics(result_df)
        results.append(result_df)

    # Step 4: Plot equity curves
    plot_multiple_equity_curves(results)

def run_crypto_sentiment_strategy():
    df = fetch_data_for_strategy(CRYPTO_SENTIMENT_STRATEGY_SETTINGS)
    pretty_print_df(df.tail())
    # strategy = CryptoSentimentStrategy()

    # backtester = BacktestEngine()
    # signal_df = strategy.generate_signals(df)
    # # start_date = date(2022, 10, 1)
    # # subset_df = signal_df[signal_df['date'] >= start_date].head(50)
    # # pretty_print_df(subset_df)

    # # pretty_print_df(signal_df.tail())
    # plot_signals(signal_df)
    # results_df = backtester.run_backtest(signal_df)
    # # pretty_print_df(results_df.tail())
    # plot_equity_curve(results_df)
    # plot_equity_vs_benchmark(results_df)
    # print_performance_metrics(results_df)
def run_fifty_week_ma_strategy():
    # executor = AlpacaExecutor()
    # executor.place_order("AAPL", 1, "buy")
    # executor.liquidate_all_positions()
    df = fetch_data_for_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
    pretty_print_df(df.tail())
    # strategy = FiftyWeekMAStrategy()
    # backtester = BacktestEngine()
    # signal_df = strategy.generate_signals(df)
    # # pretty_print_df(signal_df.tail())
    # plot_signals(signal_df)
    # results_df = backtester.run_backtest(signal_df)
    # # pretty_print_df(results_df.tail())
    # plot_equity_curve(results_df)
    # plot_equity_vs_benchmark(results_df)
    # print_performance_metrics(results_df)

def run_vix_spy_strategy():
    # 1. Fetch and generate signals
    df = fetch_data_for_strategy(VIX_SPY_STRATEGY_SETTINGS)
    strategy = VixSpyStrategy(vix_threshold=45.0)  # Adjust VIX threshold here
    backtester = BacktestEngine()

    signal_df = strategy.generate_signals(df)

    # 2. Ensure the index is a datetime format
    signal_df.index = pd.to_datetime(signal_df.index)

    # 3. Filter signals from a start date
    start_date = '2008-12-20'
    subset_df = signal_df.loc[start_date:]

    # 4. Preserve metadata after slicing
    subset_df.title = getattr(signal_df, 'title', 'No Title')
    subset_df.ticker = getattr(signal_df, 'ticker', 'No Ticker')

    # 5. Print signals subset
    pretty_print_df(subset_df.head(50))

    # 6. Plot signals
    plot_signals(subset_df)

    # 7. Run backtest using the filtered signal data
    results_df = backtester.run_backtest(subset_df)  # Adjust take-profit %

    # Add close price to results_df for benchmarking plots
    results_df['close'] = subset_df['close']

    # 8. Ensure datetime index
    results_df.index = pd.to_datetime(results_df.index)

    # 9. Filter results from same start date
    results_subset = results_df.loc[start_date:]

    # ✅ Reapply custom metadata after slicing
    results_subset.title = getattr(results_df, 'title', 'No Title')
    results_subset.ticker = getattr(results_df, 'ticker', 'No Ticker')

    # 10. Plot equity curve and benchmark
    plot_equity_curve(results_subset)
    plot_equity_vs_benchmark(results_subset)

    # 11. Plot cash vs holdings over time
    results_subset[['cash', 'holdings']].plot(
        title='Cash vs Equity Holdings Over Time', figsize=(12, 6)
    )

    # 12. Preview position data
    preview_cols = ['cash', 'holdings', 'total_equity', 'position', 'trade']
    pretty_print_df(results_subset[preview_cols].head(50))

    # 13. Optional: show signal subset again
    subset_df_reset = subset_df.reset_index()
    pretty_print_df(subset_df_reset.head(10))

    # 14. Print performance metrics only for filtered range
    print_performance_metrics(results_subset)
