# main.py

from config.secrets_config import *
from config.universe_config import *
# from alpaca.alpaca_executor import *

from strategies.fifty_week_ma_strategy import *
from strategies.crypto_sentiment_strategy import *

from strategies.vix_btc_strategy import *
from strategies.vix_spy_strategy import *
from utils.yfinance_data_fetch import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import *
from utils.pretty_print_df import *
from backtest.performance_metrics import *
from utils.cmc_data_fetch import *
from optimization.grid_search import *
import pandas as pd
import requests
from datetime import date

strategy_list = [
    (CryptoSentimentStrategy, CRYPTO_SENTIMENT_STRATEGY_SETTINGS),
    (FiftyWeekMAStrategy, FIFTY_WEEK_MA_STRATEGY_SETTINGS),
    (VixBtcStrategy, VIX_BTC_STRATEGY_SETTINGS),
    (VixSpyStrategy, VIX_SPY_STRATEGY_SETTINGS)
    ]

def main():

    # compare_strategies(strategy_list)
    
    # run_fifty_week_ma_strategy()
    # run_crypto_sentiment_strategy()
    # run_vix_spy_strategy()
    # run_vix_btc_strategy()
    run_strategy_grid_search(strategy_class=VixSpyStrategy, strategy_settings=VIX_SPY_STRATEGY_SETTINGS, performance_metric='sharpe')
    

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

        # Ensure datetime index here
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                # Try to convert index anyway
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    print(f"Warning: Could not convert index to datetime for {df.ticker if hasattr(df, 'ticker') else 'unknown ticker'}: {e}")

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
    strategy = CryptoSentimentStrategy()

    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    # start_date = date(2022, 10, 1)
    # subset_df = signal_df[signal_df['date'] >= start_date].head(50)
    # pretty_print_df(subset_df)

    # pretty_print_df(signal_df.tail())
    plot_signals(signal_df)
    results_df = backtester.run_backtest(signal_df)
    # pretty_print_df(results_df.tail())
    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    print_performance_metrics(results_df)
def run_fifty_week_ma_strategy():
    # executor = AlpacaExecutor()
    # executor.place_order("AAPL", 1, "buy")
    # executor.liquidate_all_positions()
    df = fetch_data_for_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
    strategy = FiftyWeekMAStrategy()
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    # pretty_print_df(signal_df.tail())
    plot_signals(signal_df)
    results_df = backtester.run_backtest(signal_df)
    # pretty_print_df(results_df.tail())
    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    print_performance_metrics(results_df)

def run_vix_spy_strategy():
    df = fetch_data_for_strategy(VIX_SPY_STRATEGY_SETTINGS)
    strategy = VixSpyStrategy(vix_threshold=10, take_profit_pct=.1, partial_exit_pct=.2)
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)
    # pretty_print_df(results_df.tail())
    # Ensure 'date' column is datetime
    results_df['date'] = pd.to_datetime(results_df['date'])

    # Filter starting from 2020-02-01
    # filtered_df = results_df.loc[results_df['date'] >= '2020-03-04']
    # Show the first 50 rows starting from that date
    # pretty_print_df(filtered_df.head(50))
    # pretty_print_df(signal_df.tail())
    plot_signals(signal_df)

    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    # pretty_print_df(results_df.tail())
    print_performance_metrics(results_df)


def run_vix_btc_strategy():
    df = fetch_data_for_strategy(VIX_BTC_STRATEGY_SETTINGS)
    strategy = VixBtcStrategy()
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)
    # pretty_print_df(results_df.tail())
    # Ensure 'date' column is datetime
    results_df['date'] = pd.to_datetime(results_df['date'])

    # Filter starting from 2020-02-01
    # Show the first 50 rows starting from that date
    # pretty_print_df(filtered_df.head(50))
    # pretty_print_df(signal_df.tail())
    plot_signals(signal_df)

    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    # print_performance_metrics(results_df)
if __name__ == "__main__":
    main()