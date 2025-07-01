# main.py

from config.secrets_config import *
from config.universe_config import *
# from alpaca.alpaca_executor import *

from strategies.fifty_week_ma_strategy import *
from strategies.crypto_sentiment_strategy import *

from utils.yfinance_data_fetch import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import *
from utils.pretty_print_df import *
from backtest.performance_metrics import *
from utils.cmc_data_fetch import *
import pandas as pd
import requests
from datetime import date

pd.set_option('display.float_format', '{:,.2f}'.format)


def main():
    strategy_list = [
        (CryptoSentimentStrategy, CRYPTO_SENTIMENT_STRATEGY_SETTINGS),
        (FiftyWeekMAStrategy, FIFTY_WEEK_MA_STRATEGY_SETTINGS)
        ]
    # compare_strategies(strategy_list)
    
    run_fifty_week_ma_strategy()
    # run_crypto_sentiment_strategy()
    

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
    # plot_equity_curve(results_df)
    # plot_equity_vs_benchmark(results_df)
    # print_performance_metrics(results_df)

if __name__ == "__main__":
    main()