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
    run_crypto_sentiment_strategy()
    # fetch_fear_and_greed_index()
    # run_fifty_week_ma()

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
def run_fifty_week_ma():
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