# main.py

from config.secrets_config import *
from config.universe_config import *
# from alpaca.alpaca_executor import *

from strategies.fifty_week_ma_strategy import *
from strategies.crypto_sentiment_strategy import *

from strategies.vix_btc_strategy import *
from utils.fetch_mvrv_historical import *
from utils.fetch_data_for_btc_composite_risk_strategy import *
from utils.fetch_sopr_historical import *
from utils.fetch_200w_sma_vs_prev_top import *
from utils.fetch_pi_cycle_historical import *
from strategies.vix_spy_strategy import *
from strategies.btc_buy_and_hold_strategy import *
from strategies.spy_buy_and_hold_strategy import *
from optimization.bayesian_optimization import *
from strategies.slow_fast_ma_strategy import *
from utils.data_fetch import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from visualizations.plot_composite_btc_risk import *
from backtest.backtest_engine import *
from utils.pretty_print_df import *
from backtest.performance_metrics import *
from utils.fetch_fear_and_greed_index_data import *
from optimization.grid_search import *
import pandas as pd
import requests
from datetime import date

strategy_list = [
    (CryptoSentimentStrategy, CRYPTO_SENTIMENT_STRATEGY_SETTINGS),
    (FiftyWeekMAStrategy, FIFTY_WEEK_MA_STRATEGY_SETTINGS),
    (VixBtcStrategy, VIX_BTC_STRATEGY_SETTINGS),
    (VixSpyStrategy, VIX_SPY_STRATEGY_SETTINGS),
    (SlowFastMAStrategy, SLOW_FAST_MA_STRATEGY_SETTINGS),
    (BtcBuyAndHoldStrategy, BTC_BUY_AND_HOLD_STRATEGY_SETTINGS),
    (SpyBuyAndHoldStrategy, SPY_BUY_AND_HOLD_STRATEGY_SETTINGS),
]    

def main():

    # compare_strategies(strategy_list)
    # run_slow_fast_ma_strategy()
    # run_fifty_week_ma_strategy()
    # run_crypto_sentiment_strategy()
    # run_vix_spy_strategy()
    # run_vix_btc_strategy()
    # run_strategy_grid_search(strategy_class=VixSpyStrategy, strategy_settings=VIX_SPY_STRATEGY_SETTINGS, performance_metric='sharpe')
    # run_strategy_grid_search(strategy_class=SlowFastMAStrategy, strategy_settings=SLOW_FAST_MA_STRATEGY_SETTINGS, performance_metric='sharpe')
    # run_strategy_optuna_optimization(strategy_class=SlowFastMAStrategy,strategy_settings=SLOW_FAST_MA_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=50)
    # run_strategy_optuna_optimization(strategy_class=VixSpyStrategy,strategy_settings=VIX_SPY_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=1000)
    # run_strategy_optuna_optimization(strategy_class=CryptoSentimentStrategy,strategy_settings=CRYPTO_SENTIMENT_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=5000)
    # plot_btc_with_risk_metric(fetch_data_for_btc_composite_risk_strategy(), risk_col='sma_cycle_risk')
    # plot_btc_color_coded_risk_metric(fetch_data_for_btc_composite_risk_strategy(), risk_col='sma_cycle_risk')
    # plot_btc_with_risk_metric(fetch_data_for_btc_composite_risk_strategy())
    plot_btc_color_coded_risk_metric(fetch_data_for_btc_composite_risk_strategy())
    pass

def compare_strategies(strategy_class_and_settings_list):
    """
    Runs, trims, collects metrics, and plots strategies from a list of (strategy_class, settings_dict) tuples.
    Each settings dict must include 'title' and 'ticker'.
    """
    backtester = BacktestEngine()

    # Step 1: Fetch and tag data
    raw_data = []
    for strategy_class, settings in strategy_class_and_settings_list:
        df = fetch_data_for_strategy(settings)

        # Ensure datetime index
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

        # Attach metadata directly on DataFrame attrs for safer storage
        df.attrs['title'] = settings.get('title', 'Untitled Strategy')
        df.attrs['ticker'] = settings.get('ticker', 'Unknown')

        raw_data.append((strategy_class, df))

    # Step 2: Align on common start date
    latest_start = max(df.index.min() for _, df in raw_data)

    trimmed_data = []
    for strategy_class, df in raw_data:
        trimmed = df[df.index >= latest_start].copy()
        trimmed.attrs['title'] = df.attrs['title']
        trimmed.attrs['ticker'] = df.attrs['ticker']
        trimmed_data.append((strategy_class, trimmed))

    # Step 3: Run each strategy, backtest, and collect metrics
    metrics_list = []
    results = []
    for strategy_class, df in trimmed_data:
        strategy = strategy_class()
        signal_df = strategy.generate_signals(df)
        result_df = backtester.run_backtest(signal_df)

        result_df.attrs['title'] = df.attrs['title']
        result_df.attrs['ticker'] = df.attrs['ticker']

        metrics = extract_performance_metrics_dict(result_df)
        metrics_list.append(metrics)
        results.append(result_df)

    # Step 4: Create a DataFrame of all strategy metrics and print it
    metrics_df = pd.DataFrame(metrics_list)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pretty_print_df(metrics_df)

    # Step 5: Plot equity curves
    plot_multiple_equity_curves(results)



def run_fifty_week_ma_strategy():
    df = fetch_data_for_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
    strategy = FiftyWeekMAStrategy()
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    plot_signals(signal_df)
    results_df = backtester.run_backtest(signal_df)
    # Optional: add these plots back if needed
    # plot_equity_curve(results_df)
    # plot_equity_vs_benchmark(results_df)
    pretty_print_df(pd.DataFrame([extract_performance_metrics_dict(results_df)]))

def run_crypto_sentiment_strategy():
    df = fetch_data_for_strategy(CRYPTO_SENTIMENT_STRATEGY_SETTINGS)
    strategy = CryptoSentimentStrategy()
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)
    plot_signals(signal_df)
    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    pretty_print_df(pd.DataFrame([extract_performance_metrics_dict(results_df)]))


def run_vix_spy_strategy():
    df = fetch_data_for_strategy(VIX_SPY_STRATEGY_SETTINGS)
    strategy = VixSpyStrategy(vix_threshold=36, take_profit_pct=.61, partial_exit_pct=.21)
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)
    results_df['date'] = pd.to_datetime(results_df['date'])
    plot_signals(signal_df)
    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    pretty_print_df(pd.DataFrame([extract_performance_metrics_dict(results_df)]))


def run_vix_btc_strategy():
    df = fetch_data_for_strategy(VIX_BTC_STRATEGY_SETTINGS)
    strategy = VixBtcStrategy()
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)
    plot_signals(signal_df)
    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    pretty_print_df(pd.DataFrame([extract_performance_metrics_dict(results_df)]))


def run_slow_fast_ma_strategy():
    df = fetch_data_for_strategy(SLOW_FAST_MA_STRATEGY_SETTINGS)
    strategy = SlowFastMAStrategy()
    backtester = BacktestEngine()
    signal_df = strategy.generate_signals(df)
    results_df = backtester.run_backtest(signal_df)
    plot_signals(signal_df)
    plot_equity_curve(results_df)
    plot_equity_vs_benchmark(results_df)
    pretty_print_df(pd.DataFrame([extract_performance_metrics_dict(results_df)]))

if __name__ == "__main__":
    main()
