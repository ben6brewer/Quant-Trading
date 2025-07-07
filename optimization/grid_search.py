# optimization/grid_search.py

from config.universe_config import *
from utils.yfinance_data_fetch import *
from backtest.performance_metrics import *
from backtest.backtest_engine import *
from visualizations.plot_equity_curve import *
from strategies.base_strategy import BaseStrategy
import numpy as np
import itertools
import os
import pandas as pd
import hashlib

# Mapping performance metrics to functions
METRIC_FUNCTIONS = {
    'sharpe': calculate_sharpe_ratio,
    'cagr': calculate_cagr,
    'sortino': calculate_sortino_ratio,
    'max_drawdown': calculate_max_drawdown,
}

def evaluate_strategy(strategy_class, df, params, strategy_settings, performance_metric):
    # Create and configure strategy
    strategy = strategy_class()
    for param, val in params.items():
        setattr(strategy, param, val)

    df_copy = df.copy(deep=True)
    df_copy.attrs = df.attrs.copy()

    # Run signals and backtest
    signal_df = strategy.generate_signals(df_copy)
    results_df = BacktestEngine().run_backtest(signal_df)

    # Calculate performance
    metric_func = METRIC_FUNCTIONS[performance_metric]
    metric_value = metric_func(results_df)

    # Attach metadata
    results_df.attrs['params'] = params
    results_df.attrs['title'] = df_copy.attrs.get('title', 'Unknown')
    results_df.attrs['ticker'] = df_copy.attrs.get('ticker', 'Unknown')

    return metric_value, params, results_df


def run_strategy_grid_search(strategy_class, strategy_settings, performance_metric='sharpe'):
    df = fetch_data_for_strategy(strategy_settings)

    df.attrs['title'] = strategy_settings.get('title', 'Strategy')
    df.attrs['ticker'] = strategy_settings.get('ticker', 'Unknown')

    optimization_params = strategy_settings.get('optimization_params', {})
    grid_search_params = {
        key: np.round(np.arange(start, stop, step), 6).tolist()
        for key, (start, stop, step) in optimization_params.items()
    }

    keys = list(grid_search_params.keys())
    param_combinations = list(itertools.product(*grid_search_params.values()))
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to test: {total_combinations}\n")

    os.makedirs("data/optimization_testcases", exist_ok=True)

    results_dfs = []
    best_metric_value = -np.inf if performance_metric != 'max_drawdown' else np.inf
    best_params = None
    best_results_df = None

    title = df.attrs['title']
    ticker = df.attrs['ticker']

    for i, values in enumerate(param_combinations, 1):
        params = dict(zip(keys, values))
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        filename = f"{title}_{ticker}_{param_hash}.parquet"
        filepath = os.path.join("data/optimization_testcases", filename)

        if os.path.exists(filepath):
            result_df = pd.read_parquet(filepath)
            result_df.attrs['params'] = params
            result_df.attrs['title'] = title
            result_df.attrs['ticker'] = ticker
            print(f"Loaded from cache ({i}/{total_combinations}): {filepath}")
        else:
            print(f"Evaluating ({i}/{total_combinations}): " + ", ".join(f"{k}={v}" for k, v in params.items()))
            metric_value, params, result_df = evaluate_strategy(strategy_class, df, params, strategy_settings, performance_metric)
            result_df.to_parquet(filepath)
            print(f"Saved: {filepath}")

        results_dfs.append(result_df)

        metric_func = METRIC_FUNCTIONS[performance_metric]
        metric_value = metric_func(result_df)

        is_better = (
            metric_value < best_metric_value if performance_metric == 'max_drawdown'
            else metric_value > best_metric_value
        )

        if is_better:
            best_metric_value = metric_value
            best_params = params
            best_results_df = result_df

    print(f"\nBest {performance_metric}: {best_metric_value:.4f}")
    print(f"Best Parameters: {best_params}")

    plot_grid_search_equity_curves(results_dfs, best_params)

    return results_dfs, best_params, best_results_df
