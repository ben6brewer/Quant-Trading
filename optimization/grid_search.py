# optimization/grid_search.py

from config.universe_config import *
from utils.data_fetch import *
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

def generate_valid_combinations(grid_search_params, keys, validation_func=None):
    for combo in itertools.product(*grid_search_params.values()):
        params = dict(zip(keys, combo))
        if validation_func is None or validation_func(params):
            yield combo



def evaluate_strategy(strategy_class, df, params, strategy_settings, performance_metric):
    # Map params keys if needed
    param_key_map = strategy_settings.get('param_key_map', {})
    mapped_params = {param_key_map.get(k, k): v for k, v in params.items()}

    # You can add type casting here if needed, e.g. int or float for specific params

    strategy = strategy_class(**mapped_params)

    df_copy = df.copy(deep=True)
    df_copy.attrs = df.attrs.copy()

    signal_df = strategy.generate_signals(df_copy)
    results_df = BacktestEngine().run_backtest(signal_df)

    metric_func = METRIC_FUNCTIONS[performance_metric]
    metric_value = metric_func(results_df)

    results_df.attrs['params'] = params
    results_df.attrs['title'] = df_copy.attrs.get('title', 'Unknown')
    results_df.attrs['ticker'] = df_copy.attrs.get('ticker', 'Unknown')

    return metric_value, params, results_df

def run_strategy_grid_search(strategy_class, strategy_settings, performance_metric='sharpe'):
    """
    Runs a grid search optimization over the parameters specified in strategy_settings.
    
    Args:
        strategy_class: The strategy class to instantiate and test.
        strategy_settings: Dictionary containing keys:
            - 'optimization_params': dict of param: (start, stop, step)
            - Optional 'param_validation': callable(params_dict) -> bool
            - Optional 'param_key_map': dict mapping grid param keys to strategy constructor keys
            - Optional 'title' and 'ticker' strings
        performance_metric: One of the keys in METRIC_FUNCTIONS indicating the metric to optimize.
        
    Returns:
        results_dfs: List of DataFrames from all tested parameter combinations.
        best_params: The parameter dictionary of the best performing run.
        best_results_df: The DataFrame results of the best performing run.
        buy_hold_df: A benchmark DataFrame representing buy-and-hold equity.
    """
    df = fetch_data_for_strategy(strategy_settings)

    df.attrs['title'] = strategy_settings.get('title', 'Strategy')
    df.attrs['ticker'] = strategy_settings.get('ticker', 'Unknown')

    optimization_params = strategy_settings.get('optimization_params', {})
    grid_search_params = {
        key: np.round(np.arange(start, stop, step), 6).tolist()
        for key, (start, stop, step) in optimization_params.items()
    }

    keys = list(grid_search_params.keys())
    validation_func = strategy_settings.get('param_validation')

    param_combinations = list(generate_valid_combinations(grid_search_params, keys, validation_func))
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

        # Create a unique hash for caching
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
            try:
                metric_value, _, result_df = evaluate_strategy(strategy_class, df, params, strategy_settings, performance_metric)
            except Exception as e:
                print(f"Failed evaluation for params {params}: {e}")
                continue

            result_df.to_parquet(filepath)
            print(f"Saved: {filepath}")

        results_dfs.append(result_df)

        # Recalculate metric in case loaded from cache
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

    # Generate Buy & Hold Benchmark
    buy_hold_df = df.copy()
    buy_hold_df['signal'] = 1  # always long
    buy_hold_df = BacktestEngine().run_backtest(buy_hold_df)
    buy_hold_df.attrs['title'] = 'Buy & Hold'
    buy_hold_df.attrs['ticker'] = ticker
    buy_hold_df.attrs['params'] = {}

    plot_grid_search_equity_curves(results_dfs, best_params, buy_hold_df)

    return results_dfs, best_params, best_results_df, buy_hold_df

