# optimization/grid_search.py

from config.universe_config import *
from utils.yfinance_data_fetch import *
from backtest.performance_metrics import *
from backtest.backtest_engine import *
from visualizations.plot_equity_curve import *
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import itertools

# Mapping performance metrics to functions
METRIC_FUNCTIONS = {
    'sharpe': calculate_sharpe_ratio,
    'cagr': calculate_cagr,
    'sortino': calculate_sortino_ratio,
    'max_drawdown': calculate_max_drawdown,
}

def evaluate_strategy(args):
    strategy_class, df, params, strategy_settings, performance_metric = args

    # Create a fresh instance of the strategy
    strategy = strategy_class()
    for param, val in params.items():
        setattr(strategy, param, val)

    # Copy the DataFrame and metadata
    df_copy = df.copy(deep=True)
    df_copy.attrs = df.attrs.copy()

    # Run strategy and backtest
    signal_df = strategy.generate_signals(df_copy)
    results_df = BacktestEngine().run_backtest(signal_df)

    # Calculate performance metric
    metric_func = METRIC_FUNCTIONS[performance_metric]
    metric_value = metric_func(results_df)

    # Attach metadata for later use
    results_df.attrs['params'] = params
    results_df.attrs['title'] = df_copy.attrs.get('title', 'Unknown')
    results_df.attrs['ticker'] = df_copy.attrs.get('ticker', 'Unknown')

    return metric_value, params, results_df


def run_strategy_grid_search(strategy_class, strategy_settings, performance_metric='sharpe'):
    # Fetch historical data
    df = fetch_data_for_strategy(strategy_settings)

    # Add metadata to df
    df.attrs['title'] = strategy_settings.get('title', 'Strategy')
    df.attrs['ticker'] = strategy_settings.get('ticker', 'Unknown')

    # Generate grid from parameter ranges
    optimization_params = strategy_settings.get('optimization_params', {})
    grid_search_params = {
        key: np.round(np.arange(start, stop, step), 6).tolist()
        for key, (start, stop, step) in optimization_params.items()
    }

    keys = list(grid_search_params.keys())
    param_combinations = list(itertools.product(*grid_search_params.values()))
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to test: {total_combinations}\n")

    # Prepare arguments for multiprocessing
    args_list = [
        (strategy_class, df, dict(zip(keys, values)), strategy_settings, performance_metric)
        for values in param_combinations
    ]

    # Tracking best result
    best_metric_value = -np.inf if performance_metric != 'max_drawdown' else np.inf
    best_params = None
    best_results_df = None
    results_dfs = []

    # Parallel evaluation
    with ProcessPoolExecutor() as executor:
        for i, (metric_value, params, results_df) in enumerate(executor.map(evaluate_strategy, args_list), 1):
            print(f"Evaluated combination {i} / {total_combinations}: " +
                  ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()))

            if performance_metric == 'max_drawdown':
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    best_results_df = results_df
            else:
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    best_results_df = results_df

            results_dfs.append(results_df)

    print(f"\nBest {performance_metric}: {best_metric_value:.4f}")
    print(f"Best Parameters: {best_params}")

    # Plot equity curves for all combinations, highlighting the best one
    plot_grid_search_equity_curves(results_dfs, best_params)

    return results_dfs, best_params, best_results_df
