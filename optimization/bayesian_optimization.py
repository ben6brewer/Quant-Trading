import optuna
from config.universe_config import *
from utils.data_fetch import *
from backtest.performance_metrics import *
from backtest.backtest_engine import *
from visualizations.plot_equity_curve import *
from strategies.base_strategy import BaseStrategy
import os
import pandas as pd
import hashlib

METRIC_FUNCTIONS = {
    'sharpe': calculate_sharpe_ratio,
    'cagr': calculate_cagr,
    'sortino': calculate_sortino_ratio,
    'max_drawdown': calculate_max_drawdown,
}

def evaluate_strategy(strategy_class, df, params, strategy_settings, performance_metric):
    param_key_map = strategy_settings.get('param_key_map', {})
    mapped_params = {param_key_map.get(k, k): v for k, v in params.items()}

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

    return metric_value, results_df

def run_strategy_optuna_optimization(strategy_class, strategy_settings, performance_metric='sharpe', n_trials=50):
    df = fetch_data_for_strategy(strategy_settings)
    df.attrs['title'] = strategy_settings.get('title', 'Strategy')
    df.attrs['ticker'] = strategy_settings.get('ticker', 'Unknown')

    optimization_params = strategy_settings.get('optimization_params', {})
    validation_func = strategy_settings.get('param_validation')
    title = df.attrs['title']
    ticker = df.attrs['ticker']
    os.makedirs("data/optimization_testcases", exist_ok=True)

    def objective(trial):
        params = {}
        for key, (start, stop, step) in optimization_params.items():
            if isinstance(step, int):
                params[key] = trial.suggest_int(key, int(start), int(stop))
            else:
                params[key] = trial.suggest_float(key, float(start), float(stop))
        if validation_func and not validation_func(params):
            # Return a bad score if params invalid
            return float('inf') if performance_metric != 'max_drawdown' else -float('inf')

        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        filename = f"{title}_{ticker}_{param_hash}.parquet"
        filepath = os.path.join("data/optimization_testcases", filename)

        if os.path.exists(filepath):
            result_df = pd.read_parquet(filepath)
            result_df.attrs['params'] = params
            result_df.attrs['title'] = title
            result_df.attrs['ticker'] = ticker
        else:
            try:
                metric_value, result_df = evaluate_strategy(strategy_class, df, params, strategy_settings, performance_metric)
            except Exception as e:
                print(f"Trial failed: {params} with error {e}")
                return float('inf') if performance_metric != 'max_drawdown' else -float('inf')
            result_df.to_parquet(filepath)

        metric_func = METRIC_FUNCTIONS[performance_metric]
        value = metric_func(result_df)

        # Optuna minimizes objective, so invert metric if maximizing
        return -value if performance_metric != 'max_drawdown' else value

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"\nBest Parameters: {best_params}")
    best_value = -study.best_value if performance_metric != 'max_drawdown' else study.best_value
    print(f"Best {performance_metric}: {best_value:.4f}")

    _, best_df = evaluate_strategy(strategy_class, df, best_params, strategy_settings, performance_metric)

    buy_hold_df = df.copy()
    buy_hold_df['signal'] = 1
    buy_hold_df = BacktestEngine().run_backtest(buy_hold_df)
    buy_hold_df.attrs['title'] = 'Buy & Hold'
    buy_hold_df.attrs['params'] = {}
    buy_hold_df.attrs['ticker'] = df.attrs['ticker']

    plot_grid_search_equity_curves([best_df], best_params, buy_hold_df)

    return best_params, best_df, buy_hold_df
