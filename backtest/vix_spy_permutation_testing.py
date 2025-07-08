# backtest/vix_spy_permutation_testing.py

# backtest/vix_spy_permutation_testing.py

import optuna
import numpy as np
import pandas as pd
import traceback
import time

from pathlib import Path

from strategies.vix_spy_strategy import VixSpyStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.performance_metrics import *
from utils.yfinance_data_fetch import *
from config.universe_config import *

# Optional: for caching results if you want
CACHE_DIR = Path("data/backtest_optuna_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def run_backtest(params: dict, data: pd.DataFrame) -> dict | None:
    """
    Run backtest for given parameters and return all metrics as dict.
    Return None if failed.
    """
    try:
        strategy = VixSpyStrategy(**params)
        signal_df = strategy.generate_signals(data)
        if signal_df is None or signal_df.empty:
            return None

        result_df = BacktestEngine().run_backtest(signal_df)

        sharpe = calculate_sharpe_ratio(result_df)
        cagr = calculate_cagr(result_df)
        sortino = calculate_sortino_ratio(result_df)
        max_dd = calculate_max_drawdown(result_df)

        # If any metric is nan, fail
        if any(np.isnan(x) for x in [sharpe, cagr, sortino, max_dd]):
            return None

        return {
            "sharpe_ratio": round(sharpe, 3),
            "cagr": round(cagr, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(max_dd, 3),
        }

    except Exception:
        print(f"❌ Failed for params {params}:")
        traceback.print_exc()
        return None


def objective(trial: optuna.Trial, data: pd.DataFrame) -> float:
    vix_threshold = trial.suggest_int("vix_threshold", 1, 101)
    take_profit_pct = trial.suggest_float("take_profit_pct", 0.01, 2.51, step=0.01)
    partial_exit_pct = trial.suggest_float("partial_exit_pct", 0.01, 0.25, step=0.01)

    params = {
        "vix_threshold": vix_threshold,
        "take_profit_pct": take_profit_pct,
        "partial_exit_pct": partial_exit_pct,
    }

    metrics = run_backtest(params, data)
    if metrics is None:
        trial.report(-float("inf"), step=0)
        raise optuna.exceptions.TrialPruned()

    # Set user attributes for all metrics except objective
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    # Report intermediate value for pruning
    trial.report(metrics["sharpe_ratio"], step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return metrics["sharpe_ratio"]


def run_optimization(n_trials=500):
    data = fetch_data_for_strategy(VIX_SPY_STRATEGY_SETTINGS)
    data = ensure_datetime_index(data)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    print(f"🔄 Starting Optuna optimization for {n_trials} trials...")
    start_time = time.time()

    study.optimize(lambda trial: objective(trial, data), n_trials=n_trials, timeout=None)

    print(f"✅ Optimization complete. Best trial:")
    best = study.best_trial
    print(f"  Value (Sharpe ratio): {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    elapsed = time.time() - start_time
    print(f"⏳ Total time: {elapsed/60:.2f} minutes")

    # Extract all trial results including user attrs into a DataFrame
    records = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rec = {**t.params, "sharpe_ratio": t.value}
            rec.update(t.user_attrs)  # Add cagr, sortino, max_drawdown, etc.
            records.append(rec)

    df_results = pd.DataFrame(records)
    df_results.to_csv("data/vix_optuna_results.csv", index=False)
    print("✅ All trial results saved to data/vix_optuna_results.csv.")

    return df_results

