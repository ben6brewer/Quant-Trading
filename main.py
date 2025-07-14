# main.py

from config.secrets_config import *
from config.universe_config import *
# from alpaca.alpaca_executor import *
from analysis.analyze import *
from utils.fetch_data_for_btc_composite_risk_strategy import *
from visualizations.plot_composite_btc_risk import *

from optimization.bayesian_optimization import *
from utils.pretty_print_df import *
from optimization.grid_search import *
import pandas as pd
from datetime import date

def main():

    # compare_strategies(STRATEGY_SETTINGS_LIST)
    # analyze_strategy(SLOW_FAST_MA_STRATEGY_SETTINGS)
    # analyze_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
    # analyze_strategy(CRYPTO_SENTIMENT_STRATEGY_SETTINGS)
    analyze_strategy(VIX_SPY_STRATEGY_SETTINGS)
    # analyze_strategy(VIX_BTC_STRATEGY_SETTINGS)
    # analyze_strategy(BTC_BUY_AND_HOLD_STRATEGY_SETTINGS)
    # analyze_strategy(SPY_BUY_AND_HOLD_STRATEGY_SETTINGS)
    # plot_btc_with_risk_metric(fetch_data_for_btc_composite_risk_strategy())
    # plot_btc_color_coded_risk_metric(fetch_data_for_btc_composite_risk_strategy())

    # run_strategy_grid_search(strategy_class=VixSpyStrategy, strategy_settings=VIX_SPY_STRATEGY_SETTINGS, performance_metric='sharpe')
    # run_strategy_grid_search(strategy_class=SlowFastMAStrategy, strategy_settings=SLOW_FAST_MA_STRATEGY_SETTINGS, performance_metric='sharpe')
    # run_strategy_optuna_optimization(strategy_class=SlowFastMAStrategy,strategy_settings=SLOW_FAST_MA_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=50)
    # run_strategy_optuna_optimization(strategy_class=VixSpyStrategy,strategy_settings=VIX_SPY_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=1000)
    # run_strategy_optuna_optimization(strategy_class=CryptoSentimentStrategy,strategy_settings=CRYPTO_SENTIMENT_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=5000)

    pass
if __name__ == "__main__":
    main()
