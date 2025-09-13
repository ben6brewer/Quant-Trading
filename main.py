# main.py

from config.secrets_config import *
from config.universe_config import *
# from alpaca.alpaca_executor import *
from analysis.analyze import *
from utils.fetch_data_for_btc_composite_risk_strategy import *
from visualizations.plot_composite_btc_risk import *
from visualizations.plot_analyst import *
from utils.parse_analyst_excel import *

from optimization.bayesian_optimization import *
from utils.pretty_print_df import *
from optimization.grid_search import *
import pandas as pd
from datetime import date

from utils.fetch_sp500_historical_tickers import *
from utils.fetch_equity_metric_data import *
from utils.fetch_tickers_by_metric import *

def main():
    # preprocess_analyst_data()
    metrics = [
        "Buy %",         # normalized to 0–1
        "Hold %",        # normalized to 0–1
        "Sell %",        # normalized to 0–1
        "Buy % - Sell %",      # Buy % − Sell % (−1..1 range)
        "Target Spread", # Target Price − Last Price (absolute $ difference)
        "Price-Target",
        "Upside %"       # (Target / Last − 1), implied % return
    ]

    # plot_analyst( metric_to_plot="Buy % - Sell %", dfs=load_many_analyst_dfs(["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"]), names=["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"])
    plot_analyst_color_coded( metric_to_plot="Buy % - Sell %", dfs=load_many_analyst_dfs(["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"]), names=["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"])


    # compare_strategies(STRATEGY_SETTINGS_LIST)
    # compare_strategies([SPY_BUY_AND_HOLD_STRATEGY_SETTINGS, TSN_BUY_AND_HOLD_STRATEGY_SETTINGS, XLP_BUY_AND_HOLD_STRATEGY_SETTINGS])
    # compare_strategies([SPY_BUY_AND_HOLD_STRATEGY_SETTINGS, TSN_BUY_AND_HOLD_STRATEGY_SETTINGS, CAG_BUY_AND_HOLD_STRATEGY_SETTINGS, HRL_BUY_AND_HOLD_STRATEGY_SETTINGS, CPB_BUY_AND_HOLD_STRATEGY_SETTINGS, GIS_BUY_AND_HOLD_STRATEGY_SETTINGS, XLP_BUY_AND_HOLD_STRATEGY_SETTINGS])
    # compare_strategies([SPY_BUY_AND_HOLD_STRATEGY_SETTINGS, TSN_BUY_AND_HOLD_STRATEGY_SETTINGS, XLP_BUY_AND_HOLD_STRATEGY_SETTINGS])

    # analyze_strategy(SLOW_FAST_MA_STRATEGY_SETTINGS)
    # analyze_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
    # analyze_strategy(CRYPTO_SENTIMENT_STRATEGY_SETTINGS)
    # analyze_strategy(VIX_SPY_STRATEGY_SETTINGS)
    # analyze_strategy(VIX_BTC_STRATEGY_SETTINGS)
    # analyze_strategy(BTC_BUY_AND_HOLD_STRATEGY_SETTINGS)
    # analyze_strategy(SPY_BUY_AND_HOLD_STRATEGY_SETTINGS)
    # plot_btc_with_risk_metric(fetch_data_for_btc_composite_risk_strategy())
    # plot_btc_color_coded_risk_metric(fetch_data_for_btc_composite_risk_strategy())

    # equity_ticker_list = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
    # metric = "trailingPE"
    # pretty_print_df(fetch_equity_metric_data_list(equity_ticker_list, metric).head(10))

    # fetch_sp500_historical_tickers()
    # update_all_sp500_tickers()
    # pretty_print_df(fetch_top_tickers_by_metric(10, "trailingPE", "quarter").tail())

    # run_strategy_grid_search(strategy_class=VixSpyStrategy, strategy_settings=VIX_SPY_STRATEGY_SETTINGS, performance_metric='sharpe')
    # run_strategy_grid_search(strategy_class=SlowFastMAStrategy, strategy_settings=SLOW_FAST_MA_STRATEGY_SETTINGS, performance_metric='sharpe')
    # run_strategy_optuna_optimization(strategy_class=SlowFastMAStrategy,strategy_settings=SLOW_FAST_MA_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=50)
    # run_strategy_optuna_optimization(strategy_class=VixSpyStrategy,strategy_settings=VIX_SPY_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=1000)
    # run_strategy_optuna_optimization(strategy_class=CryptoSentimentStrategy,strategy_settings=CRYPTO_SENTIMENT_STRATEGY_SETTINGS,performance_metric='sharpe',n_trials=5000)

    # analyze_strategy(TSN_BUY_AND_HOLD_STRATEGY_SETTINGS)
    # analyze_strategy(SPY_BUY_AND_HOLD_STRATEGY_SETTINGS)


    pass
if __name__ == "__main__":
    main()
