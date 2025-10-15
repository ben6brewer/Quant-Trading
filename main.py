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

from strategies.endowment.static_mix_iwv_agg_strategy import StaticMixIWVAGG
from backtest.backtest_engine import BacktestEngine

from strategies.endowment.static_mix_iwv_vxus_agg import StaticMixIWV_VXUS_AGG

def main():
    # compare_strategies(build_iwv_agg_grid())
    # compare_strategies(build_iwv_vxus_agg_grid())



    # compare_strategies(strategy_settings_list=STRATEGY_SETTINGS_LIST)
    # ANALYST INSIGHTS
    # preprocess_analyst_data()
    # metrics = [
    #     "Buy %",         # normalized to 0–1
    #     "Hold %",        # normalized to 0–1
    #     "Sell %",        # normalized to 0–1
    #     "Buy % - Sell %",      # Buy % − Sell % (−1..1 range)
    #     "Target Spread", # Target Price − Last Price (absolute $ difference)
    #     "Price-Target",
    #     "Upside %"       # (Target / Last − 1), implied % return
    # ]

    
    # plot_analyst(metric_to_plot="Upside %", dfs=load_many_analyst_dfs(["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"]), names=["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"])
    # plot_analyst_color_coded( metric_to_plot="Upside %", dfs=load_many_analyst_dfs(["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"]), names=["NVDA", "AAPL", "MSFT", "CRM", "CRWD", "CSCO", "INTC", "TSM"])
    # analyze_strategy(UNIVERSITY_ENDOWMENT_SPENDING_STRATEGY_SETTINGS)
    fig, _ = plot_endowment_payout_comparison(
    UNIVERSITY_ENDOWMENT_SPENDING_STRATEGY_SETTINGS,
    label_dates=("2009-03-13", "2020-03-25"),
    add_end_payout_labels=True
    )
    # then:
    import matplotlib.pyplot as plt
    plt.show(block=True)


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

def build_iwv_vxus_agg_grid(
    start: float = 0.0,
    stop: float = 1.0,
    step: float = 0.05,
    mix_list: list[str] | None = None
):
    # mix_list = ['100/0/0','7/0/93','0/100/0', '0/0/100', '85/0/15', '70/0/30']
    """
    Build a grid of IWV/VXUS/AGG static mixes over the simplex.
    Returns a list[dict] suitable for compare_strategies().
    """
    settings = []

    def add(iwv_w: float, vxus_w: float, agg_w: float):
        iwv_pct  = int(round(iwv_w  * 100))
        vxus_pct = int(round(vxus_w * 100))
        agg_pct  = int(round(agg_w  * 100))
        label = f"{iwv_pct}/{vxus_pct}/{agg_pct} IWV/VXUS/AGG"
        settings.append({
            "strategy_class": StaticMixIWV_VXUS_AGG,
            "w_iwv":  iwv_w,
            "w_vxus": vxus_w,
            "w_agg":  agg_w,
            "title": label,
            "label": label,
            "interval": "1d",
            "engine_class": BacktestEngine,
            "engine_kwargs": {}
        })

    # 1) Explicit list mode
    if mix_list:
        for mix in mix_list:
            try:
                a,b,c = (int(p.strip()) for p in mix.split('/'))
                if a + b + c != 100:
                    raise ValueError("percentages must sum to 100")
                add(a/100.0, b/100.0, c/100.0)
            except Exception as e:
                print(f"⚠️ Skipping invalid mix '{mix}': {e}")
        return settings

    # 2) Grid sweep mode over integer simplex
    # Convert to integer grid: weights are i/m, j/m, k/m with i+j+k=m
    m = int(round(1.0 / step))
    if abs(m*step - 1.0) > 1e-12:
        raise ValueError("step must divide 1.0 exactly, e.g., 0.01, 0.02, 0.05")

    a = int(round(start * m))
    b = int(round(stop  * m))
    if not (0 <= a <= b <= m):
        raise ValueError("start/stop must satisfy 0 ≤ start ≤ stop ≤ 1")

    # Ensure all three weights lie within [a,b] and sum to m.
    # i runs until there is room for j and k >= a
    i_max = min(b, m - 2*a)
    for i in range(a, i_max + 1):
        # j must be ≥ a, ≤ b, and leave k = m - i - j ≥ a  ⇒ j ≤ m - i - a
        j_max = min(b, m - i - a)
        for j in range(a, j_max + 1):
            k = m - i - j
            if k < a or k > b:
                continue
            add(i/m, j/m, k/m)

    return settings



def build_iwv_agg_grid(start: float = 0.0, stop: float = 1.0, step: float = 0.01, mix_list: list[str] | None = None):
    """
    Build a grid of IWV/AGG static mixes. IWV in [start, stop] with step; AGG = 1 - IWV.
    Returns list[dict] for compare_strategies().
    """
    settings = []
    mix_list = ['8/92','54/46', '100/0']

    def add(w_iwv: float):
        w_agg = 1.0 - w_iwv
        eq_pct = int(round(w_iwv * 100))
        fi_pct = 100 - eq_pct
        label = f"{eq_pct}/{fi_pct} IWV/AGG"
        settings.append({
            "strategy_class": StaticMixIWVAGG,
            "w_iwv": w_iwv,
            "w_agg": w_agg,
            "title": label,
            "label": label,
            "interval": "1d",
            "engine_class": BacktestEngine,
            "engine_kwargs": {}
        })

    # 1) Explicit list mode
    if mix_list:
        for mix in mix_list:
            try:
                eq_str, fi_str = mix.split('/')
                eq_pct = int(eq_str.strip())
                fi_pct = int(fi_str.strip())
                if eq_pct + fi_pct != 100:
                    raise ValueError("percentages must sum to 100")
                add(eq_pct / 100.0)
            except Exception as e:
                print(f"⚠️ Skipping invalid mix '{mix}': {e}")
        return settings

    # 2) Grid sweep mode using integer simplex (no float drift)
    m = int(round(1.0 / step))
    if abs(m * step - 1.0) > 1e-12:
        raise ValueError("step must divide 1.0 exactly (e.g., 0.01, 0.02, 0.05)")
    a = int(round(start * m))
    b = int(round(stop * m))
    if not (0 <= a <= b <= m):
        raise ValueError("start/stop must satisfy 0 ≤ start ≤ stop ≤ 1")

    for i in range(a, b + 1):
        add(i / m)

    return settings


if __name__ == "__main__":
    main()
