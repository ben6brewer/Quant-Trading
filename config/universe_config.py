# config/universe_config.py

from strategies.endowment.static_mix_iwv_agg_strategy import StaticMixIWVAGG
from strategies.moving_averages.slow_fast_ma_strategy import SlowFastMAStrategy
from strategies.moving_averages.fifty_week_ma_strategy import FiftyWeekMAStrategy
from strategies.sentiment.crypto_sentiment_strategy import CryptoSentimentStrategy
from strategies.vix.vix_btc_strategy import VixBtcStrategy
from strategies.vix.vix_spy_strategy import VixSpyStrategy
from strategies.buy_and_hold.btc_buy_and_hold_strategy import BtcBuyAndHoldStrategy
from strategies.buy_and_hold.spy_buy_and_hold_strategy import SpyBuyAndHoldStrategy
from strategies.buy_and_hold.tsn_buy_and_hold_strategy import TsnBuyAndHoldStrategy
from strategies.buy_and_hold.xlp_buy_and_hold_strategy import XlpBuyAndHoldStrategy
from strategies.buy_and_hold.cag_buy_and_hold_strategy import CagBuyAndHoldStrategy
from strategies.buy_and_hold.hrl_buy_and_hold_strategy import HrlBuyAndHoldStrategy
from strategies.buy_and_hold.cpb_buy_and_hold_strategy import CpbBuyAndHoldStrategy
from strategies.buy_and_hold.jbs_buy_and_hold_strategy import GisBuyAndHoldStrategy
from strategies.buy_and_hold.amzn_buy_and_hold_strategy import AmznBuyAndHoldStrategy
from strategies.buy_and_hold.nvda_buy_and_hold_strategy import NvdaBuyAndHoldStrategy
from strategies.buy_and_hold.azo_buy_and_hold_strategy import AzoBuyAndHoldStrategy
from strategies.endowment.university_endowment_buy_and_hold_strategy import UniversityEndowmentSpendingStrategy
from backtest.endowment_payout_engine import * 

SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "V", "UNH", "NVDA", "XOM", "JNJ", "PG", "HD", "DIS"
]

PE_STRATEGY_SETTINGS = {
    "quantile": 0.2,
    "min_pe": 2.0,
    "max_pe": 50.0,
    "rebalance_frequency": "monthly",
}

FIFTY_WEEK_MA_STRATEGY_SETTINGS = {
    "title": "50 week MA Strategy",
    "ticker": "BTC-USD",
    "strategy_class": FiftyWeekMAStrategy,
    "period": "max",
    "interval": "1d",
}

CRYPTO_SENTIMENT_STRATEGY_SETTINGS = {
    "title": "Crypto Sentiment Strategy",
    "ticker": "BTC-USD",
    "strategy_class": CryptoSentimentStrategy,
    "period": "max",
    "interval": "1d",
    "optimization_params": {
    "fear_threshold": (1, 100, 1),
    "greed_threshold": (1, 100, 1),
    "fear_days_required": (1, 100, 1),
    "greed_days_required": (1, 100, 1)
    },
    "optimized_params": {
        "fear_threshold": 22,
        "greed_threshold": 80,
        "fear_days_required": 20,
        "greed_days_required": 10
    }
}

VIX_SPY_STRATEGY_SETTINGS = {
    "title": "VIX Strategy",
    "ticker": "SPY",
    "strategy_class": VixSpyStrategy,
    "period": "max",
    "interval": "1d",
    "optimization_params": {
        "vix_threshold": (1.0, 100.0, 1),
        "take_profit_pct": (0.01, 1.0, 0.01),
        "partial_exit_pct": (0.01, 1.0, 0.01)
    }
}

VIX_BTC_STRATEGY_SETTINGS = {
    "title": "VIX Strategy",
    "ticker": "BTC-USD",
    "strategy_class": VixBtcStrategy,
    "period": "max",
    "interval": "1d",
}

SLOW_FAST_MA_STRATEGY_SETTINGS = {
    "title": "Slow Fast MA Strategy",
    "ticker": "BTC-USD",
    "strategy_class": SlowFastMAStrategy,
    "period": "max",
    "interval": "1d",
    'param_validation': lambda params: params.get('slow_ma', 0) > params.get('fast_ma', 0),
    "optimization_params": {
        "slow_ma": (1, 400.0, 1),
        "fast_ma": (1, 200.0, 1),
    },
    "optimized_params": {
        "slow_ma": 124,
        "fast_ma": 2
    }
}

BTC_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "BTC-USD",
    "strategy_class": BtcBuyAndHoldStrategy,
    "period": "max",
    "interval": "1d",
}

CAG_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "CAG",
    "strategy_class": CagBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
    "type": "equity",
}

HRL_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "HRL",
    "strategy_class": HrlBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
    "type": "equity",
}

CPB_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "CPB",
    "strategy_class": CpbBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
    "type": "equity",
}

GIS_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "GIS",
    "strategy_class": GisBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
    "type": "equity",
}

SPY_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "SPY",
    "strategy_class": SpyBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
}

TSN_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "TSN",
    "strategy_class": TsnBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
    "type": "equity",
}

XLP_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "XLP",
    "strategy_class": XlpBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
    "type": "equity",
}

# UNIVERSITY_ENDOWMENT_SPENDING_STRATEGY_SETTINGS = {
#     "title": "Endowment Spending 70/30",
#     "ticker": "IWV/AGG",
#     "strategy_class": UniversityEndowmentSpendingStrategy,
#     "engine_class": EndowmentPayoutBacktestEngine,
#     # "start": "2015-01-01",
#     "period": "max",
#     "interval": "1d",
#     "engine_kwargs": {
#         "w_iwv": 0.70,
#         "w_agg": 0.30,
#         # choose a default spending rule for the 70/30 view
#         "spending_rule": "growing_payout",
#         "g": 0.05,                 # 70/30 growth rate
#         "initial_spend_rate": 0.01,# 70/30 first-year annual payout as % of initial equity
#         "payout_frequency": "quarterly",
#     },

#     # 👇 IWV-only overrides used *only* for the 100% IWV growing run
#     "iwv100_growth_overrides": {
#         "initial_spend_rate": 0.01,   # e.g., 3% instead of 2%
#         "g": 0.06,                    # e.g., 6% annual growth instead of 5%
#         # you can also override initial_cash/commission/slippage here if desired
#         # "initial_cash": 1_000_000,
#         # "commission_pct": 0.0005,
#         # "slippage_pct": 0.0005,
#     }
# }

UNIVERSITY_ENDOWMENT_SPENDING_STRATEGY_SETTINGS = {
    "strategy_class": StaticMixIWVAGG,     # ✅ has REQUIRED_COLUMNS = {"close_IWV","close_AGG"}
    "w_iwv": 0.85,                         # base mix (85/15); the code also builds 70/30 variants
    "w_agg": 0.15,
    "title": "Endowment (IWV/AGG base 85/15)",
    "label": "Endowment 85/15",
    "interval": "1d",
    "engine_class": EndowmentPayoutBacktestEngine,   # use your payout engine
    "engine_kwargs": {
        # quarter payout = 0.25% of NAV
        "spending_rule": "percent_of_equity",
        "payout_rate_quarterly": 0.0025,

        # These are read by analyze_strategy() to synthesize benchmarks and run variants
        "w_iwv": 0.85,
        "w_agg": 0.15,

        # (optional, only used when you open the “growing” tabs)
        "initial_spend_rate": 0.01,    # 1% starting rate
        "g": 0.00,                     # 0% quarterly growth for the “growing” example unless you change it
        "payout_frequency": "quarterly",
    },

    # Optional “IWV only” overrides for the IWV tabs (not required)
    "iwv100_growth_overrides": {
        "initial_spend_rate": 0.01,
        "g": 0.00
    },
    # You can also limit the period here if you want:
    "start": "2003-10-01",
    # "end": "2025-01-01",
}



AGG_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "AGG",
    "strategy_class": UniversityEndowmentSpendingStrategy,
    "start": "2015-01-01",
    # "period": "max",
    "interval": "1d",
    "type": "equity",
}


BACKTEST_CONFIG = {
    "initial_cash": 100,
    "commission_pct_per_trade": 0.000,
    "slippage_pct": 0.000,
}

AZO_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "AZO",
    "strategy_class": AzoBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
}

NVDA_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "NVDA",
    "strategy_class": NvdaBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
}

AMZN_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "AMZN",
    "strategy_class": AmznBuyAndHoldStrategy,
    "period": "max",  # <-- use start instead of period
    "interval": "1d",
}

STRATEGY_SETTINGS_LIST = [
    # CRYPTO_SENTIMENT_STRATEGY_SETTINGS,
    # FIFTY_WEEK_MA_STRATEGY_SETTINGS,
    # VIX_BTC_STRATEGY_SETTINGS,
    # VIX_SPY_STRATEGY_SETTINGS,
    # SLOW_FAST_MA_STRATEGY_SETTINGS,
    BTC_BUY_AND_HOLD_STRATEGY_SETTINGS,
    SPY_BUY_AND_HOLD_STRATEGY_SETTINGS,
    AZO_BUY_AND_HOLD_STRATEGY_SETTINGS,
    NVDA_BUY_AND_HOLD_STRATEGY_SETTINGS,
    AMZN_BUY_AND_HOLD_STRATEGY_SETTINGS
]  

