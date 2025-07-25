# config/universe_config.py

from strategies.moving_averages.slow_fast_ma_strategy import SlowFastMAStrategy
from strategies.moving_averages.fifty_week_ma_strategy import FiftyWeekMAStrategy
from strategies.sentiment.crypto_sentiment_strategy import CryptoSentimentStrategy
from strategies.vix.vix_btc_strategy import VixBtcStrategy
from strategies.vix.vix_spy_strategy import VixSpyStrategy
from strategies.buy_and_hold.btc_buy_and_hold_strategy import BtcBuyAndHoldStrategy
from strategies.buy_and_hold.spy_buy_and_hold_strategy import SpyBuyAndHoldStrategy

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

SPY_BUY_AND_HOLD_STRATEGY_SETTINGS = {
    "title": "Buy and Hold",
    "ticker": "SPY",
    "strategy_class": SpyBuyAndHoldStrategy,
    "period": "max",
    "interval": "1d",
}

BACKTEST_CONFIG = {
    "initial_cash": 100000,
    "commission_pct_per_trade": 0.002,
    "slippage_pct": 0.001,
}

STRATEGY_SETTINGS_LIST = [
    CRYPTO_SENTIMENT_STRATEGY_SETTINGS,
    FIFTY_WEEK_MA_STRATEGY_SETTINGS,
    VIX_BTC_STRATEGY_SETTINGS,
    VIX_SPY_STRATEGY_SETTINGS,
    SLOW_FAST_MA_STRATEGY_SETTINGS,
    BTC_BUY_AND_HOLD_STRATEGY_SETTINGS,
    SPY_BUY_AND_HOLD_STRATEGY_SETTINGS,
]  

