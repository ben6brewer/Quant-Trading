# config/universe_config.py

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
    "period": "max",
    "interval": "1d",
}

CRYPTO_SENTIMENT_STRATEGY_SETTINGS = {
    "title": "Crypto Sentiment Strategy",
    "ticker": "BTC-USD",
    "period": "max",
    "interval": "1d",
}

VIX_SPY_STRATEGY_SETTINGS = {
    "title": "VIX Strategy",
    "ticker": "SPY",
    "period": "max",
    "interval": "1d",
    "optimization_params": {
        "vix_threshold": (48, 49, 1),
        "take_profit_pct": (0.09, 0.1, 0.01),
        "partial_exit_pct": (0.09, 0.1, 0.01)
    }
}

VIX_BTC_STRATEGY_SETTINGS = {
    "title": "VIX Strategy",
    "ticker": "BTC-USD",
    "period": "max",
    "interval": "1d",
}


BACKTEST_CONFIG = {
    "initial_cash": 100000,
    "commission_per_trade": 1.0,           # Flat fee per trade
    "position_size_pct": 100.0,            # Fraction of portfolio to allocate per signal
    "slippage_pct": 0.001,                 # 0.1% slippage
    "rebalance_on": "signal_change",       # Could also be 'weekly', 'monthly', etc.
    "trade_on_close": True                 # Whether to trade on the closing price of the signal day
}
