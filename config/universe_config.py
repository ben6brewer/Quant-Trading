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
    "tickers": ["BTC-USD"],
    "period": "max",
    "interval": "1d",
}


BACKTEST_CONFIG = {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_cash": 100000,
    "commission_per_trade": 1.0
}
