# run.py

from strategies.pe_ratio_strategy import PERatioStrategy
from backtest.backtest_engine import run_backtest
from config.universe_config import PE_STRATEGY_SETTINGS, SP500_TICKERS
from config.secrets_config import ENVIRONMENT
from execution.alpaca_executor import AlpacaExecutor

def main():
    strategy = PERatioStrategy(**PE_STRATEGY_SETTINGS)

    if ENVIRONMENT == "dev":
        run_backtest(strategy, SP500_TICKERS)
    else:
        executor = AlpacaExecutor()
        executor.run_live(strategy, SP500_TICKERS)

if __name__ == "__main__":
    main()
