# main.py

# from strategies.pe_ratio_strategy import PERatioStrategy
# from backtest.backtest_engine import run_backtest
# from config.universe_config import PE_STRATEGY_SETTINGS, SP500_TICKERS
from config.secrets_config import ENVIRONMENT
from alpaca.alpaca_executor import AlpacaExecutor

def main():
    if ENVIRONMENT == "dev":
        pass
    else:
        executor = AlpacaExecutor()
        # executor.place_order("BTC", 1, "buy")
        # executor.liquidate_all_positions()
        # executor.cancel_all_orders()
        # print(executor.api.list_orders(status='all', limit=100, nested=True))

if __name__ == "__main__":
    main()
