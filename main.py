# main.py

# from strategies.pe_ratio_strategy import PERatioStrategy
# from backtest.backtest_engine import run_backtest
from config.secrets_config import *
from config.universe_config import *
from alpaca.alpaca_executor import *
from strategies.fifty_week_ma_strategy import *
from utils.data_fetch import *

def main():
    if ENVIRONMENT == "dev":
        pass
    else:
        data_dict = fetch_data_for_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
        print(data_dict)
        strategy = FiftyWeekMAStrategy()

        for ticker, df in data_dict.items():
            print(f"Generating signals for {ticker}")
            signals = strategy.generate_signals(df)
            print(f"Signals for {ticker}:\n", signals)

if __name__ == "__main__":
    main()