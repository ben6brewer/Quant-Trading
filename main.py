# main.py

from config.secrets_config import *
from config.universe_config import *
from alpaca.alpaca_executor import *
from strategies.fifty_week_ma_strategy import *
from utils.data_fetch import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import *
from alpaca.alpaca_executor import *
import pandas as pd
pd.set_option('display.float_format', '{:,.2f}'.format)


def main():
    # executor = AlpacaExecutor()
    # executor.place_order("AAPL", 1, "buy")
    # executor.liquidate_all_positions()
    data_dict = fetch_data_for_strategy(FIFTY_WEEK_MA_STRATEGY_SETTINGS)
    strategy = FiftyWeekMAStrategy()

    backtester = BacktestEngine()

    for ticker, df in data_dict.items():
        signal_df = strategy.generate_signals(df)
        # plot_signals(df)
        results_df = backtester.run_backtest(signal_df)
        # print(results_df.tail())
        # plot_equity_curve(results_df)
        plot_equity_vs_benchmark(results_df)


if __name__ == "__main__":
    main()