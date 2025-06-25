# backtest/backtest_engine.py

import pandas as pd
import numpy as np
from config.universe_config import BACKTEST_CONFIG


class BacktestEngine:
    def __init__(self):
        # Load config params
        self.initial_cash = BACKTEST_CONFIG.get("initial_cash", 100000)
        self.commission = BACKTEST_CONFIG.get("commission_per_trade", 1.0)
        self.position_size_pct = BACKTEST_CONFIG.get("position_size_pct", 100.0) / 100.0
        self.slippage_pct = BACKTEST_CONFIG.get("slippage_pct", 0.0)
        self.rebalance_on = BACKTEST_CONFIG.get("rebalance_on", "signal_change")
        self.trade_on_close = BACKTEST_CONFIG.get("trade_on_close", True)

    def run_backtest(self, data: pd.DataFrame):
        """
        Backtest over data with 'close' and 'signal' columns.
        Signal: 1=long, -1=short, 0=flat.
        Uses position_size_pct of portfolio.
        """
        df = data.copy()

        df['position'] = np.zeros(len(df), dtype=float)
        df['trade'] = np.zeros(len(df), dtype=float)
        df['cash'] = np.full(len(df), float(self.initial_cash))
        df['holdings'] = np.zeros(len(df), dtype=float)
        df['total_equity'] = np.full(len(df), float(self.initial_cash))
        df['returns'] = np.zeros(len(df), dtype=float)

        cash = float(self.initial_cash)
        position = 0.0

        for i in range(1, len(df)):
            signal = df['signal'].iloc[i]
            price = df['close'].iloc[i]
            prev_signal = df['signal'].iloc[i - 1]
            signal_changed = (signal != prev_signal)

            # Skip trading unless signal changes or rebalancing strategy applies
            if self.rebalance_on == "signal_change" and not signal_changed:
                df.at[df.index[i], 'position'] = float(position)
                df.at[df.index[i], 'cash'] = float(cash)
                df.at[df.index[i], 'holdings'] = float(position * price)
                df.at[df.index[i], 'total_equity'] = float(cash + position * price)
                df.at[df.index[i], 'returns'] = df['total_equity'].pct_change().iloc[i]
                continue

            # Close existing position
            if position != 0 and signal != prev_signal:
                sell_price = price * (1 - self.slippage_pct)
                proceeds = position * sell_price - self.commission
                cash += proceeds
                df.at[df.index[i], 'trade'] = -float(position)
                position = 0.0

            units_to_buy = None  # Default to None to avoid unbound reference

            # Open new long position
            if signal == 1:
                trade_cash = cash * self.position_size_pct
                buy_price = price * (1 + self.slippage_pct)
                units_to_buy = (trade_cash - self.commission) / buy_price

                if units_to_buy > 0:
                    cost = units_to_buy * buy_price + self.commission
                    cash -= cost
                    position = units_to_buy
                    df.at[df.index[i], 'trade'] = float(units_to_buy)

            elif signal == -1:
                trade_cash = cash * self.position_size_pct
                sell_price = price * (1 - self.slippage_pct)
                units_to_short = (trade_cash - self.commission) / sell_price

                if units_to_short > 0:
                    proceeds = units_to_short * sell_price - self.commission
                    cash += proceeds
                    position = -units_to_short
                    df.at[df.index[i], 'trade'] = -float(units_to_short)
                else:
                    position = 0.0
                    df.at[df.index[i], 'trade'] = 0.0

            holdings = position * price
            total_equity = cash + holdings

            df.at[df.index[i], 'position'] = float(position)
            df.at[df.index[i], 'cash'] = float(cash)
            df.at[df.index[i], 'holdings'] = float(holdings)
            df.at[df.index[i], 'total_equity'] = float(total_equity)
            df.at[df.index[i], 'returns'] = df['total_equity'].pct_change().iloc[i]

        return df
