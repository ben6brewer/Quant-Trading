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
        df = data.copy()
        df.attrs['title'] = data.attrs.get('title', 'Strategy')
        df.attrs['ticker'] = data.attrs.get('ticker', 'Unknown')


        df['position'] = np.zeros(len(df), dtype=float)
        df['trade'] = np.zeros(len(df), dtype=float)
        df['cash'] = np.full(len(df), float(self.initial_cash))
        df['holdings'] = np.zeros(len(df), dtype=float)
        df['total_equity'] = np.full(len(df), float(self.initial_cash))
        df['returns'] = np.zeros(len(df), dtype=float)

        cash = float(self.initial_cash)
        position = 0.0
        prev_signal = 0.0

        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]

            if not np.isclose(signal, prev_signal):  # Only trade if signal changed
                total_equity_prev = df['total_equity'].iloc[i - 1]
                target_position_value = signal * total_equity_prev

                trade_price = price * (1 + self.slippage_pct) if signal > prev_signal else price * (1 - self.slippage_pct)
                target_units = target_position_value / trade_price
                delta_units = target_units - position
                trade_cost = abs(delta_units) * trade_price + self.commission

                # If buying and not enough cash, scale down
                if delta_units > 0 and trade_cost > cash:
                    delta_units = (cash - self.commission) / trade_price
                    trade_cost = abs(delta_units) * trade_price + self.commission

                cash -= delta_units * trade_price + self.commission
                position += delta_units
                prev_signal = signal
            else:
                delta_units = 0  # No trade

            # Force cash to zero if fully allocated
            if np.isclose(signal, 1.0):
                cash = 0.0

            # Record daily values
            df.at[df.index[i], 'trade'] = delta_units
            df.at[df.index[i], 'position'] = position
            df.at[df.index[i], 'cash'] = cash
            df.at[df.index[i], 'holdings'] = position * price
            df.at[df.index[i], 'total_equity'] = cash + position * price
            df.at[df.index[i], 'returns'] = df['total_equity'].pct_change().iloc[i]

        return df
