# backtest/backtest_engine.py

import pandas as pd
import numpy as np
from config.universe_config import BACKTEST_CONFIG


class BacktestEngine:
    def __init__(self):
        self.initial_cash = BACKTEST_CONFIG.get("initial_cash", 100000)
        self.commission_pct = BACKTEST_CONFIG.get("commission_pct_per_trade", 0.0)  # e.g., 0.001 = 0.1%
        self.slippage_pct = BACKTEST_CONFIG.get("slippage_pct", 0.0)
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

            holdings_value = position * price
            total_equity = cash + holdings_value

            # Liquidation condition: equity hits zero or below
            if total_equity <= 0:
                print(f"💥 Liquidated at index {i} ({df.index[i]}): Equity={total_equity:.2f} <= 0")
                df.iloc[i:, df.columns.get_indexer(['cash', 'position', 'holdings', 'total_equity', 'returns', 'trade'])] = 0
                break

            # Proceed only if signal changes
            if not np.isclose(signal, prev_signal):
                total_equity_prev = df['total_equity'].iloc[i - 1]
                target_position_value = signal * total_equity_prev

                est_trade_price = price * (1 + self.slippage_pct) if target_position_value > holdings_value else price * (1 - self.slippage_pct)
                target_units = target_position_value / est_trade_price
                delta_units = target_units - position

                trade_price = price * (1 + self.slippage_pct) if delta_units > 0 else price * (1 - self.slippage_pct)
                trade_value = abs(delta_units) * trade_price
                commission_cost = trade_value * self.commission_pct

                if delta_units > 0:
                    max_trade_value = cash / (1 + self.commission_pct)
                    if trade_value > max_trade_value:
                        trade_value = max_trade_value
                        delta_units = trade_value / trade_price
                        commission_cost = trade_value * self.commission_pct

                if delta_units > 0 and (trade_value + commission_cost) > cash:
                    print(f"⚠️ Insufficient cash for trade at index {i} ({df.index[i]}): Needed={trade_value + commission_cost:.2f}, Available={cash:.2f}. Trade skipped.")
                    delta_units = 0
                    trade_value = 0
                    commission_cost = 0

                if delta_units > 0:
                    cash -= (trade_value + commission_cost)
                elif delta_units < 0:
                    cash += trade_value
                    cash -= commission_cost

                position += delta_units
                prev_signal = signal
            else:
                delta_units = 0

            holdings_value = position * price
            total_equity = cash + holdings_value

            df.at[df.index[i], 'trade'] = delta_units
            df.at[df.index[i], 'position'] = position
            df.at[df.index[i], 'cash'] = cash
            df.at[df.index[i], 'holdings'] = holdings_value
            df.at[df.index[i], 'total_equity'] = total_equity
            df.at[df.index[i], 'returns'] = df['total_equity'].pct_change().iloc[i]

        return df
