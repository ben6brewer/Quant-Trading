#strategy/vix_spy_strategy.py

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class VixSpyStrategy(BaseStrategy):
    def __init__(self, vix_threshold, take_profit_pct, partial_exit_pct):
        self.vix_threshold = vix_threshold
        self.take_profit_pct = take_profit_pct
        self.partial_exit_pct = partial_exit_pct

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Use .attrs for metadata
        title = data.attrs.get("title", "VIX Strategy")
        ticker = data.attrs.get("ticker", "Unknown")

        signal = []
        entry_price = None
        current_signal = 1 - self.partial_exit_pct
        took_partial_profit = False

        for idx, row in df.iterrows():
            vix = row['vix']
            price = row['close']

            # Check for VIX spike and currently not 100% invested
            if vix >= self.vix_threshold:
                if current_signal < 1.0:
                    current_signal = 1.0
                    entry_price = price
                    took_partial_profit = False

            # Check for take-profit only if fully invested and not already took partial profit
            elif current_signal == 1.0 and entry_price is not None and not took_partial_profit:
                gain = (price - entry_price) / entry_price
                if gain >= self.take_profit_pct:
                    current_signal = 1 - self.partial_exit_pct
                    took_partial_profit = True

            signal.append(current_signal)

        df['signal'] = signal
        df.attrs['title'] = title
        df.attrs['ticker'] = ticker
        return df
