#strategy/vix_btc_strategy.py

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class VixBtcStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        self.vix_threshold = kwargs.get('vix_threshold', 45.0)
        self.take_profit_pct = kwargs.get('take_profit_pct', 1.0)
        self.partial_exit_pct = kwargs.get('partial_exit_pct', 0.1)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        title = data.attrs.get("title", "VIX BTC Strategy")
        ticker = data.attrs.get("ticker", "Unknown")

        df['vix_prev'] = df['vix'].shift(1)
        df['close_prev'] = df['close'].shift(1)

        signal = []
        entry_price = None
        current_signal = 1 - self.partial_exit_pct
        took_partial_profit = False

        for idx, row in df.iterrows():
            vix = row['vix_prev']
            price = row['close_prev']

            if pd.isna(vix) or pd.isna(price):
                signal.append(current_signal)
                continue

            if vix >= self.vix_threshold:
                if current_signal < 1.0:
                    current_signal = 1.0
                    entry_price = price
                    took_partial_profit = False

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
