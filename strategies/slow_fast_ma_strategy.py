# strategies/slow_fast_ma_strategy.py

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from strategies.base_strategy import BaseStrategy

class SlowFastMAStrategy(BaseStrategy):
    def __init__(self, slow_window: int = 230, fast_window: int = 100):
        """
        slow_window: int, period for slow moving average (e.g. 50 days)
        fast_window: int, period for fast moving average (e.g. 10 days)
        """
        if fast_window >= slow_window:
            raise ValueError("fast_window should be smaller than slow_window")
        self.slow_window = slow_window
        self.fast_window = fast_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.attrs.update(data.attrs)

        # Store MA window sizes as metadata for plotting
        df.attrs['fast_ma_window'] = self.fast_window
        df.attrs['slow_ma_window'] = self.slow_window

        df['slow_ma'] = df['close'].rolling(window=self.slow_window).mean()
        df['fast_ma'] = df['close'].rolling(window=self.fast_window).mean()

        df['signal'] = np.nan

        fast_above = (df['fast_ma'] > df['slow_ma']).astype(bool)
        fast_above_shifted = fast_above.shift(1).fillna(False).astype(bool)

        # Detect crossovers
        df.loc[(fast_above) & (~fast_above_shifted), 'signal'] = 1   # Golden cross (go long)
        df.loc[(~fast_above) & (fast_above_shifted), 'signal'] = -1  # Death cross (go short)

        # Hold signal forward
        df['signal'] = df['signal'].ffill().fillna(0).astype(int)

        return df
