# strategies/slow_fast_ma_strategy.py

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from strategies.base_strategy import BaseStrategy
from config.universe_config import SLOW_FAST_MA_STRATEGY_SETTINGS


class SlowFastMAStrategy(BaseStrategy):
    def __init__(self, slow_ma: int = SLOW_FAST_MA_STRATEGY_SETTINGS['optimized_params']['slow_ma'], fast_ma: int = SLOW_FAST_MA_STRATEGY_SETTINGS['optimized_params']['fast_ma']):
        """
        Initialize the strategy with slow and fast moving average periods.

        Args:
            slow_ma (int): Period for the slow moving average (e.g. 230).
            fast_ma (int): Period for the fast moving average (e.g. 100).

        Raises:
            ValueError: If fast_ma is greater than or equal to slow_ma.
        """
        self.slow_ma = int(slow_ma)
        self.fast_ma = int(fast_ma)    
        if self.fast_ma >= self.slow_ma:
            raise ValueError(f"Invalid parameters: fast_ma ({self.fast_ma}) must be less than slow_ma ({self.slow_ma})")


    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.attrs.update(data.attrs)

        # Store MA ma sizes as metadata for plotting
        df.attrs['fast_ma_ma'] = self.fast_ma
        df.attrs['slow_ma_ma'] = self.slow_ma

        df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()

        df['signal'] = np.nan

        fast_above = (df['fast_ma'] > df['slow_ma']).astype(bool)
        fast_above_shifted = fast_above.shift(1).fillna(False).astype(bool)

        # Detect crossovers
        df.loc[(fast_above) & (~fast_above_shifted), 'signal'] = 1   # Golden cross (go long)
        df.loc[(~fast_above) & (fast_above_shifted), 'signal'] = -1  # Death cross (go short)

        # Hold signal forward
        df['signal'] = df['signal'].ffill().fillna(0).astype(int)

        return df
