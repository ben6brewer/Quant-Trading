# strategies/slow_fast_ma_strategy.py

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from strategies.base_strategy import BaseStrategy


class SlowFastMAStrategy(BaseStrategy):
    def __init__(self, slow_ma=None, fast_ma=None, settings=None, **kwargs):
        """
        Initialize the strategy.

        Priority:
        1. Explicit slow_ma / fast_ma arguments
        2. optimized_params from settings (or from kwargs directly)
        3. Defaults
        """
        # Load optimized_params from settings OR from kwargs directly
        optimized = {}

        if isinstance(settings, dict):
            optimized = settings.get("optimized_params", {})
        elif "optimized_params" in kwargs:
            optimized = kwargs["optimized_params"]

        if slow_ma is None:
            slow_ma = optimized.get("slow_ma", 230)
        if fast_ma is None:
            fast_ma = optimized.get("fast_ma", 100)

        self.slow_ma = int(slow_ma)
        self.fast_ma = int(fast_ma)

        if self.fast_ma >= self.slow_ma:
            raise ValueError(
                f"Invalid parameters: fast_ma ({self.fast_ma}) must be less than slow_ma ({self.slow_ma})"
            )



    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.attrs.update(data.attrs)

        df.attrs['fast_ma_ma'] = self.fast_ma
        df.attrs['slow_ma_ma'] = self.slow_ma

        df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()

        fast_above = (df['fast_ma'] > df['slow_ma']).astype(bool)
        fast_above_shifted = fast_above.shift(1).fillna(False).astype(bool)

        df['raw_signal'] = np.nan
        df.loc[(fast_above) & (~fast_above_shifted), 'raw_signal'] = 1
        df.loc[(~fast_above) & (fast_above_shifted), 'raw_signal'] = -1

        # Only forward-fill after both MAs are valid
        df['signal'] = df['raw_signal']
        df.loc[df['slow_ma'].isna() | df['fast_ma'].isna(), 'signal'] = np.nan
        df['signal'] = df['signal'].shift(1).ffill().fillna(0).astype(int)

        return df
