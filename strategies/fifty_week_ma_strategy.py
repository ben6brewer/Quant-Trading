# strategies/fifty_week_ma_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy
import numpy as np

class FiftyWeekMAStrategy(BaseStrategy):
    def __init__(self, window_weeks=None, days_per_week=None, threshold_pct=None, settings=None, **kwargs):
        """
        Initialize with:
        1. explicit args
        2. optimized_params in settings (or kwargs)
        3. defaults
        """
        optimized = {}

        if isinstance(settings, dict):
            optimized = settings.get("optimized_params", {})
        elif "optimized_params" in kwargs:
            optimized = kwargs["optimized_params"]

        if window_weeks is None:
            window_weeks = optimized.get("window_weeks", 50)
        if days_per_week is None:
            days_per_week = optimized.get("days_per_week", 7)
        if threshold_pct is None:
            threshold_pct = optimized.get("threshold_pct", 0.03)

        self.window_days = int(window_weeks) * int(days_per_week)
        self.threshold_pct = float(threshold_pct)

        # You can handle extra kwargs like 'period' here if you want to ignore or raise errors

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy.attrs.update(data.attrs)

        data_copy['50_week_ma'] = data_copy['close'].rolling(window=self.window_days).mean()

        upper_bound = data_copy['50_week_ma'] * (1 + self.threshold_pct)
        lower_bound = data_copy['50_week_ma'] * (1 - self.threshold_pct)

        prev_close = data_copy['close'].shift(1)

        signal = pd.Series(0, index=data_copy.index)
        signal[prev_close > upper_bound] = 1
        signal[prev_close < lower_bound] = -1

        signal = signal.shift(1).fillna(0).astype(int)
        data_copy['signal'] = signal.replace(0, np.nan).ffill().fillna(0).astype(int)

        return data_copy

