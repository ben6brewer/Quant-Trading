# strategies/fifty_week_ma_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy
import numpy as np

class FiftyWeekMAStrategy(BaseStrategy):
    def __init__(self, window_weeks: int = 50, days_per_week: int = 7, threshold_pct: float = 0.03):
        self.window_days = window_weeks * days_per_week  # typically 7 for crypto, 5 for stocks
        self.threshold_pct = threshold_pct  # e.g., 0.05 for 5%

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy.attrs['title'] = data.attrs.get('title', 'Strategy')
        data_copy.attrs['ticker'] = data.attrs.get('ticker', 'Unknown')
        data_copy.attrs.update(data_copy.attrs)

        # Calculate the 50-week moving average
        data_copy['50_week_ma'] = data_copy['close'].rolling(window=self.window_days).mean()

        # Define upper and lower thresholds
        upper_bound = data_copy['50_week_ma'] * (1 + self.threshold_pct)
        lower_bound = data_copy['50_week_ma'] * (1 - self.threshold_pct)

        # Use yesterday's close to generate today's signal
        prev_close = data_copy['close'].shift(1)

        signal = pd.Series(0, index=data_copy.index)
        signal[prev_close > upper_bound] = 1
        signal[prev_close < lower_bound] = -1

        # Shift signal forward so trade occurs the next day
        signal = signal.shift(1).fillna(0).astype(int)

        # Carry forward the position until an exit signal
        data_copy['signal'] = signal.replace(0, np.nan).ffill().fillna(0).astype(int)

        return data_copy
