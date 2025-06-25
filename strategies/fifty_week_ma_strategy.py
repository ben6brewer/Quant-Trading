# strategies/fifty_week_ma_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy
import numpy as np

class FiftyWeekMAStrategy(BaseStrategy):
    def __init__(self, window_weeks: int = 50, days_per_week: int = 7, threshold_pct: float = 0.05):
        self.window_days = window_weeks * days_per_week  # typically 7 for crypto, 5 for stocks
        self.threshold_pct = threshold_pct  # e.g., 0.05 for 5%

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['50_week_ma'] = data['close'].rolling(window=self.window_days).mean()

        upper_bound = data['50_week_ma'] * (1 + self.threshold_pct)
        lower_bound = data['50_week_ma'] * (1 - self.threshold_pct)

        # Raw signals based on threshold
        data['signal'] = 0
        data.loc[data['close'] > upper_bound, 'signal'] = 1
        data.loc[data['close'] < lower_bound, 'signal'] = -1

        # Shift to avoid lookahead bias
        data['signal'] = data['signal'].shift(1).fillna(0).astype(int)

        # **Forward fill the signal to maintain position until a change**
        data['signal'] = data['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)

        return data
