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
        data_copy.title = data.title
        data_copy.ticker = data.ticker

        data_copy['50_week_ma'] = data_copy['close'].rolling(window=self.window_days).mean()

        upper_bound = data_copy['50_week_ma'] * (1 + self.threshold_pct)
        lower_bound = data_copy['50_week_ma'] * (1 - self.threshold_pct)

        data_copy['signal'] = 0
        data_copy.loc[data_copy['close'] > upper_bound, 'signal'] = 1
        data_copy.loc[data_copy['close'] < lower_bound, 'signal'] = -1

        data_copy['signal'] = data_copy['signal'].shift(1).fillna(0).astype(int)
        data_copy['signal'] = data_copy['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)

        return data_copy
