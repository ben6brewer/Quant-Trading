# strategies/fifty_week_ma_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class FiftyWeekMAStrategy(BaseStrategy):
    def __init__(self, window_weeks: int = 50, days_per_week: int = 7):
        self.window_days = window_weeks * days_per_week  # use 7 for crypto

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Modifies and returns input DataFrame with added '50_week_ma' and 'signal' columns.
        Signal:
            1 = long, -1 = short, 0 = neutral
        """
        data = data.copy()

        data['50_week_ma'] = data['close'].rolling(window=self.window_days).mean()

        data['signal'] = 0
        data.loc[data['close'] > data['50_week_ma'], 'signal'] = 1
        data.loc[data['close'] < data['50_week_ma'], 'signal'] = -1

        data['signal'] = data['signal'].shift(1).fillna(0).astype(int)
        return data