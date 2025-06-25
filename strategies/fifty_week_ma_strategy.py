# strategies/fifty_week_ma_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class FiftyWeekMAStrategy(BaseStrategy):
    def __init__(self, window_weeks: int = 50, days_per_week: int = 7):
        self.window_days = window_weeks * days_per_week  # use 7 for crypto

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Expects data with a 'close' column and a DateTime index.

        Returns a DataFrame with a 'signal' column:
        1 = long, -1 = short, 0 = flat/neutral
        """
        signals = pd.DataFrame(index=data.index)
        signals['50_week_ma'] = data['close'].rolling(window=self.window_days).mean()

        signals['signal'] = 0
        signals.loc[data['close'] > signals['50_week_ma'], 'signal'] = 1
        signals.loc[data['close'] < signals['50_week_ma'], 'signal'] = -1

        signals['signal'] = signals['signal'].shift(1)

        signals['signal'] = signals['signal'].fillna(0).astype(int)
        return signals


