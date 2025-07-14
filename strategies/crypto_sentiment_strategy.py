import pandas as pd
from strategies.base_strategy import BaseStrategy
import numpy as np

class CryptoSentimentStrategy(BaseStrategy):
    def __init__(self, 
                 fear_threshold: int = None, 
                 greed_threshold: int = None,
                 fear_days_required: int = None,
                 greed_days_required: int = None,
                 settings: dict = None,
                 **kwargs):
        if settings:
            params = settings.get("optimized_params", {})
            fear_threshold = params.get("fear_threshold", 20)
            greed_threshold = params.get("greed_threshold", 80)
            fear_days_required = params.get("fear_days_required", 10)
            greed_days_required = params.get("greed_days_required", 10)

        self.fear_threshold = fear_threshold if fear_threshold is not None else 20
        self.greed_threshold = greed_threshold if greed_threshold is not None else 80
        self.fear_days_required = fear_days_required if fear_days_required is not None else 10
        self.greed_days_required = greed_days_required if greed_days_required is not None else 10


    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        
        data_copy.attrs['title'] = data.attrs.get('title', 'Strategy')
        data_copy.attrs['ticker'] = data.attrs.get('ticker', 'Unknown')

        fng_series = data_copy['F&G']

        # Boolean masks
        fear_condition = fng_series <= self.fear_threshold
        greed_condition = fng_series >= self.greed_threshold

        # Apply rolling logic separately
        fear_streak = fear_condition.rolling(window=self.fear_days_required).apply(lambda x: all(x), raw=True).fillna(0)
        greed_streak = greed_condition.rolling(window=self.greed_days_required).apply(lambda x: all(x), raw=True).fillna(0)

        # Signal series: 1 = long, -1 = exit/cash, 0 = hold
        signal_raw = pd.Series(0, index=data_copy.index)
        signal_raw[fear_streak == 1] = 1
        signal_raw[greed_streak == 1] = -1

        # Shift signal forward so trade executes next day
        signal_raw = signal_raw.shift(1).fillna(0)

        # Convert signals to persistent position state
        position = []
        current_position = 0

        for signal in signal_raw:
            if signal == 1:
                current_position = 1
            elif signal == -1:
                current_position = 0
            position.append(current_position)

        data_copy['signal'] = position
        return data_copy
