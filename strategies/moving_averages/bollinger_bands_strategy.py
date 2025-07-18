# strategies/moving_averages/bollinger_bands.py

import pandas as pd
from strategies.base_strategy import BaseStrategy
import numpy as np
from utils.pretty_print_df import *

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, sma_window=20, k=2, compression_pct=0.03, **kwargs):
        """
        Args:
            sma_window (int): Rolling window size.
            k (float): Std deviation multiplier for bands.
            compression_pct (float): Max band width as % of close price to trigger long.
        """
        super().__init__(**kwargs)
        self.sma_window = sma_window
        self.k = k
        self.compression_pct = compression_pct

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        data['sma'] = data['close'].rolling(self.sma_window).mean()
        data['std'] = data['close'].rolling(self.sma_window).std()
        data['upper_band'] = data['sma'] + self.k * data['std']
        data['lower_band'] = data['sma'] - self.k * data['std']
        data['band_width'] = data['upper_band'] - data['lower_band']

        data['signal'] = 0
        data.loc[
            (data['band_width'] / data['close']) <= self.compression_pct,
            'signal'
        ] = -1
        return data
