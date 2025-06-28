# strategies/crypto_sentiment_strategy.py

from utils.cmc_data_fetch import *
import pandas as pd
from strategies.base_strategy import BaseStrategy
import numpy as np

class CryptoSentimentStrategy(BaseStrategy):
    def __init__(self, fear_threshold: int = 20, greed_threshold: int = 80):
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy.title = data.title
        data_copy.ticker = data.ticker

        # Initialize signal series with default (cash position)
        signal_raw = pd.Series(0, index=data_copy.index)

        # Buy when F&G <= fear_threshold
        signal_raw[data_copy['F&G'] <= self.fear_threshold] = 1

        # Go to cash when F&G >= greed_threshold
        signal_raw[data_copy['F&G'] >= self.greed_threshold] = -1  # temporary sell signal marker

        # Shift so the action takes place the *next day*
        signal_raw = signal_raw.shift(1).fillna(0)

        # Convert raw signals to positions: 1 (long), 0 (cash)
        position = []
        current_position = 0

        for signal in signal_raw:
            if signal == 1:
                current_position = 1  # Enter long
            elif signal == -1:
                current_position = 0  # Exit to cash
            # else: maintain current_position
            position.append(current_position)

        data_copy['signal'] = position
        return data_copy
