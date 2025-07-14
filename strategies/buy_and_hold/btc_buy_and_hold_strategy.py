# strategies.btc_buy_and_hold_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class BtcBuyAndHoldStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # In case the base strategy needs it

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.attrs['title'] = data.attrs.get('title', 'BTC Buy and Hold Strategy')
        df.attrs['ticker'] = data.attrs.get('ticker', 'BTC')

        # Signal = 1 for entire period (long), no exit signals
        df['signal'] = 1

        return df
