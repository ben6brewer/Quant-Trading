# strategies.tsn_buy_and_hold_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class NvdaBuyAndHoldStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.attrs['title'] = data.attrs.get('title', 'NVDA Buy and Hold Strategy')
        df.attrs['ticker'] = data.attrs.get('ticker', 'NVDA')

        # Signal = 1 for entire period (long), no exit signals
        df['signal'] = 1

        return df
