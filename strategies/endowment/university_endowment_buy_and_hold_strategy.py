# strategies/endowment/university_endowment_spending_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class UniversityEndowmentSpendingStrategy(BaseStrategy):
    def __init__(self, w_iwv: float = 0.70, w_agg: float = 0.30, **_):
        self.w_iwv = w_iwv
        self.w_agg = w_agg

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Ensure required columns present
        required = {'close_IWV', 'close_AGG'}
        if not required.issubset(df.columns):
            raise ValueError(f"Expected {required} in data; got {set(df.columns)}")

        # Put weights into attrs for reference
        df.attrs['title'] = data.attrs.get('title', 'Endowment Spending (70/30)')
        df.attrs['ticker'] = data.attrs.get('ticker', 'IWV/AGG')
        df.attrs['w_iwv'] = self.w_iwv
        df.attrs['w_agg'] = self.w_agg
        # No 'signal' needed for the new engine
        return df
