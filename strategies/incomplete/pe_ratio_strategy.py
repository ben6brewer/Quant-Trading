# strategies/pe_ratio_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class PERatioStrategy(BaseStrategy):
    def __init__(self, quantile=0.2, min_pe=2.0, max_pe=50.0):
        self.quantile = quantile
        self.min_pe = min_pe
        self.max_pe = max_pe

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Example: data should have MultiIndex (date, ticker) and a 'pe_ratio' column
        signals = []

        for date, group in data.groupby(level=0):
            df = group.copy()
            df = df.dropna(subset=['pe_ratio'])
            # Filter by PE range
            df = df[(df['pe_ratio'] >= self.min_pe) & (df['pe_ratio'] <= self.max_pe)]
            n = int(len(df) * self.quantile)
            df['signal'] = 0
            df = df.sort_values('pe_ratio')
            df.iloc[:n, df.columns.get_loc('signal')] = 1    # Long bottom quantile
            df.iloc[-n:, df.columns.get_loc('signal')] = -1  # Short top quantile
            signals.append(df[['signal']])

        return pd.concat(signals)
