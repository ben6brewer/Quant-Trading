# strategy/vix_btc_strategy.py

import pandas as pd
from strategies.base_strategy import BaseStrategy

class VixBtcStrategy(BaseStrategy):
    def __init__(self, vix_threshold: float = 45.0, take_profit_pct: float = 1.0, partial_exit_pct: float = 0.1):
        self.vix_threshold = vix_threshold
        self.take_profit_pct = take_profit_pct
        self.partial_exit_pct = partial_exit_pct

    def generate_signals(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        If data is not provided, fetch BTC historical data including VIX data merged.
        Then generate signals based on VIX threshold and BTC price.
        """
        if data is None:
            df = fetch_btc_historical()
        else:
            df = data.copy()

        df.title = "BTC with VIX"
        df.ticker = "BTC-USD"

        signal = []
        entry_price = None
        current_signal = 1 - self.partial_exit_pct  # Start partially invested (e.g., 0.9)
        took_partial_profit = False

        for idx, row in df.iterrows():
            vix = row.get('vix', None)
            price = row['close']

            if vix is None:
                raise ValueError("Data must include 'vix' column.")

            # VIX spike: go fully invested if not already
            if vix >= self.vix_threshold:
                if current_signal < 1.0:
                    current_signal = 1.0
                    entry_price = price
                    took_partial_profit = False

            # Take partial profit after gain
            elif current_signal == 1.0 and entry_price is not None and not took_partial_profit:
                gain = (price - entry_price) / entry_price
                if gain >= self.take_profit_pct:
                    current_signal = 1 - self.partial_exit_pct
                    took_partial_profit = True

            signal.append(current_signal)

        df['signal'] = signal
        return df
