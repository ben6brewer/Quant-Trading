# strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given input data (price/fundamentals),
        return DataFrame of signals: 1=long, -1=short, 0=flat
        """
        pass
