import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class StaticMixIWV_VXUS_AGG(BaseStrategy):
    """
    Constant-weight blend of IWV / VXUS / AGG with configurable rebalance frequency.
      rebalance = "D" (daily), "M" (monthly), "Q" (quarterly), "A" (annual)
    We simulate positions and reset weights to targets at each rebalance date.
    """
    REQUIRED_COLUMNS = {"close_IWV", "close_VXUS", "close_AGG"}

    def __init__(
        self,
        w_iwv: float = 0.49,
        w_vxus: float = 0.21,
        w_agg: float = 0.30,
        base: float = 100.0,
        col_iwv: str = "close_IWV",
        col_vxus: str = "close_VXUS",
        col_agg: str = "close_AGG",
        rebalance: str = "M",  # ← change to "D" if you want daily as default
    ):
        self.w_iwv  = float(w_iwv)
        self.w_vxus = float(w_vxus)
        self.w_agg  = float(w_agg)
        self.base   = float(base)
        self.col_iwv  = col_iwv
        self.col_vxus = col_vxus
        self.col_agg  = col_agg
        self.rebalance = rebalance.upper()

        s = self.w_iwv + self.w_vxus + self.w_agg
        if not np.isclose(s, 1.0):
            raise ValueError(f"weights must sum to 1.0 (got {s:.6f})")
        if self.w_iwv < 0 or self.w_vxus < 0 or self.w_agg < 0:
            raise ValueError("All weights must be non-negative.")
        if self.rebalance not in {"D", "M", "Q", "A"}:
            raise ValueError("rebalance must be one of {'D','M','Q','A'}")

    def _is_rebalance_date(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Return a boolean Series marking rebalance dates on this index."""
        if len(idx) == 0:
            return pd.Series([], dtype=bool, index=idx)

        if self.rebalance == "D":
            # rebalance every day
            return pd.Series(True, index=idx)

        # Choose period granularity
        if self.rebalance == "M":
            periods = idx.to_period("M")
        elif self.rebalance == "Q":
            periods = idx.to_period("Q")
        elif self.rebalance == "A":
            periods = idx.to_period("A")
        else:
            raise ValueError("rebalance must be one of {'D','M','Q','A'}")

        # Mark first row True, then True whenever the period changes
        flags = np.ones(len(idx), dtype=bool)             # first day is always a rebalance
        flags[1:] = periods[1:] != periods[:-1]           # period boundary
        return pd.Series(flags, index=idx)


    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        required = {self.col_iwv, self.col_vxus, self.col_agg}
        if not required.issubset(df.columns):
            raise ValueError(f"Expected {required} in data; got {set(df.columns)}")

        # Daily returns
        r_iwv  = df[self.col_iwv].pct_change().fillna(0.0)
        r_vxus = df[self.col_vxus].pct_change().fillna(0.0)
        r_agg  = df[self.col_agg].pct_change().fillna(0.0)

        idx = df.index
        is_reb = self._is_rebalance_date(idx)

        # Position values (simulate portfolio)
        w_target = np.array([self.w_iwv, self.w_vxus, self.w_agg], dtype=float)
        vals = np.zeros((len(idx), 3), dtype=float)
        nav = np.zeros(len(idx), dtype=float)

        if len(idx) == 0:
            return pd.DataFrame(columns=["close", "signal"])

        # Initialize on first day
        nav[0] = self.base
        vals[0, :] = w_target * nav[0]

        # Iterate days
        for i in range(1, len(idx)):
            # apply asset returns to yesterday's values
            vals[i, 0] = vals[i-1, 0] * (1.0 + r_iwv.iloc[i])
            vals[i, 1] = vals[i-1, 1] * (1.0 + r_vxus.iloc[i])
            vals[i, 2] = vals[i-1, 2] * (1.0 + r_agg.iloc[i])

            nav[i] = vals[i, :].sum()

            # if it's a rebalance date, reset to targets
            if is_reb.iloc[i]:
                vals[i, :] = w_target * nav[i]

        out = pd.DataFrame(index=idx)
        out["close"]  = nav
        out["signal"] = 1.0  # always long the synthetic blend

        # Labels
        eq_total = self.w_iwv + self.w_vxus
        eq_pct  = int(round(eq_total * 100))
        bd_pct  = 100 - eq_pct
        label = f"{eq_pct}/{bd_pct} EQ/Bonds"
        title = (
            f"{int(round(self.w_iwv*100))}/"
            f"{int(round(self.w_vxus*100))}/"
            f"{int(round(self.w_agg*100))} IWV/VXUS/AGG"
            f" (Reb={self.rebalance})"
        )

        out.attrs["title"]  = title
        out.attrs["ticker"] = title
        out.attrs["label"]  = label
        out.attrs["eq_pct"] = eq_pct
        out.attrs["rebalance"] = self.rebalance

        return out
