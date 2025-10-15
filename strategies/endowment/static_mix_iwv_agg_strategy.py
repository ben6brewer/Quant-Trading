# strategies/endowment/static_mix_iwv_agg.py

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class StaticMixIWVAGG(BaseStrategy):
    REQUIRED_COLUMNS = {"close_IWV", "close_AGG"}
    """
    Constant-weight blend of IWV (equity) and AGG (bonds).
    Supports daily ('D') or monthly ('M') rebalancing.

    Daily: reset weights every day (classic constant-mix).
    Monthly: reset weights on the first trading day of each month; weights drift within the month.
    """

    def __init__(
        self,
        w_iwv: float = 0.70,
        w_agg: float = 0.30,
        base: float = 100.0,
        col_iwv: str = "close_IWV",
        col_agg: str = "close_AGG",
        rebalance: str = "M",  # 'D' (daily) or 'M' (monthly)
    ):
        self.w_iwv = float(w_iwv)
        self.w_agg = float(w_agg)
        self.base = float(base)
        self.col_iwv = col_iwv
        self.col_agg = col_agg
        self.rebalance = rebalance.upper()

        if not np.isclose(self.w_iwv + self.w_agg, 1.0):
            raise ValueError(f"w_iwv + w_agg must equal 1.0, got {self.w_iwv + self.w_agg:.6f}")
        if self.w_iwv < 0 or self.w_agg < 0:
            raise ValueError("Weights must be non-negative.")
        if self.rebalance not in {"D", "M"}:
            raise ValueError("rebalance must be 'D' (daily) or 'M' (monthly)")

    def _nav_daily_rebalanced(self, df: pd.DataFrame) -> pd.Series:
        """Classic constant-mix: weights reset every day."""
        r_iwv = df[self.col_iwv].pct_change().fillna(0.0)
        r_agg = df[self.col_agg].pct_change().fillna(0.0)
        mix_ret = self.w_iwv * r_iwv + self.w_agg * r_agg
        return self.base * (1.0 + mix_ret).cumprod()

    def _nav_monthly_rebalanced(self, df: pd.DataFrame) -> pd.Series:
        """
        Reset weights on the first trading day of each month; let them drift within the month.
        Within a month, NAV_t = V0 * (w_iwv * cum_rel_iwv_t + w_agg * cum_rel_agg_t),
        where cum_rel_* are cumulative growth relatives starting at 1.0 at month start.
        """
        # Daily gross returns
        g_iwv = (df[self.col_iwv].pct_change().fillna(0.0) + 1.0)
        g_agg = (df[self.col_agg].pct_change().fillna(0.0) + 1.0)

        periods = df.index.to_period("M")
        nav = pd.Series(index=df.index, dtype=float)
        prev_nav = float(self.base)

        # Iterate month by month for clarity & correctness
        for p in periods.unique():
            mask = (periods == p)
            # Cum relatives within the month, start at 1 on first day
            rel_iwv = g_iwv[mask].cumprod()
            rel_agg = g_agg[mask].cumprod()
            rel_nav = self.w_iwv * rel_iwv + self.w_agg * rel_agg
            nav.loc[mask] = prev_nav * rel_nav
            prev_nav = float(nav.loc[mask].iloc[-1])

        return nav

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        required = {self.col_iwv, self.col_agg}
        if not required.issubset(df.columns):
            raise ValueError(f"Expected {required} in data; got {set(df.columns)}")
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        if self.rebalance == "D":
            nav = self._nav_daily_rebalanced(df)
            rebalance_label = "daily"
        else:  # "M"
            nav = self._nav_monthly_rebalanced(df)
            rebalance_label = "monthly"

        out = pd.DataFrame(index=df.index)
        out["close"] = nav
        out["signal"] = 1.0  # Always long the blended synthetic asset

        # Friendly label (keeps two-number pattern used elsewhere)
        eq_pct = int(round(self.w_iwv * 100))
        fi_pct = 100 - eq_pct
        label = f"{eq_pct}/{fi_pct} IWV/AGG"

        out.attrs["title"] = df.attrs.get("title", label)
        out.attrs["ticker"] = out.attrs["title"]
        out.attrs["w_iwv"] = self.w_iwv
        out.attrs["w_agg"] = self.w_agg
        out.attrs["rebalance"] = rebalance_label

        return out
