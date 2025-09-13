# backtest/endowment_payout_engine.py

import numpy as np
import pandas as pd

# Local defaults (override via engine_kwargs in universe_config)
DEFAULT_INITIAL_CASH = 1_000_000.0
DEFAULT_COMMISSION_PCT = 0.0
DEFAULT_SLIPPAGE_PCT = 0.0


class EndowmentPayoutBacktestEngine:
    """
    Multi-asset engine for endowment spending:
      - Holds IWV & AGG at fixed starting weights (no rebalance)
      - Spending rules:
          1) percent_of_equity: each payout is a fixed percent of current total equity (e.g., 0.25% per quarter).
          2) growing_payout: first-year annual payout = initial_spend_rate * initial_equity.
             If payout_frequency='quarterly', the FIRST quarter pays (initial_spend_rate/4)*initial_equity,
             and each subsequent quarter grows by a per-quarter growth rate derived from g:
               - per_period_growth='compounded' (default): r_q = (1+g)^(1/4) - 1
               - per_period_growth='g_over_4':       r_q = g/4
             If payout_frequency='annual', it pays the whole annual budget once per year, growing at g annually.

      - Payout funded by selling IWV and AGG:
          * 50/50 when both positions have units
          * 100% from the asset that exists if the other has zero units (fix for IWV-only case)
      - Records detailed payout ledger.

    Expected input columns: close_IWV, close_AGG (aligned on DatetimeIndex).
    """

    def __init__(
        self,
        w_iwv: float = 0.70,
        w_agg: float = 0.30,
        # --- Rule 1: percent_of_equity ---
        payout_rate_quarterly: float = 0.0025,  # 0.25% per quarter
        # --- Rule 2: growing_payout ---
        spending_rule: str | None = None,       # "percent_of_equity" or "growing_payout" (auto if None)
        g: float | None = None,                 # annual growth rate (e.g., 0.04 for 4%)
        initial_spend_rate: float = 0.01,       # 1% of initial equity (annual budget for year 1)
        payout_frequency: str = "quarterly",    # "annual" or "quarterly"
        per_period_growth: str = "compounded",  # "compounded" (default) or "g_over_4"
        # --- Trading frictions / capital ---
        initial_cash: float | None = None,
        commission_pct: float | None = None,
        slippage_pct: float | None = None,
    ):
        # No import from universe_config to avoid circular imports — use local defaults:
        self.initial_cash = float(DEFAULT_INITIAL_CASH if initial_cash is None else initial_cash)
        self.commission_pct = float(DEFAULT_COMMISSION_PCT if commission_pct is None else commission_pct)
        self.slippage_pct = float(DEFAULT_SLIPPAGE_PCT if slippage_pct is None else slippage_pct)

        # Normalize weights
        s = w_iwv + w_agg
        self.w_iwv = float(w_iwv) / s
        self.w_agg = float(w_agg) / s

        # Spending rule selection
        if spending_rule is None:
            spending_rule = "growing_payout" if (g is not None) else "percent_of_equity"
        self.spending_rule = spending_rule

        # Params for percent_of_equity
        self.payout_rate_q = float(payout_rate_quarterly)

        # Params for growing_payout
        self.g = 0.0 if g is None else float(g)
        self.initial_spend_rate = float(initial_spend_rate)
        if payout_frequency not in {"annual", "quarterly"}:
            raise ValueError("payout_frequency must be 'annual' or 'quarterly'")
        self.payout_frequency = payout_frequency
        if per_period_growth not in {"compounded", "g_over_4"}:
            raise ValueError("per_period_growth must be 'compounded' or 'g_over_4'")
        self.per_period_growth = per_period_growth

        # Filled by run_backtest
        self.payouts_df: pd.DataFrame | None = None

    # ------------------------ helpers ------------------------

    @staticmethod
    def _is_quarter_end_index(idx: pd.DatetimeIndex) -> pd.Series:
        """True on the last available row of each quarter in the input index."""
        q = idx.quarter
        last = np.r_[q[1:] != q[:-1], True]
        return pd.Series(last, index=idx)

    @staticmethod
    def _is_year_end_index(idx: pd.DatetimeIndex) -> pd.Series:
        """True on the last available row of each year in the input index."""
        y = idx.year
        last = np.r_[y[1:] != y[:-1], True]
        return pd.Series(last, index=idx)

    # ------------------------ core ---------------------------

    def run_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().sort_index()
        df.attrs['title'] = data.attrs.get('title', 'Endowment Spending (70/30)')
        df.attrs['ticker'] = data.attrs.get('ticker', 'IWV/AGG')

        if not {'close_IWV', 'close_AGG'}.issubset(df.columns):
            raise ValueError("DataFrame must contain columns: close_IWV, close_AGG")

        out = pd.DataFrame(index=df.index)
        out['price_IWV'] = df['close_IWV'].astype(float)
        out['price_AGG'] = df['close_AGG'].astype(float)

        # State variables
        cash = float(self.initial_cash)
        units_iwv = 0.0
        units_agg = 0.0

        ledger_rows = []

        # Buy initial positions at first bar close
        first = out.index[0]
        total_equity_prev = cash
        target_iwv_val = self.w_iwv * total_equity_prev
        target_agg_val = self.w_agg * total_equity_prev

        p_iwv_buy = out.at[first, 'price_IWV'] * (1 + self.slippage_pct)
        p_agg_buy = out.at[first, 'price_AGG'] * (1 + self.slippage_pct)

        units_iwv = 0.0 if p_iwv_buy <= 0 else target_iwv_val / p_iwv_buy
        units_agg = 0.0 if p_agg_buy <= 0 else target_agg_val / p_agg_buy

        gross_iwv = units_iwv * p_iwv_buy
        gross_agg = units_agg * p_agg_buy
        commission_init = (gross_iwv + gross_agg) * self.commission_pct
        cash = cash - gross_iwv - gross_agg - commission_init
        if cash < 0:
            cash = 0.0

        # Equity right after initial buys:
        initial_val_iwv = units_iwv * out.at[first, 'price_IWV']
        initial_val_agg = units_agg * out.at[first, 'price_AGG']
        initial_holdings = initial_val_iwv + initial_val_agg
        initial_total_equity = cash + initial_holdings

        # Determine payout event dates & per-event target
        idx = out.index
        is_qe = self._is_quarter_end_index(idx)
        is_ye = self._is_year_end_index(idx)

        if self.spending_rule == "growing_payout":
            start_year = int(idx[0].year)
            base_annual_budget = self.initial_spend_rate * initial_total_equity  # year 1 budget
            if self.payout_frequency == "annual":
                event_mask = is_ye

                def payout_amount_for_timestamp(ts) -> float:
                    years_elapsed = int(ts.year) - start_year
                    return base_annual_budget * ((1.0 + self.g) ** years_elapsed)
            else:
                # Quarterly payouts with quarterly growth
                event_mask = is_qe
                base_quarter_budget = base_annual_budget / 4.0  # first quarter = 0.25% of initial equity
                # per-quarter growth rate
                if self.per_period_growth == "compounded":
                    r_q = (1.0 + self.g) ** 0.25 - 1.0   # exact quarterly rate
                else:
                    r_q = self.g / 4.0                   # approximation
                q_counter = -1  # will become 0 on first payout quarter

                def payout_amount_for_timestamp(ts) -> float:
                    nonlocal q_counter
                    q_counter += 1
                    return base_quarter_budget * ((1.0 + r_q) ** q_counter)
        else:
            # percent_of_equity rule: each event pays payout_rate_quarterly * current equity
            event_mask = is_qe

        # Prepare output columns
        for col in [
            'units_IWV','units_AGG','val_IWV','val_AGG','cash','holdings','total_equity',
            'payout','cum_payout','payout_IWV','payout_AGG','returns','wealth_plus_paid'
        ]:
            out[col] = 0.0

        cum_paid = 0.0
        prev_total_equity = None

        # Iterate through time
        for ts in idx:
            p_iwv = out.at[ts, 'price_IWV']
            p_agg = out.at[ts, 'price_AGG']

            val_iwv = units_iwv * p_iwv
            val_agg = units_agg * p_agg
            holdings = val_iwv + val_agg
            total_equity = cash + holdings

            payout = 0.0
            payout_iwv = 0.0
            payout_agg = 0.0

            # Determine payout at event dates
            do_payout = bool(event_mask.at[ts]) and (total_equity > 0)

            if do_payout:
                if self.spending_rule == "growing_payout":
                    payout = float(payout_amount_for_timestamp(ts))
                else:
                    payout = self.payout_rate_q * total_equity

                # ----------- FUNDING THE PAYOUT (FIXED LOGIC) -----------
                # Decide how to fund payout across assets:
                if units_iwv > 0 and units_agg > 0:
                    share_iwv, share_agg = 0.5, 0.5       # even split when both positions exist
                elif units_iwv > 0:
                    share_iwv, share_agg = 1.0, 0.0       # all from IWV if AGG has no units
                elif units_agg > 0:
                    share_iwv, share_agg = 0.0, 1.0       # all from AGG if IWV has no units
                else:
                    share_iwv, share_agg = 0.0, 0.0       # nothing to sell

                target_iwv = payout * share_iwv
                target_agg = payout * share_agg

                # For a sell, include slippage: trade price lower than mark
                p_iwv_sell = p_iwv * (1 - self.slippage_pct)
                p_agg_sell = p_agg * (1 - self.slippage_pct)

                # Need gross sale so that net after commission >= target part
                denom = (1 - self.commission_pct) if self.commission_pct < 1 else 1.0
                gross_iwv_needed = target_iwv / denom if target_iwv > 0 else 0.0
                gross_agg_needed = target_agg / denom if target_agg > 0 else 0.0

                units_iwv_to_sell = 0.0 if p_iwv_sell <= 0 else gross_iwv_needed / p_iwv_sell
                units_agg_to_sell = 0.0 if p_agg_sell <= 0 else gross_agg_needed / p_agg_sell

                # Clamp by available units
                units_iwv_to_sell = min(units_iwv_to_sell, units_iwv)
                units_agg_to_sell = min(units_agg_to_sell, units_agg)

                # Execute sells
                gross_iwv_exec = units_iwv_to_sell * p_iwv_sell
                gross_agg_exec = units_agg_to_sell * p_agg_sell
                commission = (gross_iwv_exec + gross_agg_exec) * self.commission_pct

                # Add sale proceeds to cash, reduce positions
                cash += gross_iwv_exec + gross_agg_exec
                units_iwv -= units_iwv_to_sell
                units_agg -= units_agg_to_sell

                # Pay out (cash leaves system). If not enough raised (e.g., thin positions),
                # pay whatever is available up to the target payout.
                available = cash
                pay_amount = min(payout, available)
                cash -= pay_amount
                cum_paid += pay_amount

                # Attribute the ACTUAL paid amount to assets proportionally to net raised
                net_iwv = gross_iwv_exec * (1 - self.commission_pct)
                net_agg = gross_agg_exec * (1 - self.commission_pct)
                net_total = net_iwv + net_agg
                if net_total > 0:
                    scale = pay_amount / net_total
                    payout_iwv = net_iwv * scale
                    payout_agg = net_agg * scale
                else:
                    payout_iwv = payout_agg = 0.0

                # Ledger (one row per asset)
                ledger_rows.append({
                    'date': ts, 'asset': 'IWV',
                    'units_sold': units_iwv_to_sell,
                    'price': p_iwv_sell,
                    'gross_proceeds': gross_iwv_exec,
                    'commission_pct': self.commission_pct,
                    'est_net_proceeds': net_iwv,
                    'target_payout_total': payout,
                    'actual_payout_total': pay_amount,
                    'spending_rule': self.spending_rule,
                    'payout_frequency': self.payout_frequency if self.spending_rule == "growing_payout" else 'quarterly',
                    'per_period_growth': self.per_period_growth if self.spending_rule == "growing_payout" else None
                })
                ledger_rows.append({
                    'date': ts, 'asset': 'AGG',
                    'units_sold': units_agg_to_sell,
                    'price': p_agg_sell,
                    'gross_proceeds': gross_agg_exec,
                    'commission_pct': self.commission_pct,
                    'est_net_proceeds': net_agg,
                    'target_payout_total': payout,
                    'actual_payout_total': pay_amount,
                    'spending_rule': self.spending_rule,
                    'payout_frequency': self.payout_frequency if self.spending_rule == "growing_payout" else 'quarterly',
                    'per_period_growth': self.per_period_growth if self.spending_rule == "growing_payout" else None
                })

                # Recompute post-trade valuations
                val_iwv = units_iwv * p_iwv
                val_agg = units_agg * p_agg
                holdings = val_iwv + val_agg
                total_equity = cash + holdings

                # Report actual paid amount
                payout = pay_amount

            # Fill outputs
            out.at[ts, 'units_IWV'] = units_iwv
            out.at[ts, 'units_AGG'] = units_agg
            out.at[ts, 'val_IWV'] = val_iwv
            out.at[ts, 'val_AGG'] = val_agg
            out.at[ts, 'cash'] = cash
            out.at[ts, 'holdings'] = holdings
            out.at[ts, 'total_equity'] = total_equity
            out.at[ts, 'payout'] = payout
            out.at[ts, 'payout_IWV'] = payout_iwv
            out.at[ts, 'payout_AGG'] = payout_agg
            out.at[ts, 'cum_payout'] = cum_paid
            out.at[ts, 'wealth_plus_paid'] = total_equity + cum_paid

            if prev_total_equity is None:
                out.at[ts, 'returns'] = 0.0
            else:
                out.at[ts, 'returns'] = (total_equity - prev_total_equity) / max(prev_total_equity, 1e-9)

            prev_total_equity = total_equity

        # Build ledger dataframe
        self.payouts_df = pd.DataFrame(ledger_rows) if ledger_rows else pd.DataFrame(
            columns=[
                'date','asset','units_sold','price','gross_proceeds','commission_pct','est_net_proceeds',
                'target_payout_total','actual_payout_total','spending_rule','payout_frequency','per_period_growth'
            ]
        )
        if not self.payouts_df.empty:
            self.payouts_df = self.payouts_df.sort_values('date').reset_index(drop=True)

        # Attach attrs
        out.attrs = df.attrs
        out.attrs['payouts_df'] = self.payouts_df

        return out
