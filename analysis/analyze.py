# analysis/analyze.py

from config.secrets_config import *
from config.universe_config import *  # can export strategy classes and grid builders
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import BacktestEngine
from utils.pretty_print_df import *
from backtest.performance_metrics import extract_performance_metrics_dict
from utils.data_fetch import fetch_data_for_strategy
from backtest.performance_metrics import extract_performance_metrics_dict, extract_eif_metrics_dict


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # date axis formatting
from copy import deepcopy
from collections import defaultdict
# --- force a consistent RF into a metrics dict ---
from backtest.performance_metrics import (
    calculate_sharpe_ratio, calculate_sortino_ratio
)

def _apply_rf_to_metrics(metrics: dict, results_df, rf: float) -> dict:
    """Rewrite Sharpe/Sortino in `metrics` using rf, adding the keys if missing."""
    sr = float(calculate_sharpe_ratio(results_df, risk_free_rate=rf))
    so = float(calculate_sortino_ratio(results_df, risk_free_rate=rf))

    has_sharpe = False
    has_sortino = False
    for k in list(metrics.keys()):
        lk = k.lower()
        if "sharpe" in lk:
            metrics[k] = sr
            has_sharpe = True
        if "sortino" in lk:
            metrics[k] = so
            has_sortino = True

    if not has_sharpe:
        metrics["Sharpe"] = sr
    if not has_sortino:
        metrics["Sortino"] = so
    return metrics


# ---------- Shared plotting style ----------
XROT = 45        # x-axis label rotation (degrees)
XTICK_SIZE = 9   # x-axis tick label size
YTICK_SIZE = 9   # y-axis tick label size
YLABEL_SIZE = 10 # y-axis label font size

# Optional import of the endowment payout engine; fallback to BacktestEngine if unavailable
try:
    from backtest.endowment_payout_engine import EndowmentPayoutBacktestEngine
except Exception:
    EndowmentPayoutBacktestEngine = None


# ----------------------- Index / fetch helpers -----------------------

def _ensure_datetime_index(df: pd.DataFrame, label_for_logs: str = "") -> pd.DataFrame:
    """Make sure the DataFrame has a DatetimeIndex."""
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"⚠️ Could not convert index to datetime{(' for ' + label_for_logs) if label_for_logs else ''}: {e}")
    return df


def _fetch_iwv_agg_join(interval: str = "1d", start: str | None = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Fetch IWV and AGG OHLCV and inner-join on DatetimeIndex.
    Uses your existing generic fetcher behind the scenes.
    """
    IWV_FETCH_SETTINGS = {
        "title": "IWV", "ticker": "IWV", "interval": interval, "start": start, "end": end, "type": "equity"
    }
    AGG_FETCH_SETTINGS = {
        "title": "AGG", "ticker": "AGG", "interval": interval, "start": start, "end": end, "type": "equity"
    }

    iwv_df = fetch_data_for_strategy(IWV_FETCH_SETTINGS)
    agg_df = fetch_data_for_strategy(AGG_FETCH_SETTINGS)

    iwv_df = _ensure_datetime_index(iwv_df, "IWV")
    agg_df = _ensure_datetime_index(agg_df, "AGG")

    joined = iwv_df.join(
        agg_df, how="inner", lsuffix="_IWV", rsuffix="_AGG"
    ).sort_index()

    required_cols = {"close_IWV", "close_AGG"}
    if not required_cols.issubset(joined.columns):
        raise ValueError(f"Joined IWV/AGG frame is missing {required_cols}. Columns found: {list(joined.columns)}")

    joined.attrs['title'] = "IWV + AGG (joined)"
    joined.attrs['ticker'] = "IWV/AGG"
    return joined


def _synthesize_close_from_mix(df: pd.DataFrame, w_iwv: float, w_agg: float, base: float = 100.0) -> pd.Series:
    """
    Build a constant-mix (daily rebalanced) portfolio index from close_IWV and close_AGG.
    Returns a price-like index starting at 'base'.
    """
    if not {'close_IWV', 'close_AGG'}.issubset(df.columns):
        raise ValueError("Expected columns close_IWV and close_AGG to synthesize 'close'.")

    s = w_iwv + w_agg
    w_iwv = float(w_iwv) / s
    w_agg = float(w_agg) / s

    r_iwv = df['close_IWV'].pct_change()
    r_agg = df['close_AGG'].pct_change()
    port_r = (w_iwv * r_iwv) + (w_agg * r_agg)
    idx = (1.0 + port_r.fillna(0.0)).cumprod()
    return base * idx


def _data_key_for_strategy(strategy_class):
    """
    A cache key describing the required columns for a given strategy class.
    """
    req = getattr(strategy_class, "REQUIRED_COLUMNS", None)
    if not req:
        return ("DEFAULT", )
    return tuple(sorted(req))


def _fetch_by_strategy_requirements(settings: dict, fetch_cache: dict) -> pd.DataFrame:
    """
    Choose what data to fetch by inspecting strategy_class.REQUIRED_COLUMNS.
    Uses a cache so we don't refetch the same panel for the grid.
    """
    strategy_class = settings["strategy_class"]
    interval = settings.get("interval", "1d")
    start = settings.get("start")
    end = settings.get("end")

    key = (_data_key_for_strategy(strategy_class), interval, start, end)
    if key in fetch_cache:
        return fetch_cache[key].copy()

    required = getattr(strategy_class, "REQUIRED_COLUMNS", None)

    # Route: IWV + AGG joined panel
    if required and {"close_IWV", "close_AGG"}.issubset(required):
        df = _fetch_iwv_agg_join(interval=interval, start=start, end=end)
    else:
        # Generic path (expects your fetcher to use settings['ticker'] or other params)
        df = fetch_data_for_strategy(settings)

    df = _ensure_datetime_index(df)
    fetch_cache[key] = df
    return df.copy()


# ----------------------- Strategy comparers -----------------------

def compare_strategies(strategy_settings_list):
    """
    Runs, trims, collects metrics, and plots strategies from a list of strategy settings dicts.
    Data is chosen based on strategy_class.REQUIRED_COLUMNS (no fake tickers needed).
    """
    # >>> keep table + chart on the SAME rf <<<
    RF = 0.025

    raw_data = []
    fetch_cache = {}

    # Step 1: Fetch and tag data based on strategy requirements
    for settings in strategy_settings_list:
        df = _fetch_by_strategy_requirements(settings, fetch_cache)

        strategy_class = settings["strategy_class"]
        title = settings.get('title', strategy_class.__name__)

        df.attrs['title'] = title
        df.attrs['ticker'] = strategy_class.__name__

        raw_data.append((settings, df))

    if not raw_data:
        print("No strategies provided.")
        return

    # Step 2: Align start date
    latest_start = max(df.index.min() for _, df in raw_data)

    trimmed_data = []
    for settings, df in raw_data:
        trimmed = df[df.index >= latest_start].copy()
        trimmed.attrs['title'] = df.attrs['title']
        trimmed.attrs['ticker'] = df.attrs['ticker']
        trimmed_data.append((settings, trimmed))

    # Step 3: Run each strategy + engine and collect metrics
    metrics_list = []
    results = []
    for settings, df in trimmed_data:
        strategy_class = settings["strategy_class"]
        strategy_args = {
            k: v for k, v in settings.items()
            if k not in ['strategy_class', 'title', 'ticker', 'engine_class', 'engine_kwargs',
                         'interval', 'start', 'end', 'label']
        }
        strategy = strategy_class(**strategy_args)

        engine_cls = settings.get("engine_class", BacktestEngine)
        engine_kwargs = settings.get("engine_kwargs", {})
        try:
            backtester = engine_cls(**engine_kwargs)
        except TypeError:
            backtester = BacktestEngine()

        signal_df = strategy.generate_signals(df)
        result_df = backtester.run_backtest(signal_df)

        label = settings.get("label") or settings.get("title")
        result_df.attrs['title']  = label
        result_df.attrs['ticker'] = label

        # >>> make table use the same RF <<<
                # >>> compute metrics, then force RF-consistent Sharpe/Sortino <<<
        if strategy_class.__name__ in ("StaticMixIWVAGG", "StaticMixIWV_VXUS_AGG"):
            metrics = extract_eif_metrics_dict(result_df)  # no rf arg supported
            metrics = _apply_rf_to_metrics(metrics, result_df, rf=RF)
        else:
            # If your extract_performance_metrics_dict accepts rf, great;
            # if not, the helper will still enforce RF.
            try:
                metrics = extract_performance_metrics_dict(result_df, risk_free_rate=RF)
            except TypeError:
                metrics = extract_performance_metrics_dict(result_df)
            metrics = _apply_rf_to_metrics(metrics, result_df, rf=RF)


        metrics_list.append(metrics)
        results.append(result_df)

    # Step 4: Format and print metrics
    metrics_df = pd.DataFrame(metrics_list)

    df_fmt = metrics_df.copy()
    for col in df_fmt.columns:
        col_lower = col.lower()
        if col_lower.startswith("sharpe") or col_lower.startswith("sortino"):
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.5f}" if pd.notnull(x) else "")
        elif any(k in col_lower for k in ["cagr", "return", "stdev", "vol", "drawdown", "variance"]):
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

    pretty_print_df(df_fmt)

    # Step 5: Plot equity curves (interactive tabs: linear/log/frontier) with the SAME rf
    fig = plt.figure(figsize=(12, 6))
    plot_equity_tab(fig, results, normalize=True, risk_free_rate=RF)

# ----------------------- Endowment payout analysis -----------------------

def _extract_payout_events_df(results_df: pd.DataFrame, payouts_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build a minimal (date, amount) payout events DataFrame.

    Priority:
    1) Use results_df['payout'] > 0 if present.
    2) Else, aggregate backtester.payouts_df (sum est_net_proceeds by date) if present.
    """
    # Path 1: from results_df['payout']
    if 'payout' in results_df.columns:
        events = results_df.loc[results_df['payout'] > 0, 'payout']
        if not events.empty:
            out = events.reset_index().rename(columns={'index': 'date', 'payout': 'amount'})
            out['date'] = pd.to_datetime(out['date'])
            out['amount'] = out['amount'].astype(float)
            out = out.sort_values('date').reset_index(drop=True)
            out.attrs = {}  # prevent pandas attrs-comparison issues on merge
            return out

    # Path 2: from ledger payouts_df (sum by date)
    if payouts_df is not None and not payouts_df.empty:
        col = 'est_net_proceeds' if 'est_net_proceeds' in payouts_df.columns else (
              'gross_proceeds' if 'gross_proceeds' in payouts_df.columns else None)
        if col:
            grp = payouts_df.groupby('date', as_index=False)[col].sum().rename(columns={col: 'amount'})
            grp['date'] = pd.to_datetime(grp['date'])
            grp['amount'] = grp['amount'].astype(float)
            grp = grp.loc[grp['amount'] > 0]
            grp = grp.sort_values('date').reset_index(drop=True)
            grp.attrs = {}  # prevent attrs carryover
            return grp

    return pd.DataFrame(columns=['date', 'amount'])


def analyze_strategy(strategy_settings):
    """
    Analyze a single strategy, showing FIVE TABS:
      Tab 1: 70/30 Percent-of-Equity — Payouts panel + Equity vs No-Payout
      Tab 2: 70/30 Growing (quarterly) — Payouts panel + Equity vs No-Payout
      Tab 3: 100% IWV Percent-of-Equity — Payouts panel + Equity vs No-Payout
      Tab 4: 100% IWV Growing (quarterly) — Payouts panel + Equity vs No-Payout
      Tab 5: Comparison — TOP: overlay both mixes’ payout amounts (four bars) + cumulative lines
                         BOTTOM: No-payout benchmarks (70/30 & IWV-only) vs all four payout NAVs
    """
    # Build the input DataFrame according to the strategy requirements
    fetch_cache = {}
    df = _fetch_by_strategy_requirements(strategy_settings, fetch_cache)

    # ✅ Ensure datetime index
    df = _ensure_datetime_index(df)

    # ------------------------------------------------------------
    # Instantiate strategy and generate signals once
    # ------------------------------------------------------------
    strategy_kwargs = {k: v for k, v in strategy_settings.items()
                       if k not in ['strategy_class', 'title', 'ticker', 'engine_class', 'engine_kwargs',
                                    'iwv100_growth_overrides', 'interval', 'start', 'end', 'label']}
    strategy = strategy_settings["strategy_class"](**strategy_kwargs)

    base_engine_kwargs = deepcopy(strategy_settings.get("engine_kwargs", {}))
    base_w_iwv = float(base_engine_kwargs.get('w_iwv', 0.70))
    base_w_agg = float(base_engine_kwargs.get('w_agg', 0.30))
    g_val = base_engine_kwargs.get('g', None)
    payout_rate_q = float(base_engine_kwargs.get('payout_rate_quarterly', 0.0025))

    # IWV-only growing overrides (optional per-universe_config)
    iwv_over = deepcopy(strategy_settings.get("iwv100_growth_overrides", {}))
    g_val_iwv = iwv_over.get('g', g_val)
    init_spend_iwv = iwv_over.get('initial_spend_rate', base_engine_kwargs.get("initial_spend_rate", 0.01))

    signal_df = strategy.generate_signals(df)

    # Synthesize 'close' if needed so plotters have a display price
    if 'close' not in signal_df.columns and {'close_IWV', 'close_AGG'}.issubset(signal_df.columns):
        signal_df['close'] = _synthesize_close_from_mix(signal_df, base_w_iwv, base_w_agg, base=100.0)
    if 'signal' not in signal_df.columns:
        signal_df['signal'] = 1.0

    # Helper: run a variant and build its own benchmark (no-payout) using THAT variant's weights
    def _run_variant(name: str, w_iwv: float, w_agg: float, overrides: dict):
        engine_cls = strategy_settings.get("engine_class", BacktestEngine)
        kw = deepcopy(base_engine_kwargs)
        kw.update({"w_iwv": w_iwv, "w_agg": w_agg})
        kw.update(overrides)
        try:
            engine = engine_cls(**kw)
        except TypeError:
            engine = BacktestEngine()

        res = engine.run_backtest(signal_df)
        payouts = getattr(engine, "payouts_df", None)

        # Inject a proper benchmark 'close' for this weight mix, scaled to NAV start
        if 'close' not in res.columns and {'close_IWV', 'close_AGG'}.issubset(signal_df.columns):
            bench_idx = _synthesize_close_from_mix(signal_df, w_iwv, w_agg, base=1.0)
            if len(res) > 0:
                start_equity = float(res['total_equity'].iloc[0])
                start_bench = float(bench_idx.iloc[0]) if bench_idx.iloc[0] != 0 else 1.0
                res['close'] = bench_idx * (start_equity / start_bench)
            else:
                res['close'] = bench_idx

        title = f"{name}"
        return res, payouts, title

    # ---------------- Run four variants ----------------
    # 70/30 percent-of-equity
    res_pct7030, pay_pct7030, title_pct7030 = _run_variant(
        name=f"70/30 Percent-of-Equity (q={payout_rate_q*100:.3f}%)",
        w_iwv=base_w_iwv, w_agg=base_w_agg,
        overrides={"spending_rule": "percent_of_equity", "payout_rate_quarterly": payout_rate_q}
    )

    # 70/30 growing (quarterly)
    res_grow7030, pay_grow7030, title_grow7030 = _run_variant(
        name=f"70/30 Growing (quarterly, g={float(g_val) if g_val is not None else 0.0:.3%})",
        w_iwv=base_w_iwv, w_agg=base_w_agg,
        overrides={
            "spending_rule": "growing_payout",
            "g": float(g_val) if g_val is not None else 0.0,
            "initial_spend_rate": float(base_engine_kwargs.get("initial_spend_rate", 0.01)),
            "payout_frequency": "quarterly",
            "per_period_growth": base_engine_kwargs.get("per_period_growth", "compounded"),
        }
    )

    # 100% IWV percent-of-equity
    res_pctIWV, pay_pctIWV, title_pctIWV = _run_variant(
        name=f"100% IWV Percent-of-Equity (q={payout_rate_q*100:.3f}%)",
        w_iwv=1.0, w_agg=0.0,
        overrides={"spending_rule": "percent_of_equity", "payout_rate_quarterly": payout_rate_q}
    )

    # 100% IWV growing (quarterly) — uses IWV-only overrides if provided
    res_growIWV, pay_growIWV, title_growIWV = _run_variant(
        name=f"100% IWV Growing (quarterly, g={float(g_val_iwv) if g_val_iwv is not None else 0.0:.3%})",
        w_iwv=1.0, w_agg=0.0,
        overrides={
            "spending_rule": "growing_payout",
            "g": float(g_val_iwv) if g_val_iwv is not None else 0.0,
            "initial_spend_rate": float(init_spend_iwv),
            "payout_frequency": "quarterly",
            "per_period_growth": iwv_over.get("per_period_growth", base_engine_kwargs.get("per_period_growth", "compounded")),
            **{k: v for k, v in iwv_over.items() if k in ("initial_cash", "commission_pct", "slippage_pct")}
        }
    )

    # ---------------- Plot helpers ----------------
    def _apply_xaxis_format(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))

    def _force_top_xticks(ax):
        _apply_xaxis_format(ax)
        ax.tick_params(axis='x', which='both', labelbottom=True,
                       labelrotation=XROT, labelsize=XTICK_SIZE)
        ax.tick_params(axis='y', labelsize=YTICK_SIZE)
        for lab in ax.get_xticklabels():
            lab.set_visible(True)
            lab.set_horizontalalignment('right')

    def _draw_single_panel(fig, res_df: pd.DataFrame, panel_title: str):
        """Top: quarterly payout bars + cumulative line. Bottom: equity vs benchmark."""
        fig.clear()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

        # Top: payout bars (right axis) + cumulative line (left axis)
        events = res_df.loc[res_df['payout'] > 0, 'payout']
        ax_top2 = ax_top.twinx()

        # Ensure lines (ax_top) draw above bars (ax_top2)
        ax_top.set_zorder(ax_top2.get_zorder() + 1)
        ax_top.patch.set_visible(False)

        # y-axis tick sizes/labels
        ax_top.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top2.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top.set_ylabel("Cumulative ($)", fontsize=YLABEL_SIZE)
        ax_top2.set_ylabel("Period payout ($)", fontsize=YLABEL_SIZE)

        if not events.empty:
            x = mdates.date2num(np.array(events.index.to_pydatetime()))
            if len(x) >= 2:
                period = float(np.median(np.diff(x)))
            else:
                period = 90.0  # ~quarter fallback
            width_days = period * 0.95   # fill ~95% of period
            ax_top2.bar(
                x, events.values, width=width_days, align='center',
                alpha=0.8, edgecolor='none', label="Payout (period)", zorder=1
            )
            ymax = float(events.max())
            ax_top2.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

        ax_top.plot(
            res_df.index, res_df['cum_payout'], linewidth=1.5, color="black",
            label="Cumulative payouts", zorder=3
        )
        ax_top.set_title(panel_title)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left")

        # ✅ Show/format x-axis on TOP chart too
        _force_top_xticks(ax_top)

        # Bottom: NAV vs benchmark (no payout)
        ax_bot.plot(res_df.index, res_df['total_equity'], linewidth=1.5, label="Strategy NAV")
        if 'close' in res_df.columns:
            ax_bot.plot(res_df.index, res_df['close'], linewidth=1.5, alpha=0.9, label="Benchmark (no payouts)")
        ax_bot.set_title(f"Equity Curve")
        ax_bot.set_ylabel("Equity ($)", fontsize=YLABEL_SIZE)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper left")

        _apply_xaxis_format(ax_bot)
        ax_bot.tick_params(axis='x', which='both', labelrotation=XROT, labelsize=XTICK_SIZE)
        ax_bot.tick_params(axis='y', labelsize=YTICK_SIZE)
        for lab in ax_bot.get_xticklabels():
            lab.set_horizontalalignment('right')

        # Do NOT call fig.autofmt_xdate(); it tends to re-hide top shared labels
        fig.tight_layout()

    def _draw_compare_panel(fig):
        """Top: four payout series (bars) + cumulative lines; Bottom: 2 benchmarks + 4 NAV lines."""
        fig.clear()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)
        ax_top2 = ax_top.twinx()

        # Ensure lines (ax_top) draw above bars (ax_top2)
        ax_top.set_zorder(ax_top2.get_zorder() + 1)
        ax_top.patch.set_visible(False)

        # y-axis sizes/labels for top
        ax_top.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top2.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top.set_ylabel("Cumulative ($)", fontsize=YLABEL_SIZE)
        ax_top2.set_ylabel("Payout per period ($)", fontsize=YLABEL_SIZE)

        # Build aligned payout amounts (Series -> DataFrame)
        ev_pct7030 = res_pct7030.loc[res_pct7030['payout'] > 0, 'payout']
        ev_grow7030 = res_grow7030.loc[res_grow7030['payout'] > 0, 'payout']
        ev_pctIWV = res_pctIWV.loc[res_pctIWV['payout'] > 0, 'payout']
        ev_growIWV = res_growIWV.loc[res_growIWV['payout'] > 0, 'payout']

        combined = pd.DataFrame({
            "70/30 %": ev_pct7030,
            "70/30 g": ev_grow7030,
            "IWV %": ev_pctIWV,
            "IWV g": ev_growIWV
        }).fillna(0.0)

        if not combined.empty:
            x = mdates.date2num(np.array(combined.index.to_pydatetime()))
            if len(x) >= 2:
                period = float(np.median(np.diff(x)))
            else:
                period = 90.0  # ~quarter fallback
            total_band = period * 0.98  # ~100% of period width across all 4 bars
            width = total_band / 4.0
            offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

            colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
            labels = combined.columns.tolist()
            for i, col in enumerate(labels):
                ax_top2.bar(
                    x + offsets[i], combined[col].values, width=width, align='center',
                    alpha=0.6, edgecolor='none', color=colors[i], label=f"Payout ({col})", zorder=1
                )

            ymax = float(combined.values.max())
            ax_top2.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
            ax_top2.margins(y=0.1)

        # Cumulative payout lines (left axis)
        ax_top.plot(res_pct7030.index, res_pct7030['cum_payout'], linewidth=1.5,
                    color='tab:blue', label="70/30 %", zorder=3)
        ax_top.plot(res_grow7030.index, res_grow7030['cum_payout'], linewidth=1.5,
                    color='tab:red', label="70/30 g", zorder=3)
        ax_top.plot(res_pctIWV.index, res_pctIWV['cum_payout'], linewidth=1.5,
                    color='tab:green', label="IWV %", zorder=3)
        ax_top.plot(res_growIWV.index, res_growIWV['cum_payout'], linewidth=1.5,
                    color='tab:orange', label="IWV g", zorder=3)

        ax_top.set_title("Payout Comparisons")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left")

        # ✅ Show/format x-axis on TOP chart too
        _force_top_xticks(ax_top)

        # Bottom: benchmarks + four NAVs
        ax_bot.plot(res_pct7030.index, res_pct7030['close'], linewidth=1.5, alpha=0.9, color='black', label="Benchmark 70/30 (no payouts)")
        ax_bot.plot(res_pctIWV.index, res_pctIWV['close'], linewidth=1.5, alpha=0.9, color='gray', label="Benchmark IWV-only (no payouts)")

        ax_bot.plot(res_pct7030.index, res_pct7030['total_equity'], linewidth=1.5, color='tab:blue', label="NAV 70/30 %")
        ax_bot.plot(res_grow7030.index, res_grow7030['total_equity'], linewidth=1.5, color='tab:red', label="NAV 70/30 g")
        ax_bot.plot(res_pctIWV.index, res_pctIWV['total_equity'], linewidth=1.5, color='tab:green', label="NAV IWV %")
        ax_bot.plot(res_growIWV.index, res_growIWV['total_equity'], linewidth=1.5, color='tab:orange', label="NAV IWV g")

        ax_bot.set_title("Benchmarks vs Payout Strategies")
        ax_bot.set_ylabel("Equity ($)", fontsize=YLABEL_SIZE)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper left")

        _apply_xaxis_format(ax_bot)
        ax_bot.tick_params(axis='x', which='both', labelrotation=XROT, labelsize=XTICK_SIZE)
        ax_bot.tick_params(axis='y', labelsize=YTICK_SIZE)
        for lab in ax_bot.get_xticklabels():
            lab.set_horizontalalignment('right')

        # Avoid fig.autofmt_xdate(); it can hide shared labels
        fig.tight_layout()

    # ---------------- Tabs ----------------
    plot_funcs = [
        ("70/30: Percent-of-Equity",      lambda fig: _draw_single_panel(fig, res_pct7030, title_pct7030)),
        ("70/30: Growing (Quarterly)",    lambda fig: _draw_single_panel(fig, res_grow7030, title_grow7030)),
        ("100% IWV: Percent-of-Equity",   lambda fig: _draw_single_panel(fig, res_pctIWV, title_pctIWV)),
        ("100% IWV: Growing (Quarterly)", lambda fig: _draw_single_panel(fig, res_growIWV, title_growIWV)),
        ("Comparison: Payouts & Equity",  lambda fig: _draw_compare_panel(fig)),
    ]

    idx = [0]
    fig = plt.figure(figsize=(11.5, 7))
    fig.patch.set_facecolor('white')

    def draw():
        fig.clear()
        _, plot_func = plot_funcs[idx[0]]
        plot_func(fig)
        # Extra right padding so twin y-axis labels don't clip
        fig.subplots_adjust(left=0.1, right=0.92, top=0.95, bottom=0.13)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(plot_funcs)
            draw()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(plot_funcs)
            draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()

    # ---------------- Metrics table (optional) ----------------
    metrics_rows = []
    labels = [title_pct7030, title_grow7030, title_pctIWV, title_growIWV]
    for title, res in zip(labels, [res_pct7030, res_grow7030, res_pctIWV, res_growIWV]):
        m = extract_performance_metrics_dict(res)
        m['Strategy'] = title
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows).set_index('Strategy')
    pd.set_option('display.float_format', '{:.8f}'.format)
    pretty_print_df(metrics_df)

def _fetch_by_strategy_requirements(settings: dict, fetch_cache: dict) -> pd.DataFrame:
    """
    Choose what data to fetch by inspecting strategy_class.REQUIRED_COLUMNS.
    Uses a cache so we don't refetch the same panel for the grid.

    Supports:
      - {"close_IWV", "close_AGG"}         -> _fetch_iwv_agg_join()
      - {"close_IWV", "close_VXUS", "close_AGG"} -> _fetch_iwv_vxus_agg_join()
      - otherwise -> fetch_data_for_strategy(settings)
    """
    strategy_class = settings["strategy_class"]
    interval = settings.get("interval", "1d")
    start = settings.get("start")
    end = settings.get("end")

    key = (_data_key_for_strategy(strategy_class), interval, start, end)
    if key in fetch_cache:
        return fetch_cache[key].copy()

    required = getattr(strategy_class, "REQUIRED_COLUMNS", None)

    if required and {"close_IWV", "close_VXUS", "close_AGG"}.issubset(required):
        df = _fetch_iwv_vxus_agg_join(interval=interval, start=start, end=end)
    elif required and {"close_IWV", "close_AGG"}.issubset(required):
        df = _fetch_iwv_agg_join(interval=interval, start=start, end=end)
    else:
        df = fetch_data_for_strategy(settings)

    df = _ensure_datetime_index(df)
    fetch_cache[key] = df
    return df.copy()


def _fetch_iwv_vxus_agg_join(interval: str = "1d",
                              start: str | None = "2015-01-01",
                              end: str | None = None) -> pd.DataFrame:
    """
    Fetch IWV, VXUS, AGG; rename to suffixed OHLCV columns; inner-join on DatetimeIndex.
    Produces: close_IWV, close_VXUS, close_AGG (and corresponding open/high/low/volume).
    """

    def _ensure_dtindex(df: pd.DataFrame, who: str) -> pd.DataFrame:
        df = _ensure_datetime_index(df, who)
        # Make sure columns are lower-case (your fetcher already does this)
        df.columns = [str(c).lower() for c in df.columns]
        return df

    def _suffix_ohlcv(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        # Only suffix standard OHLCV columns if present
        ohlcv = ("open", "high", "low", "close", "volume", "adj close", "adj_close")
        rename_map = {c: f"{c}_{suffix}" for c in ohlcv if c in df.columns}
        return df.rename(columns=rename_map)

    # --- Fetch raw panels ---
    IWV_SETTINGS = {"title": "IWV", "ticker": "IWV", "interval": interval, "start": start, "end": end, "type": "equity"}
    VXUS_SETTINGS = {"title": "VXUS", "ticker": "VXUS", "interval": interval, "start": start, "end": end, "type": "equity"}
    AGG_SETTINGS  = {"title": "AGG",  "ticker": "AGG",  "interval": interval, "start": start, "end": end, "type": "equity"}

    iwv = _ensure_dtindex(fetch_data_for_strategy(IWV_SETTINGS), "IWV")
    vxus = _ensure_dtindex(fetch_data_for_strategy(VXUS_SETTINGS), "VXUS")
    agg = _ensure_dtindex(fetch_data_for_strategy(AGG_SETTINGS),  "AGG")

    # --- Add suffixes BEFORE joining so there are no collisions ---
    iwv = _suffix_ohlcv(iwv,  "IWV")
    vxus = _suffix_ohlcv(vxus, "VXUS")
    agg  = _suffix_ohlcv(agg,  "AGG")

    # --- Inner-join on common dates ---
    joined = iwv.join(vxus, how="inner").join(agg, how="inner").sort_index()

    # --- Validate required cols exist ---
    req = {"close_IWV", "close_VXUS", "close_AGG"}
    if not req.issubset(joined.columns):
        raise ValueError(f"IWV/VXUS/AGG frame missing {req}. Found: {list(joined.columns)}")

    joined.attrs["title"]  = "IWV + VXUS + AGG (joined)"
    joined.attrs["ticker"] = "IWV/VXUS/AGG"
    return joined
