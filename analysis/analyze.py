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

def _mix_label(w_iwv: float, w_agg: float) -> str:
    """Return labels like '85/15' from weights 0.85/0.15."""
    return f"{int(round(w_iwv*100))}/{int(round(w_agg*100))}"


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
        df = fetch_data_for_strategy(settings)   # only for single-ticker strategies

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
    Analyze a single strategy with an endowment payout engine, showing THREE TABS:
      Tab 1: 85/15 Percent-of-Equity — Payouts panel + Equity vs No-Payout
      Tab 2: 70/30 Percent-of-Equity — Payouts panel + Equity vs No-Payout
      Tab 3: Comparison — TOP: both mixes’ payout amounts + cumulative lines
                          BOTTOM: No-payout benchmarks (85/15 & 70/30) vs both payout NAVs

    This function feeds the payout engine a panel with close_IWV & close_AGG (what it expects),
    and still synthesizes a blended 'close' for plotting the benchmark lines.
    """
    # ---------- Fetch data by strategy requirements ----------
    fetch_cache = {}
    df = _fetch_by_strategy_requirements(strategy_settings, fetch_cache)

    # ✅ Ensure datetime index
    df = _ensure_datetime_index(df)

    # ---------- Read base engine kwargs / weights ----------
    base_engine_kwargs = deepcopy(strategy_settings.get("engine_kwargs", {}))
    # Fallbacks if not present
    base_w_iwv = float(base_engine_kwargs.get('w_iwv', 0.85))
    base_w_agg = float(base_engine_kwargs.get('w_agg', 0.15))

    # Percent-of-equity payout rate (quarterly) — default 0.25%
    payout_rate_q = float(base_engine_kwargs.get('payout_rate_quarterly', 0.0025))

    # ---------- Build the DataFrame we pass to the engine ----------
    # If we're using the payout engine, it needs close_IWV/close_AGG columns present.
    engine_cls = strategy_settings.get("engine_class", BacktestEngine)
    try:
        from backtest.endowment_payout_engine import EndowmentPayoutBacktestEngine
    except Exception:
        EndowmentPayoutBacktestEngine = None

    if engine_cls is EndowmentPayoutBacktestEngine:
        # Use the joined IWV/AGG panel as-is so required columns exist
        signal_df = df.copy()
        if 'signal' not in signal_df.columns:
            signal_df['signal'] = 1.0
        # For plotting, synthesize a blended 'close' at the *current* base weights
        if 'close' not in signal_df.columns and {'close_IWV', 'close_AGG'}.issubset(signal_df.columns):
            signal_df['close'] = _synthesize_close_from_mix(signal_df, base_w_iwv, base_w_agg, base=100.0)
        signal_df.attrs['title'] = strategy_settings.get('title', 'IWV/AGG (panel)')
        signal_df.attrs['ticker'] = signal_df.attrs['title']
    else:
        # Safety: non-payout engines (not the case here), keep previous behavior
        strategy_class = strategy_settings["strategy_class"]
        strategy_kwargs = {k: v for k, v in strategy_settings.items()
                           if k not in ['strategy_class', 'title', 'ticker', 'engine_class', 'engine_kwargs',
                                        'iwv100_growth_overrides', 'interval', 'start', 'end', 'label']}
        strategy = strategy_class(**strategy_kwargs)
        signal_df = strategy.generate_signals(df)
        if 'close' not in signal_df.columns and {'close_IWV', 'close_AGG'}.issubset(df.columns):
            signal_df['close'] = _synthesize_close_from_mix(df, base_w_iwv, base_w_agg, base=100.0)
        if 'signal' not in signal_df.columns:
            signal_df['signal'] = 1.0

    # ---------- Variant runner ----------
    def _run_variant(name: str, w_iwv: float, w_agg: float, overrides: dict):
        """Run one mix through the chosen engine, and synthesize its own benchmark 'close'."""
        engine_cls_loc = strategy_settings.get("engine_class", BacktestEngine)
        kw = deepcopy(base_engine_kwargs)
        kw.update({"w_iwv": w_iwv, "w_agg": w_agg})
        kw.update(overrides)
        try:
            engine = engine_cls_loc(**kw)
        except TypeError:
            engine = BacktestEngine()

        # IMPORTANT: the payout engine wants close_IWV & close_AGG present in signal_df
        res = engine.run_backtest(signal_df)
        payouts = getattr(engine, "payouts_df", None)

        # Inject a proper benchmark 'close' for this mix, scaled to NAV start
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

    # ---------- Run two variants: 85/15 and 70/30 (percent-of-equity @ 0.25%/q) ----------
    name_8515 = f"{_mix_label(0.85, 0.15)} Percent-of-Equity (q={payout_rate_q*100:.3f}%)"
    res_8515, pay_8515, title_8515 = _run_variant(
        name=name_8515,
        w_iwv=0.85, w_agg=0.15,
        overrides={"spending_rule": "percent_of_equity", "payout_rate_quarterly": payout_rate_q}
    )

    name_7030 = f"{_mix_label(0.70, 0.30)} Percent-of-Equity (q={payout_rate_q*100:.3f}%)"
    res_7030, pay_7030, title_7030 = _run_variant(
        name=name_7030,
        w_iwv=0.70, w_agg=0.30,
        overrides={"spending_rule": "percent_of_equity", "payout_rate_quarterly": payout_rate_q}
    )

    # ---------- Plot helpers ----------
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
        """Top: quarterly payout bars + cumulative line. Bottom: equity vs benchmark (no payouts)."""
        fig.clear()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

        # Top: payout bars (right) + cumulative line (left)
        events = res_df.loc[res_df['payout'] > 0, 'payout'] if 'payout' in res_df.columns else pd.Series(dtype=float)
        ax_top2 = ax_top.twinx()
        ax_top.set_zorder(ax_top2.get_zorder() + 1)
        ax_top.patch.set_visible(False)
        ax_top.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top2.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top.set_ylabel("Cumulative ($)", fontsize=YLABEL_SIZE)
        ax_top2.set_ylabel("Period payout ($)", fontsize=YLABEL_SIZE)

        if not events.empty:
            x = mdates.date2num(np.array(events.index.to_pydatetime()))
            period = float(np.median(np.diff(x))) if len(x) >= 2 else 90.0
            width_days = period * 0.95
            ax_top2.bar(x, events.values, width=width_days, align='center',
                        alpha=0.8, edgecolor='none', zorder=1)
            ymax = float(events.max())
            ax_top2.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

        if 'cum_payout' in res_df.columns:
            ax_top.plot(res_df.index, res_df['cum_payout'], linewidth=1.5, color="black",
                        label="Cumulative payouts", zorder=3)

        ax_top.set_title(panel_title)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left")
        _force_top_xticks(ax_top)

        # Bottom: NAV vs benchmark (no payout)
        ax_bot.plot(res_df.index, res_df['total_equity'], linewidth=1.5, label="Strategy NAV")
        if 'close' in res_df.columns:
            ax_bot.plot(res_df.index, res_df['close'], linewidth=1.5, alpha=0.9, label="Benchmark (no payouts)")
        ax_bot.set_title("Equity Curve")
        ax_bot.set_ylabel("Equity ($)", fontsize=YLABEL_SIZE)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper left")
        _apply_xaxis_format(ax_bot)
        ax_bot.tick_params(axis='x', which='both', labelrotation=XROT, labelsize=XTICK_SIZE)
        ax_bot.tick_params(axis='y', labelsize=YTICK_SIZE)
        for lab in ax_bot.get_xticklabels():
            lab.set_horizontalalignment('right')

        fig.tight_layout()

    def _draw_compare_panel(fig):
        """Top: both payout series (bars) + cumulative lines; Bottom: benchmarks + both NAVs."""
        fig.clear()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)
        ax_top2 = ax_top.twinx()

        ax_top.set_zorder(ax_top2.get_zorder() + 1)
        ax_top.patch.set_visible(False)
        ax_top.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top2.tick_params(axis='y', labelsize=YTICK_SIZE)
        ax_top.set_ylabel("Cumulative ($)", fontsize=YLABEL_SIZE)
        ax_top2.set_ylabel("Payout per period ($)", fontsize=YLABEL_SIZE)

        ev_8515 = res_8515.loc[res_8515['payout'] > 0, 'payout'] if 'payout' in res_8515.columns else pd.Series(dtype=float)
        ev_7030 = res_7030.loc[res_7030['payout'] > 0, 'payout'] if 'payout' in res_7030.columns else pd.Series(dtype=float)

        combined = pd.DataFrame({"85/15 %": ev_8515, "70/30 %": ev_7030}).fillna(0.0)
        if not combined.empty:
            x = mdates.date2num(np.array(combined.index.to_pydatetime()))
            period = float(np.median(np.diff(x))) if len(x) >= 2 else 90.0
            total_band = period * 0.98
            width = total_band / 2.0
            offsets = np.array([-0.5, 0.5]) * width
            colors = ['tab:blue', 'tab:orange']
            for i, col in enumerate(combined.columns):
                ax_top2.bar(x + offsets[i], combined[col].values, width=width, align='center',
                            alpha=0.6, edgecolor='none', color=colors[i], label=f"Payout ({col})", zorder=1)
            ymax = float(combined.values.max())
            ax_top2.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
            ax_top2.margins(y=0.1)

        if 'cum_payout' in res_8515.columns:
            ax_top.plot(res_8515.index, res_8515['cum_payout'], linewidth=1.5, color='tab:orange',
                        label="85/15 cumulative", zorder=3)
        if 'cum_payout' in res_7030.columns:
            ax_top.plot(res_7030.index, res_7030['cum_payout'], linewidth=1.5, color='tab:blue',
                        label="70/30 cumulative", zorder=3)

        ax_top.set_title("Payout Comparisons")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left")
        _force_top_xticks(ax_top)

        # Bottom: benchmarks + two NAVs
        if 'close' in res_8515.columns:
            ax_bot.plot(res_8515.index, res_8515['close'], linewidth=1.5, alpha=0.9, color='black', label="Benchmark 85/15 (no payouts)")
        if 'close' in res_7030.columns:
            ax_bot.plot(res_7030.index, res_7030['close'], linewidth=1.5, alpha=0.9, color='gray', label="Benchmark 70/30 (no payouts)")
        ax_bot.plot(res_8515.index, res_8515['total_equity'], linewidth=1.5, color='tab:orange',   label="NAV 85/15 %")
        ax_bot.plot(res_7030.index, res_7030['total_equity'], linewidth=1.5, color='tab:blue', label="NAV 70/30 %")
        ax_bot.set_title("Benchmarks vs Payout Strategies")
        ax_bot.set_ylabel("Equity ($)", fontsize=YLABEL_SIZE)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper left")
        _apply_xaxis_format(ax_bot)
        ax_bot.tick_params(axis='x', which='both', labelrotation=XROT, labelsize=XTICK_SIZE)
        ax_bot.tick_params(axis='y', labelsize=YTICK_SIZE)
        for lab in ax_bot.get_xticklabels():
            lab.set_horizontalalignment('right')

        fig.tight_layout()

    # ---------- Tabs ----------
    plot_funcs = [
        (title_8515, lambda fig: _draw_single_panel(fig, res_8515, title_8515)),
        (title_7030, lambda fig: _draw_single_panel(fig, res_7030, title_7030)),
        ("Comparison: Payouts & Equity", lambda fig: _draw_compare_panel(fig)),
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

    # ---------- Metrics table (optional) ----------
    metrics_rows = []
    labels = [title_8515, title_7030]
    for title, res in zip(labels, [res_8515, res_7030]):
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

def plot_endowment_payout_comparison(
    strategy_settings,
    label_dates=("2009-03-13", "2020-03-25"),
    add_end_payout_labels: bool = True,
):
    """
    Bars (right y): quarterly payouts for 85/15 and 70/30 (percent-of-equity rule)
    Lines (left y): cumulative dollars paid
    Labels (right side): payout amounts at the specified label_dates (snapped to that quarter)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    from matplotlib.offsetbox import HPacker
    from copy import deepcopy
    from backtest.backtest_engine import BacktestEngine

    # ---------- Data fetch ----------
    fetch_cache = {}
    df = _fetch_by_strategy_requirements(strategy_settings, fetch_cache)
    df = _ensure_datetime_index(df)

    base_engine_kwargs = deepcopy(strategy_settings.get("engine_kwargs", {}))
    payout_rate_q = float(base_engine_kwargs.get("payout_rate_quarterly", 0.0025))

    engine_cls = strategy_settings.get("engine_class", BacktestEngine)
    try:
        from backtest.endowment_payout_engine import EndowmentPayoutBacktestEngine
    except Exception:
        EndowmentPayoutBacktestEngine = None

    # Build signal frame
    if engine_cls is EndowmentPayoutBacktestEngine:
        signal_df = df.copy()
        if 'signal' not in signal_df.columns:
            signal_df['signal'] = 1.0
    else:
        strategy_class = strategy_settings["strategy_class"]
        strategy_kwargs = {k: v for k, v in strategy_settings.items()
                           if k not in ['strategy_class', 'title', 'ticker', 'engine_class', 'engine_kwargs',
                                        'iwv100_growth_overrides', 'interval', 'start', 'end', 'label']}
        strategy = strategy_class(**strategy_kwargs)
        signal_df = strategy.generate_signals(df)
        if 'signal' not in signal_df.columns:
            signal_df['signal'] = 1.0

    def _run_variant(w_iwv: float, w_agg: float):
        kw = deepcopy(base_engine_kwargs)
        kw.update({
            "w_iwv": w_iwv, "w_agg": w_agg,
            "spending_rule": "percent_of_equity",
            "payout_rate_quarterly": payout_rate_q
        })
        try:
            engine = engine_cls(**kw)
        except TypeError:
            engine = BacktestEngine()
        return engine.run_backtest(signal_df)

    res_8515 = _run_variant(0.85, 0.15)
    res_7030 = _run_variant(0.70, 0.30)

    # ---------- Payout series & cumulative ----------
    def _payout_series(res_df: pd.DataFrame) -> pd.Series:
        if 'payout' not in res_df.columns:
            return pd.Series(dtype=float)
        s = res_df.loc[res_df['payout'] > 0, 'payout'].copy()
        s.index = pd.to_datetime(s.index)
        return s

    ev_8515 = _payout_series(res_8515)
    ev_7030 = _payout_series(res_7030)

    # ---------- Plot ----------
    fig, ax_left = plt.subplots(figsize=(14, 4.2))
    ax_right = ax_left.twinx()

    # Bars (payouts, right axis)
    combined = pd.DataFrame({"85/15 %": ev_8515, "70/30 %": ev_7030}).fillna(0.0)
    if not combined.empty:
        x = mdates.date2num(np.array(combined.index.to_pydatetime()))
        period = float(np.median(np.diff(x))) if len(x) >= 2 else 90.0
        total_band = period * 0.98
        width = total_band / 2.0
        offsets = np.array([-0.5, 0.5]) * width
        colors = ['tab:orange', 'tab:blue']
        cols = combined.columns.tolist()
        for i, col in enumerate(cols):
            ax_right.bar(x + offsets[i], combined[col].values,
                         width=width, align='center', alpha=0.35,
                         edgecolor='none', color=colors[i], label=f"Payout ({col})", zorder=1)
        ymax = float(combined.values.max()) if combined.size else 0.0
        ax_right.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
        ax_right.set_ylabel("Payout per period ($)")

    # Cumulative payout lines (left axis)  ← swap colors here
    if 'cum_payout' in res_8515.columns:
        ax_left.plot(
            res_8515.index, res_8515['cum_payout'],
            linewidth=2.0, color='tab:orange', label="85/15 cumulative", zorder=3
        )
    if 'cum_payout' in res_7030.columns:
        ax_left.plot(
            res_7030.index, res_7030['cum_payout'],
            linewidth=2.0, color='tab:blue', label="70/30 cumulative", zorder=3
        )


    ax_left.set_ylabel("Cumulative ($)")
    ax_left.set_title("Payout Comparisons")
    ax_left.grid(True, linestyle='--', alpha=0.4)
    ax_left.legend(loc="upper left")

    ax_left.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_left.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    for lab in ax_left.get_xticklabels():
        lab.set_rotation(45)
        lab.set_horizontalalignment('right')

    # ---------- Equity-curve-style label helpers ----------
    def _fmt_dollar(x):
        try:
            return f"${x:,.0f}"
        except Exception:
            return f"{x:.2f}"

    def _make_label(ax, xy, text, edgecolor, xoff_pts=10, yoff_pts=0, z=7, alpha=0.95):
        price = TextArea(text, textprops=dict(fontsize=9, fontweight="bold", color=edgecolor))
        packed = HPacker(children=[price], align="center", pad=0, sep=2)
        ab = AnnotationBbox(
            packed, xy, xybox=(xoff_pts, yoff_pts),
            xycoords='data', boxcoords=("offset points"),
            box_alignment=(0.0, 0.5), frameon=True,
            bboxprops=dict(boxstyle="round,pad=0.28", fc="white", ec=edgecolor, alpha=alpha),
            zorder=z
        )
        ax.add_artist(ab)
        return ab

    def _get_xybox_pts(ab): return tuple(ab.xybox)
    def _set_xybox_pts(ab, xo, yo): ab.xybox = (float(xo), float(yo))

    # Snap a date to the payout index (on/before; else nearest after)
    def _snap_to_payout(ps: pd.Series, t: pd.Timestamp):
        ps = ps.sort_index()
        t = pd.to_datetime(t)
        if (ps.index <= t).any():
            loc = ps.index.get_indexer([t], method='pad')[0]
            if loc == -1:
                loc = 0
        else:
            loc = ps.index.get_indexer([t], method='nearest')[0]
        return ps.index[loc], float(ps.iloc[loc])

    # ---------- Build right-side labels ----------
    right_boxes = []
    # End-of-series payout labels
    if add_end_payout_labels:
        if not ev_8515.empty:
            right_boxes.append(_make_label(ax_right, (ev_8515.index[-1], float(ev_8515.iloc[-1])),
                                           _fmt_dollar(float(ev_8515.iloc[-1])), edgecolor='tab:orange',
                                           xoff_pts=10, yoff_pts=0, z=8, alpha=0.95))
        if not ev_7030.empty:
            right_boxes.append(_make_label(ax_right, (ev_7030.index[-1], float(ev_7030.iloc[-1])),
                                           _fmt_dollar(float(ev_7030.iloc[-1])), edgecolor='tab:blue',
                                           xoff_pts=10, yoff_pts=0, z=8, alpha=0.95))

    # Hardcoded dates -> payout labels (no drawdowns shown)
    label_dates = [pd.to_datetime(d) for d in label_dates]
    for t in label_dates:
        if not ev_8515.empty:
            d,p = _snap_to_payout(ev_8515, t)
            right_boxes.append(_make_label(ax_right, (d, p), _fmt_dollar(p),
                                           edgecolor='tab:orange', xoff_pts=10, yoff_pts=0, z=7, alpha=0.92))
        if not ev_7030.empty:
            d,p = _snap_to_payout(ev_7030, t)
            right_boxes.append(_make_label(ax_right, (d, p), _fmt_dollar(p),
                                           edgecolor='tab:blue', xoff_pts=10, yoff_pts=0, z=7, alpha=0.92))

    # ---- Stack labels vertically so none overlap (same as equity function) ----
    if right_boxes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        dpi = fig.dpi
        px_per_point = dpi / 72.0
        MIN_GAP_PX = 40

        info = []
        for ab in right_boxes:
            x_data, y_data = ab.xy
            x_num = mdates.date2num(pd.to_datetime(x_data))
            x_disp, y_disp = ax_right.transData.transform((x_num, y_data))
            xo_pts, yo_pts = _get_xybox_pts(ab)
            x_txt = x_disp + xo_pts * px_per_point
            y_txt = y_disp + yo_pts * px_per_point
            info.append([ab, x_disp, y_disp, x_txt, y_txt, xo_pts, yo_pts])

        info.sort(key=lambda r: -r[4])  # y_txt descending
        prev_y = None
        for ab, x_disp, y_disp, x_txt, y_txt, xo_pts, yo_pts in info:
            if prev_y is None:
                prev_y = y_txt
            else:
                target_y = min(y_txt, prev_y - MIN_GAP_PX)
                delta_px = target_y - (y_disp + yo_pts * px_per_point)
                new_yo_pts = yo_pts + (delta_px / px_per_point)
                _set_xybox_pts(ab, xo_pts, new_yo_pts)
                prev_y = y_disp + new_yo_pts * px_per_point

        # keep labels within vertical bounds of right axis
        fig.canvas.draw()
        ax_disp = ax_right.get_window_extent(renderer=renderer)
        for ab in right_boxes:
            bb = ab.get_window_extent(renderer=renderer)
            xo_pts, yo_pts = _get_xybox_pts(ab)
            adjust = 0.0
            if bb.y0 < ax_disp.y0:
                adjust = (ax_disp.y0 - bb.y0) / px_per_point + 2
            elif bb.y1 > ax_disp.y1:
                adjust = - (bb.y1 - ax_disp.y1) / px_per_point - 2
            if adjust:
                _set_xybox_pts(ab, xo_pts, yo_pts + adjust)
        fig.canvas.draw()

    fig.tight_layout()
    return fig, (ax_left, ax_right)
