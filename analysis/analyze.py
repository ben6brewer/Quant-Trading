# analysis/analyze.py

from config.secrets_config import *
from config.universe_config import *
from visualizations.plot_signals import *
from visualizations.plot_equity_curve import *
from backtest.backtest_engine import *
from utils.pretty_print_df import *
from backtest.performance_metrics import *
from utils.data_fetch import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # date axis formatting
from copy import deepcopy

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


def _ensure_datetime_index(df: pd.DataFrame, label_for_logs: str = "") -> pd.DataFrame:
    """Make sure the DataFrame has a DatetimeIndex."""
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"⚠️ Could not convert index to datetime{(' for ' + label_for_logs) if label_for_logs else ''}: {e}")
    return df


def _needs_iwv_agg_join(settings: dict) -> bool:
    """
    Determine whether the strategy requires an IWV+AGG joined dataframe.
    We use this for the endowment spending engine.
    """
    engine_cls = settings.get("engine_class", BacktestEngine)
    if EndowmentPayoutBacktestEngine is not None and engine_cls is EndowmentPayoutBacktestEngine:
        return True
    return settings is globals().get("UNIVERSITY_ENDOWMENT_SPENDING_STRATEGY_SETTINGS", object())


def _fetch_iwv_agg_join() -> pd.DataFrame:
    """
    Fetch IWV and AGG raw OHLCV data and join them on DatetimeIndex.
    Kept independent of other settings to avoid circular imports / missing params.
    """
    IWV_FETCH_SETTINGS = {"title": "IWV", "ticker": "IWV",
                            # "period": "max",
                           "start": "2015-01-01",
                             "interval": "1d", "type": "equity"}
    AGG_FETCH_SETTINGS = {"title": "AGG", "ticker": "AGG",
                           # "period": "max",
                           "start": "2015-01-01",
                             "interval": "1d", "type": "equity"}

    iwv_df = fetch_data_for_strategy(IWV_FETCH_SETTINGS)
    agg_df = fetch_data_for_strategy(AGG_FETCH_SETTINGS)

    iwv_df = _ensure_datetime_index(iwv_df, "IWV")
    agg_df = _ensure_datetime_index(agg_df, "AGG")

    joined = iwv_df.join(
        agg_df,
        how="inner",
        lsuffix="_IWV",
        rsuffix="_AGG"
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


def compare_strategies(strategy_settings_list):
    """
    (unchanged) Runs, trims, collects metrics, and plots strategies from a list of strategy settings dicts.
    """
    raw_data = []

    # Step 1: Fetch and tag data
    for settings in strategy_settings_list:
        if _needs_iwv_agg_join(settings):
            df = _fetch_iwv_agg_join()
        else:
            df = fetch_data_for_strategy(settings)

        df = _ensure_datetime_index(df)
        df.attrs['title'] = settings.get('title', 'Untitled Strategy')
        df.attrs['ticker'] = settings.get('ticker', 'Unknown')

        raw_data.append((settings, df))

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
        strategy_args = {k: v for k, v in settings.items()
                         if k not in ['strategy_class', 'title', 'ticker', 'engine_class', 'engine_kwargs']}
        strategy = strategy_class(**strategy_args)

        engine_cls = settings.get("engine_class", BacktestEngine)
        engine_kwargs = settings.get("engine_kwargs", {})
        try:
            backtester = engine_cls(**engine_kwargs)
        except TypeError:
            backtester = BacktestEngine()

        signal_df = strategy.generate_signals(df)
        result_df = backtester.run_backtest(signal_df)

        result_df.attrs['title'] = df.attrs['title']
        result_df.attrs['ticker'] = df.attrs['ticker']

        metrics = extract_performance_metrics_dict(result_df)
        metrics_list.append(metrics)
        results.append(result_df)

    # Step 4: Print metrics
    metrics_df = pd.DataFrame(metrics_list)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pretty_print_df(metrics_df)

    # Step 5: Plot equity curves
    plot_multiple_equity_curves(results)


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
    # Build the input DataFrame (IWV+AGG pair)
    if _needs_iwv_agg_join(strategy_settings):
        df = _fetch_iwv_agg_join()
    else:
        df = fetch_data_for_strategy(strategy_settings)

    # ✅ Ensure datetime index
    df = _ensure_datetime_index(df)

    # ------------------------------------------------------------
    # Instantiate strategy and generate signals once
    # ------------------------------------------------------------
    strategy_kwargs = {k: v for k, v in strategy_settings.items()
                       if k not in ['strategy_class', 'title', 'ticker', 'engine_class', 'engine_kwargs', 'iwv100_growth_overrides']}
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

        # Cumulative payout lines (left axis) — keep your linestyles
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

        # Bottom: benchmarks + four NAVs (keep linestyles as given)
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
    pd.set_option('display.float_format', '{:.2f}'.format)
    pretty_print_df(metrics_df)
