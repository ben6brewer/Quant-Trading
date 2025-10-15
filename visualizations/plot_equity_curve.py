# utils/plot_equity_curve.py

import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# Metrics helpers for the frontier
from backtest.performance_metrics import (
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

# --------------------------- Helpers ---------------------------

def _display_label(df):
    """
    Prefer a human-readable title for legends/labels; fall back gracefully.
    Using 'title' first ensures we show full 3-way mixes like '60/25/15 IWV/VXUS/AGG'.
    """
    return (
        df.attrs.get('title')
        or df.attrs.get('label')
        or df.attrs.get('ticker')
        or 'Strategy'
    )

# --------------------------- Single-curve plots ---------------------------

def plot_equity_curve(results_df, ax, log_scale: bool = False):
    """Plot one strategy’s equity curve (optionally log scale) on the given Axes."""
    if 'total_equity' not in results_df.columns:
        raise ValueError("DataFrame must contain a 'total_equity' column.")

    label = _display_label(results_df)
    ax.plot(results_df.index, results_df['total_equity'], label=label, linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Equity ($)")
    ax.set_title(f"Equity Curve")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_yscale("log" if log_scale else "linear")
    ax.legend(loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


def plot_equity_vs_benchmark(results_df, ax, log_scale: bool = False):
    """Plot a single equity curve vs its benchmark (optional log scale) on the given Axes."""
    if 'total_equity' not in results_df.columns or 'close' not in results_df.columns:
        raise ValueError("DataFrame must contain both 'total_equity' and 'close' columns.")

    close_prices = results_df['close'].replace(',', '', regex=True).astype(float)
    initial_equity = float(results_df['total_equity'].iloc[0])
    initial_price = float(close_prices.iloc[0])
    buy_and_hold_equity = close_prices / initial_price * initial_equity

    label = _display_label(results_df)
    ax.plot(results_df.index, results_df['total_equity'], label=f"{label} (Strategy)", linewidth=2)
    ax.plot(results_df.index, buy_and_hold_equity, label=f"{label} (Benchmark)", linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.set_title(f"{label} — Strategy vs Benchmark{' (Log Scale)' if log_scale else ''}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_yscale("log" if log_scale else "linear")
    ax.legend(loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# --------------------------- Multi-curve comparison ---------------------------
def plot_multiple_equity_curves(
    results_dfs,
    normalize: bool = True,
    log_scale: bool = False,
    ax=None,
    add_end_price_labels: bool = True,
    dd_labels: bool = True,
    dd_threshold: float = 0.25,          # 25% drawdown threshold
    dd_max_labels_per_line: int | None = None
):
    """
    Plot multiple strategies’ equity curves (optional log scale).

    Adds:
      • End-of-line price labels (boxed/bold, color-matched) placed to the RIGHT
        of the last datapoint and vertically stacked with a fixed gap.
      • Drawdown trough labels (one per valley ≥ dd_threshold below ATH) placed to the RIGHT
        of their troughs and included in the SAME right-side vertical stack.
      • Drawdown labels show:  $PRICE  -xx.xx%  (percent in red).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

    if not results_dfs:
        print("⚠️ No results to plot.")
        return None, None

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig = ax.figure

    # -------- Align by latest common start date --------
    latest_start = max(df.index.min() for df in results_dfs)
    aligned_dfs = []
    for df in results_dfs:
        trimmed_df = df[df.index >= latest_start].copy()
        trimmed_df.attrs.update(df.attrs)
        if normalize:
            start_val = float(trimmed_df['total_equity'].iloc[0])
            trimmed_df['normalized_equity'] = trimmed_df['total_equity'] / start_val * 1_000_000
        else:
            trimmed_df['normalized_equity'] = trimmed_df['total_equity']
        aligned_dfs.append(trimmed_df)

    ax.cla()
    lines = []
    for df in aligned_dfs:
        label = _display_label(df)
        line, = ax.plot(df.index, df['normalized_equity'], label=label, linewidth=2)
        lines.append((line, df))

    ax.set_title("Equity Curve Comparison", fontsize=18)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Total Equity ($)" if not normalize else "Total Equity (normalized $)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_yscale("log" if log_scale else "linear")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # -------- Helpers --------
    def _fmt_dollar(x):
        try:
            return f"${x:,.0f}"
        except Exception:
            return f"{x:.2f}"

    def _drawdown_valley_troughs(series: np.ndarray, threshold: float) -> list[int]:
        """Return one trough index per valley >= threshold below ATH."""
        if series.size == 0:
            return []
        trough_idxs = []
        run_max = float(series[0])
        in_valley = False
        valley_min_val, valley_min_idx = None, None

        for t, val in enumerate(series):
            if val > run_max + 1e-12:
                if in_valley and valley_min_idx is not None:
                    trough_idxs.append(valley_min_idx)
                run_max = val
                in_valley = False
                valley_min_val, valley_min_idx = None, None
                continue

            dd = val / run_max - 1.0
            if dd <= -threshold:
                if not in_valley:
                    in_valley = True
                    valley_min_val, valley_min_idx = val, t
                elif val < valley_min_val:
                    valley_min_val, valley_min_idx = val, t

        if in_valley and valley_min_idx is not None:
            trough_idxs.append(valley_min_idx)
        return trough_idxs

    def _make_dualcolor_label(ax, xy, price_text, edgecolor, pct_text=None,
                              xoff_pts=10, yoff_pts=0, z=7, alpha=0.95):
        """
        Build a rounded box label to the right of xy.
        If pct_text is provided, it will appear in red after the price.
        """
        price = TextArea(price_text, textprops=dict(
            fontsize=9, fontweight="bold", color=edgecolor))
        if pct_text is not None:
            pct = TextArea(f" {pct_text}", textprops=dict(
                fontsize=9, fontweight="bold", color="red"))
            packed = HPacker(children=[price, pct], align="center", pad=0, sep=2)
        else:
            packed = price

        ab = AnnotationBbox(
            packed,
            xy, xybox=(xoff_pts, yoff_pts),
            xycoords='data',
            boxcoords=("offset points"),
            box_alignment=(0.0, 0.5),  # left-center
            frameon=True,
            bboxprops=dict(boxstyle="round,pad=0.28", fc="white", ec=edgecolor, alpha=alpha),
            zorder=z
        )
        ax.add_artist(ab)
        return ab

    # Convenience: access/modify the offset (xybox) of an AnnotationBbox
    def _get_xybox_pts(ab):
        return tuple(ab.xybox)
    def _set_xybox_pts(ab, xo, yo):
        ab.xybox = (float(xo), float(yo))

    # -------- Build all RIGHT-side labels first (end + trough), then stack them together --------
    right_boxes = []

    # End-of-line labels (price only)
    if add_end_price_labels:
        for line, df in lines:
            x_end = df.index[-1]
            y_end = float(df['normalized_equity'].iloc[-1])
            color = line.get_color()
            ab = _make_dualcolor_label(ax, (x_end, y_end), _fmt_dollar(y_end), edgecolor=color,
                                       pct_text=None, xoff_pts=10, yoff_pts=0, z=8, alpha=0.95)
            right_boxes.append(ab)

    # Drawdown trough labels (price + red percent)
    if dd_labels and dd_threshold > 0:
        for line, df in lines:
            series = np.asarray(df['normalized_equity'], dtype=float)
            run_max = np.maximum.accumulate(series)
            dd = series / run_max - 1.0
            trough_idxs = _drawdown_valley_troughs(series, dd_threshold)
            if dd_max_labels_per_line and len(trough_idxs) > dd_max_labels_per_line:
                trough_idxs = sorted(trough_idxs, key=lambda i: dd[i])[:dd_max_labels_per_line]

            color = line.get_color()
            for i in trough_idxs:
                x_i = df.index[i]
                y_i = float(series[i])
                pct_text = f"{dd[i]*100:.2f}%"
                ab = _make_dualcolor_label(ax, (x_i, y_i), _fmt_dollar(y_i), edgecolor=color,
                                           pct_text=f"-{abs(dd[i])*100:.2f}%",
                                           xoff_pts=10, yoff_pts=0, z=7, alpha=0.92)
                right_boxes.append(ab)

    # -------- Single vertical stack for ALL right-side labels --------
    if right_boxes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        dpi = fig.dpi
        px_per_point = dpi / 72.0
        MIN_GAP_PX = 40  # a comfy vertical gap between labels

        # Build info list with display coords
        info = []
        for ab in right_boxes:
            x_data, y_data = ab.xy
            x_num = mdates.date2num(x_data)
            x_disp, y_disp = ax.transData.transform((x_num, y_data))
            xo_pts, yo_pts = _get_xybox_pts(ab)
            x_txt = x_disp + xo_pts * px_per_point
            y_txt = y_disp + yo_pts * px_per_point
            info.append([ab, x_disp, y_disp, x_txt, y_txt, xo_pts, yo_pts])

        # Sort top→bottom and enforce spacing
        info.sort(key=lambda r: -r[4])  # y_txt descending
        prev_y = None
        for row in info:
            ab, x_disp, y_disp, x_txt, y_txt, xo_pts, yo_pts = row
            if prev_y is None:
                prev_y = y_txt
            else:
                target_y = min(y_txt, prev_y - MIN_GAP_PX)
                delta_px = target_y - (y_disp + yo_pts * px_per_point)
                new_yo_pts = yo_pts + (delta_px / px_per_point)
                _set_xybox_pts(ab, xo_pts, new_yo_pts)
                y_txt = y_disp + new_yo_pts * px_per_point
                prev_y = y_txt

        # Keep within vertical bounds of axes
        fig.canvas.draw()
        ax_disp = ax.get_window_extent(renderer=renderer)
        for ab in right_boxes:
            bb = ab.get_window_extent(renderer=renderer).expanded(1.0, 1.0)
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
    return fig, ax


def _parse_equity_pct(label: str) -> float | None:
    """
    Extract total equity percent from labels. Supports:
      - '80/20 IWV/AGG'           -> equity = 80
      - '60/25/15 IWV/VXUS/AGG'   -> equity = 60 + 25 = 85
      - '60/25/15'                -> equity = 60 + 25
    Returns 0..100 or None if not parsed.
    """
    if not isinstance(label, str):
        return None

    # Try to capture up to three leading numeric parts (before tickers)
    m = re.match(r'\s*(\d{1,3})(?:\s*/\s*(\d{1,3}))?(?:\s*/\s*(\d{1,3}))?', label)
    if not m:
        return None

    parts = [p for p in m.groups() if p is not None]
    try:
        nums = [int(x) for x in parts]
    except Exception:
        return None

    if any(n < 0 or n > 100 for n in nums):
        return None

    if len(nums) == 2:
        # equity/bonds pattern
        return float(nums[0])

    if len(nums) >= 3:
        # iwv / vxus / agg pattern => equity = iwv + vxus
        iwv, vxus, agg = nums[0], nums[1], nums[2]
        total = iwv + vxus + agg
        # Be tolerant of small rounding mismatches
        if abs(total - 100) > 1:
            return None
        return float(iwv + vxus)

    return None

# --------------------------- Efficient Frontier ---------------------------

def plot_efficient_frontier(results_dfs, ax, risk_free_rate: float = 0.04):
    """
    Scatter of CAGR (%) vs Volatility (%) for each results_df.

    Special roles:
      - Max Sharpe   (orange star)
      - Max Sortino  (green diamond)
      - Max CAGR     (purple triangle up)
      - Min Variance (red triangle down)

    If multiple roles fall on the same portfolio (within tolerance), they are
    grouped into a *single* symbol and combined legend entry. The base blue dot
    for any special point is not drawn (to avoid duplicates).
    """
    import numpy as np
    from matplotlib.lines import Line2D

    if not results_dfs:
        ax.text(0.5, 0.5, "No results", ha='center', va='center', transform=ax.transAxes)
        return

    # ---- Build data points ----
    points = []
    for df in results_dfs:
        label = (df.attrs.get('title')
                 or df.attrs.get('label')
                 or df.attrs.get('ticker')
                 or 'Strategy')
        try:
            cagr    = float(calculate_cagr(df) * 100.0)
            vol     = float(calculate_volatility(df) * 100.0)
            sharpe  = float(calculate_sharpe_ratio(df, risk_free_rate=risk_free_rate))
            sortino = float(calculate_sortino_ratio(df, risk_free_rate=risk_free_rate))
        except Exception:
            continue

        points.append({"label": label, "cagr": cagr, "vol": vol,
                       "sharpe": sharpe, "sortino": sortino})

    if not points:
        ax.text(0.5, 0.5, "No metrics computed", ha='center', va='center', transform=ax.transAxes)
        return

    # ---- Identify extrema with tolerance and tie-breaking by lowest vol ----
    EPS = 1e-12
    max_sharpe   = max(p["sharpe"]  for p in points)
    max_sortino  = max(p["sortino"] for p in points)
    max_cagr     = max(p["cagr"]    for p in points)
    min_variance = min(p["vol"]     for p in points)

    def argbest(cands, key):
        return min(cands, key=lambda i: points[i]["vol"])

    sharpe_cands  = [i for i, p in enumerate(points) if abs(p["sharpe"]  - max_sharpe)  < EPS]
    sortino_cands = [i for i, p in enumerate(points) if abs(p["sortino"] - max_sortino) < EPS]
    cagr_cands    = [i for i, p in enumerate(points) if abs(p["cagr"]    - max_cagr)    < EPS]
    var_cands     = [i for i, p in enumerate(points) if abs(p["vol"]     - min_variance) < EPS]

    idx_best_sharpe   = argbest(sharpe_cands,  "sharpe")
    idx_best_sortino  = argbest(sortino_cands, "sortino")
    idx_max_cagr      = argbest(cagr_cands,    "cagr")
    idx_min_variance  = argbest(var_cands,     "vol")

    # ---- Group roles by index (so overlapping winners get a single marker) ----
    role_map = {}  # idx -> set of roles
    def add_role(i, role):
        role_map.setdefault(i, set()).add(role)

    add_role(idx_best_sharpe,   "Max Sharpe")
    add_role(idx_best_sortino,  "Max Sortino")
    add_role(idx_max_cagr,      "Max CAGR")
    add_role(idx_min_variance,  "Min Variance")

    special_indices = set(role_map.keys())

    # ---- Draw base scatter for *non-special* points only ----
    ax.cla()
    xs = [p["vol"]  for i, p in enumerate(points) if i not in special_indices]
    ys = [p["cagr"] for i, p in enumerate(points) if i not in special_indices]
    if xs and ys:
        ax.scatter(xs, ys, s=45, color='tab:blue', alpha=0.9, zorder=2)

    # ---- Decide marker style for grouped roles (priority order) ----
    # If a point has multiple roles, pick one symbol using this priority:
    #   Max Sharpe > Max Sortino > Max CAGR > Min Variance
    STYLE = {
        "Max Sharpe":   dict(marker='*', color='orange', size=300),
        "Max Sortino":  dict(marker='D', color='green',  size=120),
        "Max CAGR":     dict(marker='^', color='purple', size=120),
        "Min Variance": dict(marker='v', color='red',    size=120),
    }
    PRIORITY = ["Max Sharpe", "Max Sortino", "Max CAGR", "Min Variance"]

    legend_handles = []
    for idx in sorted(special_indices):
        roles = role_map[idx]
        # Build combined legend text
        role_text = " & ".join([r for r in PRIORITY if r in roles])
        text = f"{role_text}: {points[idx]['label']}"

        # Choose a single visual style based on priority
        for r in PRIORITY:
            if r in roles:
                sty = STYLE[r]
                break

        pt = points[idx]
        sc = ax.scatter([pt["vol"]], [pt["cagr"]],
                        s=sty["size"], marker=sty["marker"],
                        color=sty["color"], zorder=5, edgecolors='none')

        # Legend handle (so we can show each grouped special point explicitly)
        legend_handles.append(
            Line2D([0], [0], marker=sty["marker"], linewidth=0, markersize=np.sqrt(sty["size"]),
                   color=sty["color"], label=text)
        )

    # ---- Axes & legend ----
    ax.set_title(f"Efficient Frontier (risk free rate: {risk_free_rate*100:.1f}%)", fontsize=16)
    ax.set_xlabel("Volatility (annualized, %)", fontsize=12)
    ax.set_ylabel("CAGR (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Dynamic padding to keep everything visible
    all_x = [p["vol"] for p in points]
    all_y = [p["cagr"] for p in points]
    if all_x:
        xr = max(all_x) - min(all_x)
        ax.set_xlim(min(all_x) - 0.1 * xr, max(all_x) + 0.15 * xr)
    if all_y:
        yr = max(all_y) - min(all_y)
        ax.set_ylim(min(all_y) - 0.1 * yr, max(all_y) + 0.2 * yr)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', frameon=True)

# --------------------------- Grid search (draws on its own fig) ---------------------------

def plot_grid_search_equity_curves(results_dfs, best_params, benchmark_df=None):
    """Plot grid search equity curves, highlight best."""
    print("Loading grid search results")
    if len(results_dfs) == 0:
        print("⚠️ No results to plot.")
        return

    # Align all DataFrames to the latest common start date
    latest_start = max(df.index.min() for df in results_dfs)
    aligned_dfs = []
    for df in results_dfs:
        trimmed_df = df[df.index >= latest_start].copy()
        trimmed_df.attrs.update(df.attrs)
        aligned_dfs.append(trimmed_df)

    if benchmark_df is not None:
        benchmark_df = benchmark_df[benchmark_df.index >= latest_start].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    # Plot all non-best strategies
    for df in aligned_dfs:
        params = df.attrs.get('params')
        if params != best_params:
            ax.plot(df.index, df['total_equity'],
                    linewidth=0.5, alpha=1, color=next(color_cycle), zorder=1)

    # Highlight best-performing strategy
    for df in aligned_dfs:
        params = df.attrs.get('params')
        if params == best_params:
            parts = [f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" for k, v in params.items()]
            label = ", ".join(parts) if parts else "Best"
            ax.plot(df.index, df['total_equity'], label=label,
                    linewidth=2.5, color='lawngreen', zorder=3)

    if benchmark_df is not None:
        bench_label = _display_label(results_dfs[0]) + " (Benchmark)"
        ax.plot(benchmark_df.index, benchmark_df['total_equity'],
                label=bench_label, linewidth=2, color='blue', zorder=4)

    title_root = results_dfs[0].attrs.get('title') or _display_label(results_dfs[0])
    ax.set_title(f"Equity Curve Comparison — {title_root}", fontsize=18)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Total Equity ($)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    plt.show()

# --------------------------- Interactive tab navigation ---------------------------

def plot_equity_tab(fig, results_dfs, normalize: bool = True, risk_free_rate: float = 0.04):
    """
    Interactive tab: use left/right arrow keys to toggle between:
      1) Equity Curve Comparison (Linear)
      2) Equity Curve Comparison (Log Scale)
      3) Efficient Frontier (CAGR vs Volatility)

    risk_free_rate is forwarded to the frontier so it matches whatever you used for the table.
    """
    fig.clear()
    gs = gridspec.GridSpec(1, 1, hspace=0.1, wspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0])

    tabs = [
        ("Equity Curve Comparison (Linear)", False, "curves"),
        ("Equity Curve Comparison (Log Scale)", True, "curves"),
        ("Efficient Frontier (CAGR vs Volatility)", None, "frontier"),
    ]
    idx = [0]

    def draw():
        title, log_flag, kind = tabs[idx[0]]
        if kind == "curves":
            plot_multiple_equity_curves(results_dfs, normalize=normalize, log_scale=bool(log_flag), ax=ax)
        else:
            # Efficient frontier — pass the SAME rf that the table used
            plot_efficient_frontier(results_dfs, ax=ax, risk_free_rate=risk_free_rate)

        # update window title if backend supports it
        mgr = getattr(fig.canvas, "manager", None)
        if mgr and hasattr(mgr, "set_window_title"):
            mgr.set_window_title(title)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(tabs)
            draw()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(tabs)
            draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show(block=True)
