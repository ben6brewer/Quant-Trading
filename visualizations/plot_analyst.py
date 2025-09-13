import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from matplotlib.ticker import FixedLocator, FuncFormatter

def load_analyst_df(ticker: str) -> pd.DataFrame:
    """
    Load analyst review CSV for a given ticker from data/analyst_reviews.
    """
    csv_path = Path("data") / "analyst_reviews" / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Dates"])
    return df


def load_many_analyst_dfs(tickers: list[str]) -> list[pd.DataFrame]:
    """Load multiple tickers in order."""
    return [load_analyst_df(t) for t in tickers]


# ---------- Plotting ----------
def plot_analyst(
    dfs: list[pd.DataFrame],
    metric_to_plot: str,
    names: list[str] | None = None,
    date_col: str = "Dates",
    price_col: str = "Last Price",
    target_col: str = "Target Price",
    buy_col: str = "Buy %",
    hold_col: str = "Hold %",
    sell_col: str = "Sell %",
):
    """
    Plot price vs ONE chosen analyst metric across multiple tickers, with a summary slide.
    Navigation:
      - ← / → : switch between tickers; one extra "Summary" slide at the end.

    metric_to_plot can be one of:
      "Buy %"         -> normalized 0–1 buy %
      "Hold %"        -> normalized 0–1 hold %
      "Sell %"        -> normalized 0–1 sell %
      "Buy % - Sell %"      -> buy % − sell % (−1..1)
      "Target Spread" -> Target − Price ($)
      "Upside %"      -> (Target/Price − 1)
      "Price-Target"  -> Price − Target ($)  [NEW]
    """

    if not dfs:
        raise ValueError("Provide at least one DataFrame.")

    display_names = names if names is not None else [f"Series {i+1}" for i in range(len(dfs))]

    # --- Prep per-DF (normalize, derive fields)
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

        def _norm01(series: pd.Series) -> pd.Series:
            s = pd.to_numeric(series, errors="coerce")
            if s.max(skipna=True) is not None and s.max(skipna=True) > 1.001:
                s = s / 100.0
            return s.clip(0, 1)

        if buy_col in df:  df["buy01"]  = _norm01(df[buy_col])
        if hold_col in df: df["hold01"] = _norm01(df[hold_col])
        if sell_col in df: df["sell01"] = _norm01(df[sell_col])

        if {"buy01","sell01"}.issubset(df.columns):
            df["buy_minus_sell"] = df["buy01"] - df["sell01"]

        if target_col in df and price_col in df:
            p = pd.to_numeric(df[price_col], errors="coerce")
            t = pd.to_numeric(df[target_col], errors="coerce")
            df["price_spread"]      = t - p
            df["upside_pct"]        = (t / p) - 1.0
            df["price_minus_target"] = p - t  # NEW

        return df

    prepared = [prepare(df) for df in dfs]

    # Map user-friendly names to columns
    metric_map = {
        "Buy %":         ("Buy %", "buy01", True),
        "Hold %":        ("Hold %", "hold01", True),
        "Sell %":        ("Sell %", "sell01", True),
        "Buy % - Sell %":      ("Buy % − Sell %", "buy_minus_sell", False),
        "Target Spread": ("Target − Price", "price_spread", False),
        "Upside %":      ("Upside %", "upside_pct", False),
        "Price-Target":  ("Price − Target", "price_minus_target", False),  # NEW
    }

    if metric_to_plot not in metric_map:
        raise ValueError(f"Unknown metric '{metric_to_plot}'. Choose from: {list(metric_map.keys())}")

    label, col, is_01 = metric_map[metric_to_plot]

    # Precompute correlations for summary slide: corr(metric, price)
    corr_rows = []
    for name, df in zip(display_names, prepared):
        if col in df.columns and price_col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[price_col], errors="coerce")
            aligned = pd.concat([x, y], axis=1).dropna()
            if len(aligned) >= 2:
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            else:
                corr = np.nan
        else:
            corr = np.nan
        corr_rows.append((name, corr))
    corr_df = pd.DataFrame(corr_rows, columns=["Ticker", "Correlation"])

    # Add one extra "slide" for summary
    SUMMARY_INDEX = len(prepared)

    state = {"page_idx": 0}  # 0..len(dfs)-1 are tickers, len(dfs) is summary

    fig, ax_price = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax_metric = ax_price.twinx()

    def draw_summary():
        ax_price.clear(); ax_metric.clear()
        ax_metric.set_visible(False)  # hide right axis for the table slide

        ax_price.set_title(f"Correlation: {label} vs Price", fontsize=15)
        ax_price.set_axis_off()

        # Build table data
        table_data = [["Ticker", "Corr"]]
        for _, row in corr_df.iterrows():
            val = row["Correlation"]
            txt = "n/a" if pd.isna(val) else f"{val:0.3f}"
            table_data.append([row["Ticker"], txt])

        # Render centered table
        table = ax_price.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.4)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    def draw_ticker(idx: int):
        ax_price.clear(); ax_metric.clear()
        ax_metric.set_visible(True)

        df = prepared[idx]
        name = display_names[idx]

        # Title
        ax_price.set_title(f"Analyst Metric: {label} vs {name}", fontsize=14)

        # Price (left axis, log, black)
        ax_price.plot(df[date_col], df[price_col], color="black", label=f"{name} Price")
        ax_price.set_yscale("log")
        ax_price.set_xlabel("Date")
        ax_price.set_ylabel("Price ($)", color="black")
        ax_price.tick_params(axis="y", colors="black")
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Custom ticks with $ and cents, no extra minor ticks
        ax_price.relim(); ax_price.autoscale_view()
        ymin, ymax = ax_price.get_ylim()
        ymin = max(ymin, 1e-12)
        fig_h_in = fig.get_size_inches()[1]
        n_ticks = max(2, int(fig_h_in / 0.5) + 1)
        ticks = np.exp(np.linspace(np.log(ymin), np.log(ymax), n_ticks))
        ax_price.yaxis.set_major_locator(FixedLocator(ticks))
        ax_price.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"${v:,.2f}"))
        ax_price.yaxis.set_minor_locator(FixedLocator([]))

        # Analyst metric (right axis, blue)
        if col not in df.columns:
            ax_metric.text(0.5, 0.5, f"Metric '{label}' not available for {name}",
                           transform=ax_metric.transAxes, ha="center", va="center",
                           bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))
        else:
            ax_metric.plot(df[date_col], df[col], color="blue", label=label)
            ax_metric.set_ylabel(label, rotation=270, labelpad=15, color="black")
            ax_metric.yaxis.set_label_position("right")
            ax_metric.tick_params(axis="y", colors="black")
            if is_01:
                ax_metric.set_ylim(0, 1)

            # Align x-range to available metric dates
            valid = df.loc[df[col].notna(), date_col]
            if not valid.empty:
                ax_price.set_xlim(valid.min(), valid.max())

        # Legend
        l1, lb1 = ax_price.get_legend_handles_labels()
        l2, lb2 = ax_metric.get_legend_handles_labels()
        if l1 or l2:
            ax_price.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=9, bbox_to_anchor=(0,1))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    def redraw():
        if state["page_idx"] == SUMMARY_INDEX:
            draw_summary()
        else:
            draw_ticker(state["page_idx"])

    def on_key(event):
        if event.key == "right":
            state["page_idx"] = (state["page_idx"] + 1) % (len(prepared) + 1)
            redraw()
        elif event.key == "left":
            state["page_idx"] = (state["page_idx"] - 1) % (len(prepared) + 1)
            redraw()

    redraw()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

# ---- Utility: enrich DF with derived columns (same logic you’ve used) ----
def _prepare_analyst_df(df,
                        date_col="Dates",
                        price_col="Last Price",
                        target_col="Target Price",
                        buy_col="Buy %",
                        hold_col="Hold %",
                        sell_col="Sell %"):
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)

    def _norm01(series):
        s = pd.to_numeric(series, errors="coerce")
        if s.max(skipna=True) is not None and s.max(skipna=True) > 1.001:
            s = s / 100.0
        return s.clip(0, 1)

    if buy_col in df:  df["buy01"]  = _norm01(df[buy_col])
    if hold_col in df: df["hold01"] = _norm01(df[hold_col])
    if sell_col in df: df["sell01"] = _norm01(df[sell_col])

    if {"buy01","sell01"}.issubset(df.columns):
        df["buy_minus_sell"] = df["buy01"] - df["sell01"]

    if target_col in df and price_col in df:
        p = pd.to_numeric(df[price_col], errors="coerce")
        t = pd.to_numeric(df[target_col], errors="coerce")
        df["price_spread"]       = t - p
        df["upside_pct"]         = (t / p) - 1.0
        df["price_minus_target"] = p - t

    return df


import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator, FuncFormatter

def plot_analyst_color_coded(
    dfs: list[pd.DataFrame],
    metric_to_plot: str,
    names: list[str] | None = None,
    date_col: str = "Dates",
    price_col: str = "Last Price",
    target_col: str = "Target Price",
    buy_col: str = "Buy %",
    hold_col: str = "Hold %",
    sell_col: str = "Sell %",
):
    """
    Color-coded price plot for ONE chosen analyst metric across multiple tickers.
    Navigation:
      - ← / → : switch between tickers

    metric_to_plot can be one of:
      "Buy %"         -> normalized 0–1 buy %
      "Hold %"        -> normalized 0–1 hold %
      "Sell %"        -> normalized 0–1 sell %
      "Buy % - Sell %"      -> buy % − sell % (−1..1)
      "Target Spread" -> Target − Price ($)
      "Upside %"      -> (Target/Price − 1)
      "Price-Target"  -> Price − Target ($)
    """

    if not dfs:
        raise ValueError("Provide at least one DataFrame.")

    display_names = names if names is not None else [f"Series {i+1}" for i in range(len(dfs))]

    # --- prep/derive the same way as plot_analyst ---
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

        def _norm01(series: pd.Series) -> pd.Series:
            s = pd.to_numeric(series, errors="coerce")
            if s.max(skipna=True) is not None and s.max(skipna=True) > 1.001:
                s = s / 100.0
            return s.clip(0, 1)

        if buy_col in df:  df["buy01"]  = _norm01(df[buy_col])
        if hold_col in df: df["hold01"] = _norm01(df[hold_col])
        if sell_col in df: df["sell01"] = _norm01(df[sell_col])

        if {"buy01","sell01"}.issubset(df.columns):
            df["buy_minus_sell"] = df["buy01"] - df["sell01"]

        if target_col in df and price_col in df:
            p = pd.to_numeric(df[price_col], errors="coerce")
            t = pd.to_numeric(df[target_col], errors="coerce")
            df["price_spread"]       = t - p
            df["upside_pct"]         = (t / p) - 1.0
            df["price_minus_target"] = p - t

        return df

    prepared = [_prepare(df) for df in dfs]

    # map the same user-facing metric strings to (label, column, is01)
    metric_map = {
        "Buy %":         ("Buy %", "buy01", True),
        "Hold %":        ("Hold %", "hold01", True),
        "Sell %":        ("Sell %", "sell01", True),
        "Buy % - Sell %":      ("Buy − Sell", "buy_minus_sell", False),
        "Target Spread": ("Target − Price", "price_spread", False),
        "Upside %":      ("Upside %", "upside_pct", False),
        "Price-Target":  ("Price − Target", "price_minus_target", False),
    }
    if metric_to_plot not in metric_map:
        raise ValueError(f"Unknown metric '{metric_to_plot}'. Choose from: {list(metric_map.keys())}")

    label, col, is_01 = metric_map[metric_to_plot]

    # normalization to [0,1] FOR COLORING ONLY (display stays in native units)
    def _norm_for_color(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if is_01:
            return s.clip(0, 1)
        if col == "buy_minus_sell":
            return ((s + 1.0) / 2.0).clip(0, 1)
        # generic min-max
        smin = s.min(skipna=True)
        smax = s.max(skipna=True)
        if pd.isna(smin) or pd.isna(smax) or smax == smin:
            return pd.Series(np.nan, index=s.index)
        return (s - smin) / (smax - smin)

    # figure with separate colorbar axis
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    cbar_ax.set_facecolor('white')

    cmap = plt.get_cmap('turbo')
    state = {"idx": 0}

    def _format_price_axis(ax_local):
        ax_local.set_yscale('log')
        ax_local.tick_params(axis='y', colors='black')
        # remove unlabeled minor ticks
        ax_local.yaxis.set_minor_locator(FixedLocator([]))
        # custom major ticks in dollars with cents
        ax_local.relim(); ax_local.autoscale_view()
        ymin, ymax = ax_local.get_ylim()
        ymin = max(ymin, 1e-12)
        fig_h_in = fig.get_size_inches()[1]
        n_ticks = max(2, int(fig_h_in / 0.5) + 1)
        ticks = np.exp(np.linspace(np.log(ymin), np.log(ymax), n_ticks))
        ax_local.yaxis.set_major_locator(FixedLocator(ticks))
        ax_local.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"${v:,.2f}"))

    def draw():
        ax.clear(); cbar_ax.clear()

        df = prepared[state["idx"]]
        name = display_names[state["idx"]]

        ax.set_title(f"{label} vs {name}", fontsize=14)
        ax.set_xlabel('Date', color='black')
        ax.set_ylabel('Price ($)', color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')

        # normalized values for coloring
        if col not in df.columns or not df[col].notna().any():
            ax.text(0.5, 0.5, f"Metric '{label}' not available for {name}",
                    transform=ax.transAxes, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))
            plt.draw()
            return

        norm_vals = _norm_for_color(df[col])

        # draw colored segments
        for i in range(len(df) - 1):
            x = [df[date_col].iloc[i], df[date_col].iloc[i + 1]]
            y = [df[price_col].iloc[i], df[price_col].iloc[i + 1]]
            v = norm_vals.iloc[i]
            if pd.notna(v) and pd.notna(y[0]) and pd.notna(y[1]):
                ax.plot(x, y, color=cmap(v), linewidth=2)

        # zoom x to metric-valid region
        valid = df.loc[norm_vals.notna(), date_col]
        if not valid.empty:
            ax.set_xlim(valid.min(), valid.max())
        else:
            ax.set_xlim(df[date_col].min(), df[date_col].max())

        # axis formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        _format_price_axis(ax)

        # colorbar 0..1 (always normalized display)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"{label} (normalized 0–1)", color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(cbar.ax.get_yticklabels(), color='black')

        fig.subplots_adjust(bottom=0.18)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["idx"] = (state["idx"] + 1) % len(prepared)
            draw()
        elif event.key == "left":
            state["idx"] = (state["idx"] - 1) % len(prepared)
            draw()

    draw()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
