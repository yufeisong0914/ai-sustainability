import csv as csv_module
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation import (
    estimate_footprint_interval,
    LOCATION_PARAMS,
    Interval,
    ALPHA_TABLE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config & constants
# ──────────────────────────────────────────────────────────────────────────────

def _load_baseline_params(csv_path: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["name"]: row for row in csv_module.DictReader(f)}

_BASELINE_PARAMS = _load_baseline_params("config/baseline_params.csv")
_GOOGLE = _BASELINE_PARAMS["google-search"]
GOOGLE_WH_PER_QUERY = float(_GOOGLE["energy_wh_per_query"])
GOOGLE_LABEL = _GOOGLE["label"]
GOOGLE_SOURCE = _GOOGLE["source"]
GOOGLE_YEAR   = _GOOGLE["year"]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
GOOGLE_COLOR = "#8172B2"

st.set_page_config(page_title="LLM Footprint Comparator", layout="wide")
st.title("🌍 LLM Footprint Comparator Dashboard")
st.caption(
    f"Google baseline: {GOOGLE_LABEL} = {GOOGLE_WH_PER_QUERY} Wh/query "
    f"({GOOGLE_SOURCE}, {GOOGLE_YEAR})"
)

models    = sorted(ALPHA_TABLE.keys())
anchors   = sorted(ALPHA_TABLE[models[0]].keys())
locations = list(LOCATION_PARAMS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def interval_from_wh(wh: Interval, location: str):
    pue = LOCATION_PARAMS[location]["pue"]
    wue = LOCATION_PARAMS[location]["wue"]
    elec  = Interval(lo=(wh.lo/1000)*pue,  mid=(wh.mid/1000)*pue,  hi=(wh.hi/1000)*pue)
    water = Interval(lo=elec.lo*wue,        mid=elec.mid*wue,        hi=elec.hi*wue)
    return elec, water

def compute_single(model, anchor, location, in_tok, out_tok):
    return estimate_footprint_interval(
        model=model, anchor=anchor,
        total_tokens=int(in_tok) + int(out_tok),
        location=location,
    )

def kwh_to_wh(iv: Interval) -> Interval:
    return Interval(lo=iv.lo*1000, mid=iv.mid*1000, hi=iv.hi*1000)

def l_to_ml(iv: Interval) -> Interval:
    return Interval(lo=iv.lo*1000, mid=iv.mid*1000, hi=iv.hi*1000)

def _sf(val):
    """Safe float conversion; returns None on failure/NaN."""
    try:
        f = float(val)
        return None if (f != f) else f   # NaN check
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# CSV processing
# ──────────────────────────────────────────────────────────────────────────────

def process_csv(df_raw: pd.DataFrame, model: str, anchor: str, location: str) -> pd.DataFrame:
    """Return per-query footprint DataFrame from uploaded CSV."""
    rows = []
    for _, row in df_raw.iterrows():
        try:
            tin  = int(float(row["token_in"]))
            tout = int(float(row["token_out"]))
        except Exception:
            continue
        r = compute_single(model, anchor, location, tin, tout)

        prompt_str = str(row.get("prompt", ""))
        rows.append({
            "ID":           row.get("ID", ""),
            "prompt":       prompt_str,
            "prompt_short": (prompt_str[:55] + "…") if len(prompt_str) > 55 else prompt_str,
            "token_in":     tin,
            "token_out":    tout,
            "token_total":  tin + tout,
            "T0":           _sf(row.get("T0 (any output)")),
            "T1":           _sf(row.get("T1 (first byte)")),
            "T2":           _sf(row.get("T2 (full-length)")),
            "citation_total": _sf(row.get("citation_total")),
            "citation_used":  _sf(row.get("citation_used")),
            "word_count_in":  _sf(row.get("word_count_in")),
            "word_count_out": _sf(row.get("word_count_out")),
            "energy_lo":   r["energy_it_wh"].lo,
            "energy_mid":  r["energy_it_wh"].mid,
            "energy_hi":   r["energy_it_wh"].hi,
            "elec_lo":     r["electricity_dc_kwh"].lo  * 1000,
            "elec_mid":    r["electricity_dc_kwh"].mid * 1000,
            "elec_hi":     r["electricity_dc_kwh"].hi  * 1000,
            "water_lo":    r["water_onsite_l"].lo  * 1000,
            "water_mid":   r["water_onsite_l"].mid * 1000,
            "water_hi":    r["water_onsite_l"].hi  * 1000,
        })
    return pd.DataFrame(rows)


def detect_tools(df: pd.DataFrame):
    """Return tool names found as paired *_in / *_out columns (old format)."""
    in_cols = {c[:-3] for c in df.columns if c.endswith("_in")}
    out_cols = {c[:-4] for c in df.columns if c.endswith("_out")}
    # Exclude scenario_N_in columns which belong to the new format
    exclude = {c.replace("_in", "") for c in df.columns if c.startswith("scenario_") and c.endswith("_in")}
    return sorted((in_cols & out_cols) - exclude)


def detect_scenarios(df: pd.DataFrame):
    """Return list of (col_prefix, display_name) from scenario_N_name/in/out columns (new format)."""
    scenarios = []
    i = 1
    while True:
        name_col = f"scenario_{i}_name"
        in_col   = f"scenario_{i}_in"
        out_col  = f"scenario_{i}_out"
        if name_col in df.columns and in_col in df.columns and out_col in df.columns:
            name = df[name_col].dropna().iloc[0] if df[name_col].notna().any() else f"Scenario {i}"
            scenarios.append((f"scenario_{i}", str(name)))
            i += 1
        else:
            break
    return scenarios


def process_wide_csv(df_raw: pd.DataFrame, prefix: str,
                     model: str, anchor: str, location: str) -> pd.DataFrame:
    """Build a narrow df for one scenario/tool from *_in/*_out columns, then compute footprint."""
    df_narrow = df_raw.copy()
    df_narrow["token_in"]  = df_raw[f"{prefix}_in"]
    df_narrow["token_out"] = df_raw[f"{prefix}_out"]
    return process_csv(df_narrow, model, anchor, location)


# ──────────────────────────────────────────────────────────────────────────────
# Shared chart utilities
# ──────────────────────────────────────────────────────────────────────────────

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linestyle="--")

def _clean_h(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linestyle="--")

# ──────────────────────────────────────────────────────────────────────────────
# Summary bar chart  (shared by both tabs)
# ──────────────────────────────────────────────────────────────────────────────
# entries: list of (label, mid, lo, hi)

def plot_summary_bars(entries, title, ylabel, note="", estimate="Mid"):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    labels = [e[0] for e in entries]
    mids   = [e[1] for e in entries]
    los    = [e[2] for e in entries]
    his    = [e[3] for e in entries]
    stds   = [e[4] if len(e) > 4 else 0 for e in entries]

    colors = PALETTE[: len(entries) - 1] + [GOOGLE_COLOR]
    xs = range(len(labels))
    has_sd = any(s > 0 for s in stds)

    vals = los if estimate == "Low" else (his if estimate == "High" else mids)
    ax.bar(xs, vals, color=colors, alpha=0.85)
    anchor = vals

    # Error bars: ±1 SD across prompts
    if has_sd:
        ax.errorbar(
            xs, anchor,
            yerr=stds,
            fmt="none", color="#222222", capsize=5, linewidth=1.4, alpha=0.8,
            label="±1 SD across prompts",
        )

    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, fontsize=7.5,
                  loc="lower center", bbox_to_anchor=(0.5, 1.08),
                  ncol=len(handles), framealpha=0.85, borderpad=0.5)

    # Label all key points on each bar
    def _lbl(x, y, txt, side="right", va="center"):
        offset = 0.42 if side == "right" else -0.42
        ha = "left" if side == "right" else "right"
        ax.text(x + offset, y, txt, ha=ha, va=va, fontsize=6.8, color="#222222")

    for i in range(len(labels)):
        sd_v = stds[i]
        _lbl(i, anchor[i], f"{anchor[i]:.4f}", "right", va="bottom")
        if sd_v > 0:
            _lbl(i, anchor[i] + sd_v, f"+SD {anchor[i] + sd_v:.4f}", "right", va="bottom")
            _lbl(i, anchor[i] - sd_v, f"−SD {anchor[i] - sd_v:.4f}", "right", va="top")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
    ax.set_ylabel(ylabel, fontsize=9)
    xlabel_parts = [f"showing {estimate.lower()} estimate"]
    if note:
        xlabel_parts.append(note)
    ax.set_xlabel("  ·  ".join(xlabel_parts), fontsize=7.5, color="#888888")
    _clean(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Breakdown chart functions
# ──────────────────────────────────────────────────────────────────────────────

def chart_energy_hist(df, name, color):
    """Histogram + KDE-style overlay of per-query IT energy."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    data = df["energy_mid"].dropna()
    n_bins = min(12, max(3, len(data)))
    ax.hist(data, bins=n_bins, color=color, alpha=0.70,
            edgecolor="white", linewidth=0.8, label="Distribution")
    mean_v = data.mean()
    std_v  = data.std()
    ax.axvline(mean_v, color="red",  linestyle="--", linewidth=1.8, label=f"Mean {mean_v:.4f}")
    ax.axvline(mean_v + std_v, color="orange", linestyle=":", linewidth=1.2, label=f"+1 SD")
    ax.axvline(mean_v - std_v, color="orange", linestyle=":", linewidth=1.2, label=f"−1 SD")
    ax.set_title(f"{name}: IT Energy per Query", fontsize=10, fontweight="bold")
    ax.set_xlabel("Energy (Wh)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=7.5)
    _clean(ax)
    plt.tight_layout()
    return fig


def chart_token_scatter(df, name, color=None):  # noqa: ARG001
    """Scatter: token_in vs token_out, bubble ∝ total, colour = energy."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    max_t = df["token_total"].max() or 1
    sc = ax.scatter(
        df["token_in"], df["token_out"],
        c=df["energy_mid"], cmap="YlOrRd",
        s=(df["token_total"] / max_t) * 350 + 25,
        alpha=0.82, edgecolors="gray", linewidth=0.4,
    )
    plt.colorbar(sc, ax=ax, label="Energy (Wh)", shrink=0.85, pad=0.02)
    ax.set_title(f"{name}: Input vs Output Tokens", fontsize=10, fontweight="bold")
    ax.set_xlabel("Input Tokens",  fontsize=9)
    ax.set_ylabel("Output Tokens", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def chart_response_times(df, name):
    """Box-plot + jitter for T0, T1, T2."""
    cols    = ["T0", "T1", "T2"]
    x_labs  = ["T0\n(first output)", "T1\n(first byte)", "T2\n(full response)"]
    data    = [df[c].dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bp = ax.boxplot(data, labels=x_labs, patch_artist=True,
                    medianprops={"color": "red", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    box_colors = ["#AEC6E8", "#FFBB78", "#98DF8A"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    np.random.seed(42)
    for i, d in enumerate(data):
        jitter = np.random.uniform(-0.13, 0.13, len(d))
        ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                   alpha=0.55, s=18, color="steelblue", zorder=5)
        # Median value label
        if len(d) > 0:
            median_v = np.median(d)
            ax.text(i + 1, median_v, f" {median_v:.2f}s",
                    ha="left", va="center", fontsize=7.5, color="red", fontweight="bold")

    ax.set_title(f"{name}: Response Time Distribution", fontsize=10, fontweight="bold")
    ax.set_ylabel("Seconds", fontsize=9)
    _clean(ax)
    plt.tight_layout()
    return fig


def chart_citation_bar(df, name):
    """Stacked bar: citations used vs unused per query."""
    valid = df[df["citation_total"].notna() & df["citation_used"].notna()].copy()
    if valid.empty:
        return None
    valid = valid.reset_index(drop=True)
    valid["unused"] = valid["citation_total"] - valid["citation_used"]
    ids = [f"Q{int(i)}" for i in valid["ID"]] if valid["ID"].notna().all() else [f"Q{i+1}" for i in valid.index]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(ids))
    ax.bar(x, valid["citation_used"], label="Used",   color="#55A868", alpha=0.85)
    ax.bar(x, valid["unused"],        label="Unused",  color="#C44E52", alpha=0.55,
           bottom=valid["citation_used"])
    # Value labels: used / total on top of each stacked bar
    for xi, (used, total) in enumerate(zip(valid["citation_used"], valid["citation_total"])):
        ax.text(xi, total + 0.15, f"{int(used)}/{int(total)}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, fontsize=8)
    ax.set_title(f"{name}: Citation Coverage per Query", fontsize=10, fontweight="bold")
    ax.set_ylabel("Citations", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    _clean(ax)
    plt.tight_layout()
    return fig


def chart_wordcount_energy(df, name, color):
    """Scatter: output word count vs energy with OLS trend line."""
    valid = df[["word_count_out", "energy_mid"]].dropna()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(valid["word_count_out"], valid["energy_mid"],
               color=color, alpha=0.78, s=65, edgecolors="gray", linewidth=0.4)
    if len(valid) > 2:
        z  = np.polyfit(valid["word_count_out"], valid["energy_mid"], 1)
        xr = np.linspace(valid["word_count_out"].min(), valid["word_count_out"].max(), 60)
        ax.plot(xr, np.poly1d(z)(xr), "r--", linewidth=1.6, alpha=0.75, label="Trend")
        r  = np.corrcoef(valid["word_count_out"], valid["energy_mid"])[0, 1]
        ax.legend(title=f"r = {r:.2f}", fontsize=8)
    ax.set_title(f"{name}: Output Words vs Energy", fontsize=10, fontweight="bold")
    ax.set_xlabel("Output Word Count", fontsize=9)
    ax.set_ylabel("Energy (Wh)",       fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


_LARGE_N = 25   # threshold above which compact chart variants kick in


def chart_perquery_bar(df, name, color, estimate="Mid"):
    """Horizontal bar (small N) or sorted dot plot (large N) of per-query IT energy."""
    val_col = {"Low": "energy_lo", "High": "energy_hi"}.get(estimate, "energy_mid")
    df_s = df.sort_values(val_col).reset_index(drop=True)
    n    = len(df_s)

    if n > _LARGE_N:
        # ── Sorted dot plot (quantile-style) ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ranks = np.arange(1, n + 1)
        ax.scatter(ranks, df_s[val_col], color=color, s=18, alpha=0.75,
                   edgecolors="none", zorder=3, label=estimate)
        ax.plot(ranks, df_s[val_col], color=color, linewidth=0.6, alpha=0.4)
        mean_v = df_s[val_col].mean()
        ax.axhline(mean_v, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_v:.5f}")
        ax.set_xlabel("Query rank (sorted ascending)", fontsize=9)
        ax.set_ylabel("IT Energy (Wh)", fontsize=9)
        ax.set_title(f"{name}: Per-Query Energy — sorted [{estimate}]",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5)
        _clean(ax)
        plt.tight_layout()
        return fig

    # ── Original horizontal bar chart (small N) ───────────────────────────────
    ids = [f"Q{int(i)}" for i in df_s["ID"]] if df_s["ID"].notna().all() \
          else [f"Q{i+1}" for i in df_s.index]
    fig, ax = plt.subplots(figsize=(5.5, max(3.2, n * 0.38)))
    bars = ax.barh(range(n), df_s[val_col], color=color, alpha=0.78)
    for bar, v in zip(bars, df_s[val_col]):
        ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.5f}", va="center", ha="left", fontsize=7.5, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_title(f"{name}: Per-Query Energy (sorted) [{estimate}]",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("IT Energy (Wh)", fontsize=9)
    _clean_h(ax)
    plt.tight_layout()
    return fig


def chart_token_hist(df, name, color=None):  # noqa: ARG001
    """Dual histogram: input vs output token counts."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    n_bins = min(12, max(3, len(df)))
    ax.hist(df["token_in"],  bins=n_bins, alpha=0.65, label="Input tokens",
            color="#4C72B0", edgecolor="white")
    ax.hist(df["token_out"], bins=n_bins, alpha=0.65, label="Output tokens",
            color="#DD8452", edgecolor="white")
    ax.set_title(f"{name}: Token Count Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Tokens", fontsize=9)
    ax.set_ylabel("Count",  fontsize=9)
    ax.legend(fontsize=8)
    _clean(ax)
    plt.tight_layout()
    return fig


def chart_cumulative_energy(df, name, color):
    """Step chart: cumulative energy across queries (in arrival order)."""
    cumsum = df["energy_mid"].cumsum().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.step(range(1, len(cumsum) + 1), cumsum, where="post",
            color=color, linewidth=2)
    ax.fill_between(range(1, len(cumsum) + 1), cumsum,
                    step="post", alpha=0.18, color=color)
    ax.set_title(f"{name}: Cumulative IT Energy", fontsize=10, fontweight="bold")
    ax.set_xlabel("Query #", fontsize=9)
    ax.set_ylabel("Cumulative Energy (Wh)", fontsize=9)
    _clean(ax)
    plt.tight_layout()
    return fig


def chart_three_distributions(df, name, color, estimate="Mid"):
    """3 panels (IT energy, DC electricity, water).
    Small N (≤25): one bar per query.  Large N: line + shaded uncertainty band.
    """
    metrics = [
        ("energy_mid", "energy_lo", "energy_hi", "IT Energy (Wh)"),
        ("elec_mid",   "elec_lo",   "elec_hi",   "DC Electricity (Wh)"),
        ("water_mid",  "water_lo",  "water_hi",  "Cooling Water (mL)"),
    ]
    col_idx = {"Low": 1, "Mid": 0, "High": 2}[estimate]
    n = len(df)

    if n > _LARGE_N:
        # ── Line + shaded band (large N) ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        xs = np.arange(1, n + 1)
        for ax, (mid_col, lo_col, hi_col, label) in zip(axes, metrics):
            val_col = [mid_col, lo_col, hi_col][col_idx]
            ax.plot(xs, df[val_col].values, color=color, linewidth=1.4,
                    alpha=0.85, label=estimate)
            mean_v = df[val_col].mean()
            ax.axhline(mean_v, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean = {mean_v:.5f}")
            ax.set_xlabel("Query #", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7.5, loc="upper left")
            _clean(ax)
        fig.suptitle(f"{name}: Per-Query Values [{estimate}]",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        return fig

    # ── Bar chart (small N) ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ids = [f"Q{int(i)}" for i in df["ID"]] if df["ID"].notna().all() \
          else [f"Q{i+1}" for i in df.index]

    for ax, (mid_col, lo_col, hi_col, label) in zip(axes, metrics):
        val_col = [mid_col, lo_col, hi_col][col_idx]
        x = np.arange(n)
        bars = ax.bar(x, df[val_col], color=color, alpha=0.75,
                      edgecolor="white", linewidth=0.8)
        mean_v = df[val_col].mean()
        ax.axhline(mean_v, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_v:.5f}")
        for bar, v in zip(bars, df[val_col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1e-12,
                    f"{v:.5f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=45, fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        _clean(ax)
    fig.suptitle(f"{name}: Per-Query Values [{estimate}]", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_combined_all(processed_dict):
    """
    3 subplots (IT, DC electricity, water).
    Each subplot: box plot per scenario with individual points overlaid.
    """
    metrics = [
        ("energy_mid", "IT Energy (Wh)"),
        ("elec_mid",   "DC Electricity (Wh)"),
        ("water_mid",  "Cooling Water (mL)"),
    ]
    names  = list(processed_dict.keys())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (col, title) in zip(axes, metrics):
        data = [processed_dict[n][col].dropna().values for n in names]

        bp = ax.boxplot(data, labels=names, patch_artist=True,
                        medianprops={"color": "red", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": "o", "markersize": 4, "alpha": 0.4})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.60)

        # Jitter all individual points on top
        np.random.seed(42)
        for i, (d, c) in enumerate(zip(data, colors)):
            jitter = np.random.uniform(-0.14, 0.14, len(d))
            ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                       alpha=0.80, s=30, color=c,
                       edgecolors="gray", linewidth=0.4, zorder=5)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        _clean(ax)

    fig.suptitle("All Scenarios: Distribution Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Combined 3×N interactive scenario grid (Altair)
# ──────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def chart_scenario_grid(processed: dict):
    """3 rows × N cols Plotly subplot grid, returned as HTML string.
    Hover a bar → that prompt's bars stay bright, others dim to 15%.
    """
    import json as _json

    scenarios = list(processed.keys())
    n = len(scenarios)

    frames = []
    for sname, df in processed.items():
        tmp = df[["ID", "prompt_short", "energy_mid", "elec_mid", "water_mid"]].copy()
        tmp["scenario"] = sname
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    combined["ID"] = combined["ID"].astype(int)
    combined["Q"]  = "Q" + combined["ID"].astype(str)

    max_ew = max(combined["energy_mid"].max(), combined["elec_mid"].max()) * 1.15
    max_w  = combined["water_mid"].max() * 1.15

    metrics = [
        ("energy_mid", "IT Energy (Wh)",      max_ew),
        ("elec_mid",   "DC Electricity (Wh)", max_ew),
        ("water_mid",  "Cooling Water (mL)",  max_w),
    ]

    fig = make_subplots(
        rows=3, cols=n,
        subplot_titles=scenarios + [""] * (2 * n),
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    # track (scenario_idx, x_array) per trace for JS use
    trace_meta = []

    for ci, sname in enumerate(scenarios):
        df_s = combined[combined["scenario"] == sname].sort_values("ID")
        r, g, b = _hex_to_rgb(PALETTE[ci % len(PALETTE)])
        qs = df_s["Q"].tolist()

        for ri, (metric_col, metric_label, _) in enumerate(metrics):
            yvals = df_s[metric_col].round(5).tolist()
            # per-point color array (start fully opaque)
            colors_init = [f"rgba({r},{g},{b},1.0)"] * len(qs)

            fig.add_trace(
                go.Bar(
                    x=qs,
                    y=yvals,
                    marker_color=colors_init,
                    marker_line_width=0,
                    showlegend=False,
                    text=[f"{v:.4f}" for v in yvals],
                    textposition="outside",
                    textfont_size=7,
                    cliponaxis=False,
                    hoverinfo="none",
                ),
                row=ri + 1, col=ci + 1,
            )
            trace_meta.append({"si": ci, "rgb": [r, g, b], "xs": qs})

    # Y axes: all on left, shared range for rows 1&2, separate for row 3
    for ri, (_, label, y_max) in enumerate(metrics):
        fig.update_yaxes(
            range=[0, y_max * 1.25],   # extra headroom for text labels
            side="left",
            title_text=label,
            title_font_size=10,
            tickfont_size=9,
            row=ri + 1,
        )

    for ri in range(2):
        fig.update_xaxes(showticklabels=False, row=ri + 1)
    fig.update_xaxes(tickangle=-45, tickfont_size=8, row=3)

    fig.update_layout(
        height=560,
        margin=dict(l=70, r=20, t=50, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_dict = fig.to_dict()
    fig_json = _json.dumps(fig_dict)
    meta_json = _json.dumps(trace_meta)

    html = f"""<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>body{{margin:0;padding:0;}}</style>
</head><body>
<div id="g" style="width:100%;"></div>
<script>
var fig = {fig_json};
var meta = {meta_json};
Plotly.newPlot('g', fig.data, fig.layout, {{responsive:true}});
var div = document.getElementById('g');

div.on('plotly_hover', function(d){{
  var hx = d.points[0].x;
  var upd = meta.map(function(m){{
    return m.xs.map(function(x){{
      var a = (x===hx) ? 1.0 : 0.15;
      return 'rgba('+m.rgb[0]+','+m.rgb[1]+','+m.rgb[2]+','+a+')';
    }});
  }});
  Plotly.restyle('g', {{'marker.color': upd}});
}});

div.on('plotly_unhover', function(){{
  var upd = meta.map(function(m){{
    return m.xs.map(function(x){{
      return 'rgba('+m.rgb[0]+','+m.rgb[1]+','+m.rgb[2]+',1.0)';
    }});
  }});
  Plotly.restyle('g', {{'marker.color': upd}});
}});
</script>
</body></html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_manual, tab_csv = st.tabs(["⚙️  Manual Input", "📂  Upload CSV"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: MANUAL INPUT
# ──────────────────────────────────────────────────────────────────────────────
with tab_manual:
    colA, colB, colC, colG = st.columns(4)

    with colA:
        st.subheader("Scenario A")
        model_a  = st.selectbox("Model",         models,    key="ma")
        anchor_a = st.selectbox("Anchor",         anchors,   key="aa")
        loc_a    = st.selectbox("Location",       locations, key="la")
        in_a     = st.number_input("Input Tokens",  0, 500000, 27,  key="ina")
        out_a    = st.number_input("Output Tokens", 0, 500000, 852, key="outa")

    with colB:
        st.subheader("Scenario B")
        model_b  = st.selectbox("Model",         models,    key="mb")
        anchor_b = st.selectbox("Anchor",         anchors,   key="ab")
        loc_b    = st.selectbox("Location",       locations, key="lb")
        in_b     = st.number_input("Input Tokens",  0, 500000, 100, key="inb")
        out_b    = st.number_input("Output Tokens", 0, 500000, 300, key="outb")

    with colC:
        st.subheader("Scenario C")
        model_c  = st.selectbox("Model",         models,    key="mc")
        anchor_c = st.selectbox("Anchor",         anchors,   key="ac")
        loc_c    = st.selectbox("Location",       locations, key="lc")
        in_c     = st.number_input("Input Tokens",  0, 500000, 500, key="inc")
        out_c    = st.number_input("Output Tokens", 0, 500000, 500, key="outc")

    with colG:
        st.subheader(f"Baseline ({GOOGLE_LABEL})")
        searches  = st.number_input("# Searches", 1, 500, 10)
        google_loc = st.selectbox("Location", locations, key="gloc")
        g_energy = Interval(
            lo=GOOGLE_WH_PER_QUERY * searches,
            mid=GOOGLE_WH_PER_QUERY * searches,
            hi=GOOGLE_WH_PER_QUERY * searches,
        )
        g_elec, g_water = interval_from_wh(g_energy, google_loc)

    rA = compute_single(model_a, anchor_a, loc_a, in_a, out_a)
    rB = compute_single(model_b, anchor_b, loc_b, in_b, out_b)
    rC = compute_single(model_c, anchor_c, loc_c, in_c, out_c)

    energy_ents = [
        ("Scenario A",   rA["energy_it_wh"].mid, rA["energy_it_wh"].lo, rA["energy_it_wh"].hi),
        ("Scenario B",   rB["energy_it_wh"].mid, rB["energy_it_wh"].lo, rB["energy_it_wh"].hi),
        ("Scenario C",   rC["energy_it_wh"].mid, rC["energy_it_wh"].lo, rC["energy_it_wh"].hi),
        ("Google Search", g_energy.mid, g_energy.lo, g_energy.hi),
    ]
    elec_ents = [
        ("Scenario A",   kwh_to_wh(rA["electricity_dc_kwh"]).mid, kwh_to_wh(rA["electricity_dc_kwh"]).lo, kwh_to_wh(rA["electricity_dc_kwh"]).hi),
        ("Scenario B",   kwh_to_wh(rB["electricity_dc_kwh"]).mid, kwh_to_wh(rB["electricity_dc_kwh"]).lo, kwh_to_wh(rB["electricity_dc_kwh"]).hi),
        ("Scenario C",   kwh_to_wh(rC["electricity_dc_kwh"]).mid, kwh_to_wh(rC["electricity_dc_kwh"]).lo, kwh_to_wh(rC["electricity_dc_kwh"]).hi),
        ("Google Search", kwh_to_wh(g_elec).mid, kwh_to_wh(g_elec).lo, kwh_to_wh(g_elec).hi),
    ]
    water_ents = [
        ("Scenario A",   l_to_ml(rA["water_onsite_l"]).mid, l_to_ml(rA["water_onsite_l"]).lo, l_to_ml(rA["water_onsite_l"]).hi),
        ("Scenario B",   l_to_ml(rB["water_onsite_l"]).mid, l_to_ml(rB["water_onsite_l"]).lo, l_to_ml(rB["water_onsite_l"]).hi),
        ("Scenario C",   l_to_ml(rC["water_onsite_l"]).mid, l_to_ml(rC["water_onsite_l"]).lo, l_to_ml(rC["water_onsite_l"]).hi),
        ("Google Search", l_to_ml(g_water).mid, l_to_ml(g_water).lo, l_to_ml(g_water).hi),
    ]

    st.divider()
    st.subheader("Comparison Results")
    ESTIMATE = st.selectbox("Estimate", ["Low", "Mid", "High"], index=1, key="estimate_manual")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(plot_summary_bars(energy_ents, "IT Equipment Energy (Wh)", "Wh",
                                    estimate=ESTIMATE))
    with c2:
        st.pyplot(plot_summary_bars(elec_ents, "Data Center Electricity (Wh)", "Wh",
                                    estimate=ESTIMATE))
    with c3:
        st.pyplot(plot_summary_bars(water_ents, "Cooling Water (mL)", "mL",
                                    estimate=ESTIMATE))

    st.divider()
    st.subheader("Interval Output Table")

    def _make_df(ents, metric):
        return pd.DataFrame([
            {"Scenario": e[0], "Low": e[2], "Mid": e[1], "High": e[3], "Metric": metric}
            for e in ents
        ])

    df_all = pd.concat([
        _make_df(energy_ents, "IT Equipment Energy (Wh)"),
        _make_df(elec_ents,   "Data Center Electricity (Wh)"),
        _make_df(water_ents,  "Cooling Water (mL)"),
    ])
    st.dataframe(df_all, width="stretch")
    st.download_button(
        "Download results as CSV",
        df_all.to_csv(index=False).encode("utf-8"),
        "footprint_comparison.csv", "text/csv",
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: UPLOAD CSV
# ──────────────────────────────────────────────────────────────────────────────
with tab_csv:
    st.markdown("""
Each row = one query. Must include **`prompt`** and tool token columns. Tool token columns must be named **`ToolName_in`** and **`ToolName_out`**
(e.g. `Consensus_in`, `Consensus_out`). 
""")

    # ── Single file upload ────────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        try:
            df_full = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_full = None
    else:
        df_full = None

    scenarios = []
    if df_full is not None:
        scenarios = detect_scenarios(df_full)
        if not scenarios:
            # Fallback: old ToolName_in / ToolName_out format
            tool_names = detect_tools(df_full)
            scenarios = [(t, t) for t in tool_names]
        if not scenarios:
            st.error("No scenario columns detected. Use `scenario_N_name/in/out` columns or the legacy `ToolName_in/ToolName_out` pattern.")
            df_full = None

    if df_full is not None:
        # Shared anchor + location
        cfg_c1, cfg_c2 = st.columns(2)
        with cfg_c1:
            shared_anchor = st.selectbox("Anchor",   anchors,   key="csv_anchor")
        with cfg_c2:
            shared_loc    = st.selectbox("Location", locations, key="csv_loc")

        # Per-scenario model selector
        st.markdown("**Model per scenario**")
        model_cols = st.columns(len(scenarios))
        scenario_models = {}
        for col, (prefix, sname) in zip(model_cols, scenarios):
            with col:
                scenario_models[prefix] = st.selectbox(sname, models, key=f"csv_model_{prefix}")

    # ── Google baseline ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚙️  Google Search Baseline", expanded=False):
        gc1, gc2 = st.columns(2)
        with gc1:
            csv_searches = st.number_input("# Searches", 1, 500, 10, key="csv_searches")
        with gc2:
            csv_gloc = st.selectbox("Location", locations, key="csv_gloc")

    csv_g_energy = Interval(
        lo=GOOGLE_WH_PER_QUERY * csv_searches,
        mid=GOOGLE_WH_PER_QUERY * csv_searches,
        hi=GOOGLE_WH_PER_QUERY * csv_searches,
    )
    csv_g_elec, csv_g_water = interval_from_wh(csv_g_energy, csv_gloc)

    # ── Process each scenario from wide-format columns ────────────────────────
    processed = {}
    if df_full is not None:
        for prefix, sname in scenarios:
            try:
                df_proc = process_wide_csv(df_full, prefix,
                                           scenario_models[prefix], shared_anchor, shared_loc)
                if not df_proc.empty:
                    processed[sname] = df_proc
                else:
                    st.warning(f"{sname}: no valid rows.")
            except Exception as e:
                st.error(f"Error processing {sname}: {e}")

    if not processed:
        st.info("Upload a CSV file above to see results.")
    else:
        # ── Build summary entries ─────────────────────────────────────────────
        def _mean_iv(df, lo, mid, hi):
            return df[mid].mean(), df[lo].mean(), df[hi].mean(), df[mid].std()

        csv_energy_ents, csv_elec_ents, csv_water_ents = [], [], []
        for sname, df_s in processed.items():
            e_mid,  e_lo,  e_hi,  e_std  = _mean_iv(df_s, "energy_lo", "energy_mid", "energy_hi")
            el_mid, el_lo, el_hi, el_std = _mean_iv(df_s, "elec_lo",   "elec_mid",   "elec_hi")
            w_mid,  w_lo,  w_hi,  w_std  = _mean_iv(df_s, "water_lo",  "water_mid",  "water_hi")
            csv_energy_ents.append((sname, e_mid,  e_lo,  e_hi,  e_std))
            csv_elec_ents.append(  (sname, el_mid, el_lo, el_hi, el_std))
            csv_water_ents.append( (sname, w_mid,  w_lo,  w_hi,  w_std))

        csv_energy_ents.append(("Google Search", csv_g_energy.mid, csv_g_energy.lo, csv_g_energy.hi, 0))
        csv_elec_ents.append(  ("Google Search", kwh_to_wh(csv_g_elec).mid,
                                                  kwh_to_wh(csv_g_elec).lo,
                                                  kwh_to_wh(csv_g_elec).hi, 0))
        csv_water_ents.append( ("Google Search", l_to_ml(csv_g_water).mid,
                                                  l_to_ml(csv_g_water).lo,
                                                  l_to_ml(csv_g_water).hi, 0))

        # ── SECTION 1: Summary charts ─────────────────────────────────────────
        st.divider()
        st.subheader("I. Summary Comparison")
        ESTIMATE = st.selectbox("Estimate", ["Low", "Mid", "High"], index=1, key="estimate_csv")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.pyplot(plot_summary_bars(csv_energy_ents,
                "IT Equipment Energy (Wh)", "Wh / query",
                estimate=ESTIMATE))
        with s2:
            st.pyplot(plot_summary_bars(csv_elec_ents,
                "Data Center Electricity (Wh)", "Wh / query",
                estimate=ESTIMATE))
        with s3:
            st.pyplot(plot_summary_bars(csv_water_ents,
                "Cooling Water (mL)", "mL / query",
                estimate=ESTIMATE))

        # ── SECTION 2: Per-scenario breakdowns ────────────────────────────────
        st.divider()
        st.subheader("II. Per-Scenario Breakdown")
        st.caption("Hover over a bar to highlight that prompt across all panels.")
        st.components.v1.html(chart_scenario_grid(processed), height=580, scrolling=False)

        # ── Download combined table ───────────────────────────────────────────
        all_frames = []
        for sname, df_s in processed.items():
            tmp = df_s.copy()
            tmp.insert(0, "scenario", sname)
            all_frames.append(tmp)
        combined_csv = pd.concat(all_frames, ignore_index=True).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download full results table (all scenarios)",
            combined_csv,
            "all_scenarios_footprint.csv",
            "text/csv",
            key="dl_all",
        )

        # ── SECTION 4: Per-prompt cross-tool comparison ───────────────────────
        st.divider()
        st.subheader("III. Per-Prompt Cross-Tool Comparison")

        first_df = next(iter(processed.values()))
        prompt_options = {
            str(row["ID"]): f"Q{int(row['ID'])}: {row['prompt']}"
            for _, row in first_df.iterrows()
        }

        sel_id = st.selectbox(
            "Select prompt",
            options=list(prompt_options.keys()),
            format_func=lambda k: prompt_options[k],
            key="cmp_prompt",
        )

        metrics_list = [
            ("IT Energy (Wh)",      "energy_mid", "energy_lo", "energy_hi"),
            ("DC Electricity (Wh)", "elec_mid",   "elec_lo",   "elec_hi"),
            ("Cooling Water (mL)",  "water_mid",  "water_lo",  "water_hi"),
        ]

        fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (metric_label, mid_col, lo_col, hi_col) in zip(axes_cmp, metrics_list):
            cmp_entries = []
            for j, (tool, df_t) in enumerate(processed.items()):
                row = df_t[df_t["ID"].astype(str) == str(sel_id)]
                if not row.empty:
                    r = row.iloc[0]
                    cmp_entries.append((tool, r[mid_col], r[lo_col], r[hi_col],
                                        PALETTE[j % len(PALETTE)]))

            xs     = range(len(cmp_entries))
            colors = [e[4] for e in cmp_entries]
            vals   = [e[1] for e in cmp_entries]

            bars = ax.bar(xs, vals, color=colors, alpha=0.82)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1e-12,
                        f"{v:.5f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

            ax.set_xticks(list(xs))
            ax.set_xticklabels([e[0] for e in cmp_entries], fontsize=10, rotation=15, ha="right")
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_title(metric_label, fontsize=10, fontweight="bold")
            _clean(ax)

        fig_cmp.suptitle(prompt_options[sel_id], fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)
