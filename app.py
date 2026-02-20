import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from evaluation import (
    estimate_footprint_interval,
    LOCATION_PARAMS,
    Interval,
    ALPHA_TABLE
)

# =====================================================
# Page Config
# =====================================================

st.set_page_config(page_title="LLM Footprint Comparator", layout="wide")

st.title("🌍 LLM Footprint Comparator Dashboard")
st.markdown("""
Compare three LLM inference workloads against a Google plain search baseline (non-AI control group).

Baseline assumption:

- Google Search = 0.3 Wh/query (Google, 2009)
""")

# =====================================================
# Helper Functions
# =====================================================

def interval_from_wh(wh_interval: Interval, location: str):
    """Convert IT energy (Wh) → electricity (kWh) → cooling water (L)."""
    pue = LOCATION_PARAMS[location]["pue"]
    wue = LOCATION_PARAMS[location]["wue"]

    electricity = Interval(
        lo=(wh_interval.lo / 1000) * pue,
        mid=(wh_interval.mid / 1000) * pue,
        hi=(wh_interval.hi / 1000) * pue,
    )

    water = Interval(
        lo=electricity.lo * wue,
        mid=electricity.mid * wue,
        hi=electricity.hi * wue,
    )

    return electricity, water


def plot_bars_with_labels(df, title, ylabel):
    """Bar chart with uncertainty + value labels."""
    fig, ax = plt.subplots()

    lower_err = df["Mid"] - df["Low"]
    upper_err = df["High"] - df["Mid"]

    bars = ax.bar(
        df["Scenario"],
        df["Mid"],
        yerr=[lower_err, upper_err],
        capsize=8,
        color=["orange", "gold", "blue", "green"]
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(df["Scenario"], rotation=15, ha="right")

    # --- Add numeric labels on top ---
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    return fig


def compute_llm(label, model, anchor, location, in_tok, out_tok):
    """Run footprint pipeline for one scenario."""
    total_tokens = int(in_tok) + int(out_tok)

    result = estimate_footprint_interval(
        model=model,
        anchor=anchor,
        total_tokens=total_tokens,
        location=location
    )

    return result


# =====================================================
# Dropdown options
# =====================================================

models = sorted(ALPHA_TABLE.keys())
anchors = sorted(list(ALPHA_TABLE[models[0]].keys()))
locations = list(LOCATION_PARAMS.keys())

# =====================================================
# Input UI: 4 Columns
# =====================================================

colA, colB, colC, colG = st.columns(4)

# ---------------- Scenario A ----------------
with colA:
    st.subheader("Scenario A (LLM)")
    model_a = st.selectbox("Model", models, key="ma")
    anchor_a = st.selectbox("Anchor", anchors, key="aa")
    loc_a = st.selectbox("Location", locations, key="la")
    in_a = st.number_input("Input Tokens", 0, 500000, 27, key="ina")
    out_a = st.number_input("Output Tokens", 0, 500000, 852, key="outa")

# ---------------- Scenario B ----------------
with colB:
    st.subheader("Scenario B (LLM)")
    model_b = st.selectbox("Model ", models, key="mb")
    anchor_b = st.selectbox("Anchor ", anchors, key="ab")
    loc_b = st.selectbox("Location ", locations, key="lb")
    in_b = st.number_input("Input Tokens ", 0, 500000, 100, key="inb")
    out_b = st.number_input("Output Tokens ", 0, 500000, 300, key="outb")

# ---------------- Scenario C ----------------
with colC:
    st.subheader("Scenario C (LLM)")
    model_c = st.selectbox("Model  ", models, key="mc")
    anchor_c = st.selectbox("Anchor  ", anchors, key="ac")
    loc_c = st.selectbox("Location  ", locations, key="lc")
    in_c = st.number_input("Input Tokens  ", 0, 500000, 500, key="inc")
    out_c = st.number_input("Output Tokens  ", 0, 500000, 500, key="outc")

# ---------------- Google Baseline ----------------
with colG:
    st.subheader("Baseline (Google)")
    searches = st.number_input("# Searches", 1, 500, 10)

    google_energy = Interval(
        lo=0.3 * searches,
        mid=0.3 * searches,
        hi=0.3 * searches
    )

    google_loc = st.selectbox(
        "Location",
        locations,
        key="gloc"
    )

    google_electricity, google_water = interval_from_wh(
        google_energy,
        google_loc
    )

# =====================================================
# Compute Results
# =====================================================

rA = compute_llm("A", model_a, anchor_a, loc_a, in_a, out_a)
rB = compute_llm("B", model_b, anchor_b, loc_b, in_b, out_b)
rC = compute_llm("C", model_c, anchor_c, loc_c, in_c, out_c)

# =====================================================
# Build DataFrames for each metric
# =====================================================

def pack(label, interval):
    return {
        "Scenario": label,
        "Low": interval.lo,
        "Mid": interval.mid,
        "High": interval.hi
    }


df_energy = pd.DataFrame([
    pack("Scenario A", rA["energy_it_wh"]),
    pack("Scenario B", rB["energy_it_wh"]),
    pack("Scenario C", rC["energy_it_wh"]),
    pack("Google Search", google_energy),
])

df_electricity = pd.DataFrame([
    pack("Scenario A", rA["electricity_dc_kwh"]),
    pack("Scenario B", rB["electricity_dc_kwh"]),
    pack("Scenario C", rC["electricity_dc_kwh"]),
    pack("Google Search", google_electricity),
])

df_water = pd.DataFrame([
    pack("Scenario A", rA["water_onsite_l"]),
    pack("Scenario B", rB["water_onsite_l"]),
    pack("Scenario C", rC["water_onsite_l"]),
    pack("Google Search", google_water),
])

# =====================================================
# Plot Charts (3 columns)
# =====================================================

st.divider()
st.subheader("Comparison Results (Mid values labeled)")

c1, c2, c3 = st.columns(3)

with c1:
    fig1 = plot_bars_with_labels(df_energy, "IT Equipment Electricity Consumption (Wh)", "Wh")
    st.pyplot(fig1)

with c2:
    fig2 = plot_bars_with_labels(df_electricity, "Data Center Electricity Consumption (kWh)", "kWh")
    st.pyplot(fig2)

with c3:
    fig3 = plot_bars_with_labels(df_water, "Cooling Water Consumption (L)", "Liters")
    st.pyplot(fig3)

# =====================================================
# Table Output
# =====================================================

st.divider()
st.subheader("Interval Output Table")

df_all = pd.concat([
    df_energy.assign(Metric="IT Equipment Electricity Consumption (Wh)"),
    df_electricity.assign(Metric="Data Center Electricity Consumption (kWh)"),
    df_water.assign(Metric="Cooling Water Consumption (L)")
])

st.dataframe(df_all, use_container_width=True)

# Download CSV
csv = df_all.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    csv,
    file_name="footprint_comparison.csv",
    mime="text/csv"
)
