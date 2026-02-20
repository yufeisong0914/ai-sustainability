# python3 evaluation.py

# anchor: use scenarios?

"""
Token-based estimation of LLM inference energy, electricity, and cooling water use
using precomputed per-token energy coefficient intervals (alpha ranges).

Workflow:
1) Load alpha ranges (Wh/token) derived from Table 4 (mean ± std) and stored externally.
2) Given a user-specified workload (total token count), scale alpha to obtain IT energy.
3) Apply location-specific PUE and WUE to estimate data center electricity and
   direct on-site cooling water consumption.

Notes:
- Users specify ONLY the workload size (total_tokens).
- Prompt configurations such as "100in_300out" are calibration anchors used
  solely to derive alpha, not constraints on user workloads.
- Location affects only the scaling stage (PUE/WUE), not alpha itself.
"""

# =====================================================================
# 1) Load precomputed alpha ranges
# =====================================================================
# alpha_ranges_full_anchors.json structure:
#   alpha_ranges[model][anchor][
#       "alpha_min_wh_per_token",
#       "alpha_mid_wh_per_token",
#       "alpha_max_wh_per_token"
#   ]

import json
from dataclasses import dataclass
from typing import Dict

with open("alpha_range_data/alpha_ranges_full_anchors.json", "r") as f:
    ALPHA_TABLE = json.load(f)


# =====================================================================
# 2) Core data structure
# =====================================================================

@dataclass(frozen=True)
class Interval:
    """
    Closed interval representation for uncertainty propagation.

    Attributes
    ----------
    lo : float
        Lower bound
    mid : float
        Mid-point (representative estimate)
    hi : float
        Upper bound
    """
    lo: float
    mid: float
    hi: float


# =====================================================================
# 3) IT energy estimation (model-level, location-agnostic)
# =====================================================================

def estimate_it_energy_interval(
    model: str,
    anchor: str,
    total_tokens: int,   # user-specified workload size
) -> Interval:
    """
    Estimate inference-time IT energy consumption (Wh) as an interval.

    Parameters
    ----------
    model : str
        Model identifier.
    anchor : str
        Calibration anchor used to derive alpha (e.g., "100in_300out").
        This is NOT a user workload choice.
    total_tokens : int
        Total number of tokens in the user workload (input + output).

    Returns
    -------
    Interval
        IT energy consumption in Wh (lo, mid, hi).
    """
    entry = ALPHA_TABLE[model][anchor]

    alpha_lo  = entry["alpha_min_wh_per_token"]
    alpha_mid = entry["alpha_mid_wh_per_token"]
    alpha_hi  = entry["alpha_max_wh_per_token"]

    return Interval(
        lo=alpha_lo  * total_tokens,
        mid=alpha_mid * total_tokens,
        hi=alpha_hi  * total_tokens,
    )


# =====================================================================
# 4) Location-specific scaling parameters
# =====================================================================
# PUE: Power Usage Effectiveness (dimensionless)
# WUE: Water Usage Effectiveness (L/kWh, direct on-site cooling water)
# Source: Microsoft Datacenter Sustainability (2023–2024)

LOCATION_PARAMS: Dict[str, Dict[str, float]] = {

    # ---- Global baseline ----
    "global-average": {"pue": 1.16, "wue": 0.30},

    # ---- United States ----
    "arizona":    {"pue": 1.13, "wue": 1.52},
    "illinois":   {"pue": 1.25, "wue": 0.52},
    "iowa":       {"pue": 1.16, "wue": 0.10},
    "texas":      {"pue": 1.28, "wue": 0.24},
    "virginia":   {"pue": 1.14, "wue": 0.18},
    "washington": {"pue": 1.16, "wue": 0.70},
    "wyoming":    {"pue": 1.12, "wue": 0.16},

    # ---- International ----
    "singapore":   {"pue": 1.30, "wue": 0.02},
    "ireland":     {"pue": 1.18, "wue": 0.02},
    "netherlands": {"pue": 1.14, "wue": 0.04},
    "sweden":      {"pue": 1.16, "wue": 0.05},
    "poland":      {"pue": 1.19, "wue": 0.44},
}


# =====================================================================
# 5) End-to-end footprint estimation (IT → electricity → water)
# =====================================================================

def estimate_footprint_interval(
    model: str,
    anchor: str,
    total_tokens: int,
    location: str,
):
    """
    Estimate energy, electricity, and on-site cooling water consumption
    as uncertainty intervals.

    Parameters
    ----------
    model : str
        Model identifier.
    anchor : str
        Calibration anchor used to derive alpha.
    total_tokens : int
        User workload size (input + output tokens).
    location : str
        Data center location key (for PUE/WUE lookup).

    Returns
    -------
    dict
        Dictionary containing interval estimates for:
        - IT energy (Wh)
        - Data center electricity (kWh)
        - Direct on-site cooling water (L)
    """
    # --- IT energy (location-agnostic) ---
    energy_it = estimate_it_energy_interval(
        model=model,
        anchor=anchor,
        total_tokens=total_tokens,
    )

    # --- Location scaling ---
    pue = LOCATION_PARAMS[location]["pue"]
    wue = LOCATION_PARAMS[location]["wue"]

    # Data center electricity (kWh)
    electricity_dc = Interval(
        lo=(energy_it.lo  / 1000.0) * pue,
        mid=(energy_it.mid / 1000.0) * pue,
        hi=(energy_it.hi  / 1000.0) * pue,
    )

    # Direct on-site cooling water (L)
    water_onsite = Interval(
        lo=electricity_dc.lo  * wue,
        mid=electricity_dc.mid * wue,
        hi=electricity_dc.hi  * wue,
    )

    return {
        "energy_it_wh": energy_it,
        "electricity_dc_kwh": electricity_dc,
        "water_onsite_l": water_onsite,
    }


# =====================================================================
# 6) Example usage
# =====================================================================

if __name__ == "__main__":

    model = "gpt-4.1"
    anchor = "100in_300out"
    input_tokens = 27
    output_tokens = 852
    total_tokens = input_tokens + output_tokens

    for loc in ["global-average"]:
        r = estimate_footprint_interval(
            model=model,
            anchor=anchor,
            total_tokens=total_tokens,
            location=loc,
        )

        print("-" * 50)
        print(f"Location: {loc}")
        print("IT energy (Wh):", r["energy_it_wh"])
        print("DC electricity (kWh):", r["electricity_dc_kwh"])
        print("On-site water (L):", r["water_onsite_l"])
