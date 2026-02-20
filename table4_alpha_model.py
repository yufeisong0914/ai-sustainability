#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derive per-token energy coefficient ranges (alpha_min, alpha_max) from Table 4 ranges.

Table 4 typically reports IT energy ranges under different inference engines / settings.
We preserve uncertainty by converting:
    Energy_IT_Wh ∈ [E_min, E_max]
to
    alpha_Wh_per_token ∈ [E_min/(in+out), E_max/(in+out)]

Outputs:
- alpha_ranges.json
- alpha_ranges.csv
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------
# Table 4 ranges (YOU FILL THESE)
# Store: (input_tokens, output_tokens, energy_min_wh, energy_max_wh)
# ---------------------------------------------------------------------
# (input_tokens, output_tokens, energy_min_wh, energy_max_wh)
TABLE4_RANGES: Dict[str, Dict[str, Tuple[int, int, float, float]]] = {

    # ===== OpenAI =====
    "gpt-4.1": {
        "100in_300out":    (100, 300, 0.871 - 0.302, 0.871 + 0.302),   # [0.569, 1.173]
        "1k_in_1k_out":    (1000, 1000, 3.161 - 0.515, 3.161 + 0.515),# [2.646, 3.676]
        "10k_in_1.5k_out": (10000, 1500, 4.833 - 0.650, 4.833 + 0.650)# [4.183, 5.483]
    },

    "gpt-4.1-mini": {
        "100in_300out":    (100, 300, 0.450 - 0.081, 0.450 + 0.081),   # [0.369, 0.531]
        "1k_in_1k_out":    (1000, 1000, 1.545 - 0.211, 1.545 + 0.211),# [1.334, 1.756]
        "10k_in_1.5k_out": (10000, 1500, 2.122 - 0.348, 2.122 + 0.348)# [1.774, 2.470]
    },

    "gpt-4.1-nano": {
        "100in_300out":    (100, 300, 0.207 - 0.047, 0.207 + 0.047),   # [0.160, 0.254]
        "1k_in_1k_out":    (1000, 1000, 0.575 - 0.108, 0.575 + 0.108),# [0.467, 0.683]
        "10k_in_1.5k_out": (10000, 1500, 0.827 - 0.094, 0.827 + 0.094)# [0.733, 0.921]
    },

    "o4-mini-high": {
        "100in_300out":    (100, 300, 3.649 - 1.468, 3.649 + 1.468),  # [2.181, 5.117]
        "1k_in_1k_out":    (1000, 1000, 7.380 - 2.177, 7.380 + 2.177),# [5.203, 9.557]
        "10k_in_1.5k_out": (10000, 1500, 7.237 - 1.674, 7.237 + 1.674)# [5.563, 8.911]
    },

    "o3": {
        "100in_300out":    (100, 300, 1.177 - 0.224, 1.177 + 0.224),  # [0.953, 1.401]
        "1k_in_1k_out":    (1000, 1000, 5.153 - 2.107, 5.153 + 2.107),# [3.046, 7.260]
        "10k_in_1.5k_out": (10000, 1500, 12.222 - 1.082, 12.222 + 1.082)# [11.140, 13.304]
    },

    "o3-mini-high": {
        "100in_300out":    (100, 300, 3.012 - 0.991, 3.012 + 0.991),  # [2.021, 4.003]
        "1k_in_1k_out":    (1000, 1000, 6.865 - 1.330, 6.865 + 1.330),# [5.535, 8.195]
        "10k_in_1.5k_out": (10000, 1500, 5.389 - 1.183, 5.389 + 1.183)# [4.206, 6.572]
    },

    "o3-mini": {
        "100in_300out":    (100, 300, 0.674 - 0.015, 0.674 + 0.015),  # [0.659, 0.689]
        "1k_in_1k_out":    (1000, 1000, 2.423 - 0.237, 2.423 + 0.237),# [2.186, 2.660]
        "10k_in_1.5k_out": (10000, 1500, 3.525 - 0.168, 3.525 + 0.168)# [3.357, 3.693]
    },

    "o1": {
        "100in_300out":    (100, 300, 2.268 - 0.654, 2.268 + 0.654),  # [1.614, 2.922]
        "1k_in_1k_out":    (1000, 1000, 4.047 - 0.497, 4.047 + 0.497),# [3.550, 4.544]
        "10k_in_1.5k_out": (10000, 1500, 6.181 - 0.877, 6.181 + 0.877)# [5.304, 7.058]
    },

    "o1-mini": {
        "100in_300out":    (100, 300, 0.535 - 0.182, 0.535 + 0.182),  # [0.353, 0.717]
        "1k_in_1k_out":    (1000, 1000, 1.547 - 0.405, 1.547 + 0.405),# [1.142, 1.952]
        "10k_in_1.5k_out": (10000, 1500, 2.317 - 0.530, 2.317 + 0.530)# [1.787, 2.847]
    },

    # ===== Claude =====
    "claude-3.7-sonnet": {
        "100in_300out":    (100, 300, 0.950 - 0.040, 0.950 + 0.040),  # [0.910, 0.990]
        "1k_in_1k_out":    (1000, 1000, 2.989 - 0.201, 2.989 + 0.201),# [2.788, 3.190]
        "10k_in_1.5k_out": (10000, 1500, 5.671 - 0.302, 5.671 + 0.302)# [5.369, 5.973]
    },

    "claude-3.5-sonnet": {
        "100in_300out":    (100, 300, 0.973 - 0.066, 0.973 + 0.066),  # [0.907, 1.039]
        "1k_in_1k_out":    (1000, 1000, 3.638 - 0.256, 3.638 + 0.256),# [3.382, 3.894]
        "10k_in_1.5k_out": (10000, 1500, 7.772 - 0.345, 7.772 + 0.345)# [7.427, 8.117]
    },

    "claude-3.5-haiku": {
        "100in_300out":    (100, 300, 0.975 - 0.063, 0.975 + 0.063),  # [0.912, 1.038]
        "1k_in_1k_out":    (1000, 1000, 4.464 - 0.283, 4.464 + 0.283),# [4.181, 4.747]
        "10k_in_1.5k_out": (10000, 1500, 8.010 - 0.338, 8.010 + 0.338)# [7.672, 8.348]
    },

    # ===== LLaMA =====
    "llama-3-8b": {
        "100in_300out":    (100, 300, 0.108 - 0.002, 0.108 + 0.002),  # [0.106, 0.110]
        "1k_in_1k_out":    (1000, 1000, 0.370 - 0.005, 0.370 + 0.005),# [0.365, 0.375]
    },

    "llama-3.70b": {
        "100in_300out":    (100, 300, 0.861 - 0.022, 0.861 + 0.022),  # [0.839, 0.883]
        "1k_in_1k_out":    (1000, 1000, 2.871 - 0.094, 2.871 + 0.094),# [2.777, 2.965]
    },

    "llama-3.1-8b": {
        "100in_300out":    (100, 300, 0.052 - 0.008, 0.052 + 0.008),  # [0.044, 0.060]
        "1k_in_1k_out":    (1000, 1000, 0.172 - 0.015, 0.172 + 0.015),# [0.157, 0.187]
        "10k_in_1.5k_out": (10000, 1500, 0.443 - 0.028, 0.443 + 0.028)# [0.415, 0.471]
    },

    "llama-3.1-70b": {
        "100in_300out":    (100, 300, 1.271 - 0.020, 1.271 + 0.020),  # [1.251, 1.291]
        "1k_in_1k_out":    (1000, 1000, 4.525 - 0.053, 4.525 + 0.053),# [4.472, 4.578]
        "10k_in_1.5k_out": (10000, 1500, 19.183 - 0.560, 19.183 + 0.560)# [18.623, 19.743]
    },

    "llama-3.1-405b": {
        "100in_300out":    (100, 300, 2.226 - 0.142, 2.226 + 0.142),  # [2.084, 2.368]
        "1k_in_1k_out":    (1000, 1000, 9.042 - 0.385, 9.042 + 0.385),# [8.657, 9.427]
        "10k_in_1.5k_out": (10000, 1500, 25.202 - 0.526, 25.202 + 0.526)# [24.676, 25.728]
    },

    "llama-3.2-1b": {
        "100in_300out":    (100, 300, 0.109 - 0.013, 0.109 + 0.013),  # [0.096, 0.122]
        "1k_in_1k_out":    (1000, 1000, 0.342 - 0.025, 0.342 + 0.025),# [0.317, 0.367]
        "10k_in_1.5k_out": (10000, 1500, 0.552 - 0.059, 0.552 + 0.059)# [0.493, 0.611]
    },

    "llama-3.2-3b": {
        "100in_300out":    (100, 300, 0.143 - 0.006, 0.143 + 0.006),  # [0.137, 0.149]
        "1k_in_1k_out":    (1000, 1000, 0.479 - 0.017, 0.479 + 0.017),# [0.462, 0.496]
        "10k_in_1.5k_out": (10000, 1500, 0.707 - 0.020, 0.707 + 0.020)# [0.687, 0.727]
    },

    "llama-3.2-vision-11b": {
        "100in_300out":    (100, 300, 0.078 - 0.021, 0.078 + 0.021),  # [0.057, 0.099]
        "1k_in_1k_out":    (1000, 1000, 0.242 - 0.071, 0.242 + 0.071),# [0.171, 0.313]
        "10k_in_1.5k_out": (10000, 1500, 1.087 - 0.060, 1.087 + 0.060)# [1.027, 1.147]
    },

    "llama-3.2-vision-90b": {
        "100in_300out":    (100, 300, 1.235 - 0.054, 1.235 + 0.054),  # [1.181, 1.289]
        "1k_in_1k_out":    (1000, 1000, 4.534 - 0.448, 4.534 + 0.448),# [4.086, 4.982]
        "10k_in_1.5k_out": (10000, 1500, 6.852 - 0.780, 6.852 + 0.780)# [6.072, 7.632]
    },

    "llama-3.3-70b": {
        "100in_300out":    (100, 300, 0.237 - 0.023, 0.237 + 0.023),  # [0.214, 0.260]
        "1k_in_1k_out":    (1000, 1000, 0.760 - 0.079, 0.760 + 0.079),# [0.681, 0.839]
        "10k_in_1.5k_out": (10000, 1500, 1.447 - 0.188, 1.447 + 0.188)# [1.259, 1.635]
    },
}

@dataclass(frozen=True)
class AlphaRange:
    model: str
    anchor: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    energy_min_wh: float
    energy_max_wh: float
    alpha_min_wh_per_token: float
    alpha_max_wh_per_token: float
    alpha_mid_wh_per_token: float  # midpoint (optional convenience)


def derive_alpha_range(model: str, anchor: str) -> AlphaRange:
    if model not in TABLE4_RANGES:
        raise ValueError(f"Model '{model}' not found in TABLE4_RANGES.")
    if anchor not in TABLE4_RANGES[model]:
        raise ValueError(f"Anchor '{anchor}' not found for model '{model}'.")

    in_tok, out_tok, e_min, e_max = TABLE4_RANGES[model][anchor]

    if in_tok < 0 or out_tok < 0:
        raise ValueError("Token counts must be non-negative.")
    total = in_tok + out_tok
    if total <= 0:
        raise ValueError("Total tokens must be > 0.")
    if e_min <= 0 or e_max <= 0:
        raise ValueError("Energy bounds must be > 0.")
    if e_min > e_max:
        raise ValueError(f"Energy min > max for {model}/{anchor}: {e_min} > {e_max}")

    a_min = e_min / total
    a_max = e_max / total
    a_mid = 0.5 * (a_min + a_max)

    return AlphaRange(
        model=model,
        anchor=anchor,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=total,
        energy_min_wh=e_min,
        energy_max_wh=e_max,
        alpha_min_wh_per_token=a_min,
        alpha_max_wh_per_token=a_max,
        alpha_mid_wh_per_token=a_mid,
    )


def build_alpha_ranges(
    preferred_anchor: str = "1k_in_1k_out",
    fallback_to_any_anchor: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, AlphaRange]]]:
    """
    Returns:
      alpha_by_model: {model: {"alpha_min":..., "alpha_max":..., "alpha_mid":..., "anchor":...}}
      alpha_details:  {model: {anchor: AlphaRange(...)}}  (all anchors)
    """
    alpha_details: Dict[str, Dict[str, AlphaRange]] = {}
    for model, anchors in TABLE4_RANGES.items():
        alpha_details[model] = {}
        for anchor in anchors.keys():
            alpha_details[model][anchor] = derive_alpha_range(model, anchor)

    alpha_by_model: Dict[str, Dict[str, float]] = {}

    for model, anchors in alpha_details.items():
        chosen: Optional[AlphaRange] = None
        if preferred_anchor in anchors:
            chosen = anchors[preferred_anchor]
        elif fallback_to_any_anchor and len(anchors) > 0:
            # deterministic fallback: pick the first anchor by sorted key
            first_anchor = sorted(anchors.keys())[0]
            chosen = anchors[first_anchor]

        if chosen is None:
            continue

        alpha_by_model[model] = {
            "alpha_min_wh_per_token": chosen.alpha_min_wh_per_token,
            "alpha_max_wh_per_token": chosen.alpha_max_wh_per_token,
            "alpha_mid_wh_per_token": chosen.alpha_mid_wh_per_token,
            "anchor_used": chosen.anchor,
        }

    return alpha_by_model, alpha_details


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: Dict[str, Dict[str, float]]) -> None:
    fieldnames = ["model", "anchor_used", "alpha_min_wh_per_token", "alpha_max_wh_per_token", "alpha_mid_wh_per_token"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model, d in sorted(rows.items()):
            w.writerow({
                "model": model,
                "anchor_used": d["anchor_used"],
                "alpha_min_wh_per_token": d["alpha_min_wh_per_token"],
                "alpha_max_wh_per_token": d["alpha_max_wh_per_token"],
                "alpha_mid_wh_per_token": d["alpha_mid_wh_per_token"],
            })


def main() -> None:
    alpha_by_model, alpha_details = build_alpha_ranges(
        preferred_anchor="1k_in_1k_out",
        fallback_to_any_anchor=True,
    )

    # Output summary alpha ranges (per model, single chosen anchor)
    write_json("alpha_ranges.json", alpha_by_model)
    write_csv("alpha_ranges.csv", alpha_by_model)

    # Output full details (all anchors) for auditability
    details_serializable = {
        model: {anchor: asdict(ar) for anchor, ar in anchors.items()}
        for model, anchors in alpha_details.items()
    }
    write_json("alpha_ranges_full_anchors.json", details_serializable)

    print("Wrote:")
    print(" - alpha_ranges.json")
    print(" - alpha_ranges.csv")
    print(" - alpha_ranges_full_anchors.json")
    print(f"Models exported: {len(alpha_by_model)}")


if __name__ == "__main__":
    main()
