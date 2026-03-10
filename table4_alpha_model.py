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
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------
# Table 4 ranges — loaded from config/table4_ranges.csv
# Columns: model, anchor, input_tokens, output_tokens,
#          energy_min_wh, energy_mid_wh, energy_max_wh
#
# To add a new model or update measurements, edit that CSV file.
# Run this script afterwards to regenerate the alpha JSON files.
# ---------------------------------------------------------------------

def _load_table4_ranges(csv_path: str) -> Dict[str, Dict[str, Tuple[int, int, float, float]]]:
    """Load Table 4 energy ranges from CSV into the nested dict format expected downstream."""
    ranges: Dict[str, Dict[str, Tuple[int, int, float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model  = row["model"]
            anchor = row["anchor"]
            entry  = (
                int(row["input_tokens"]),
                int(row["output_tokens"]),
                float(row["energy_min_wh"]),
                float(row["energy_max_wh"]),
            )
            ranges.setdefault(model, {})[anchor] = entry
    return ranges

_TABLE4_CSV = os.path.join(os.path.dirname(__file__), "config", "table4_ranges.csv")
TABLE4_RANGES: Dict[str, Dict[str, Tuple[int, int, float, float]]] = _load_table4_ranges(_TABLE4_CSV)

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
