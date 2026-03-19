# LLM Footprint Comparator Dashboard

A Streamlit dashboard for estimating and comparing the environmental footprint
(energy, electricity, and cooling water) of LLM inference workloads against a
plain Google Search baseline.

---

## Project Structure

```
ai_sustainability/
├── app.py                          # Streamlit dashboard (entry point)
├── evaluation.py                   # Core calculation logic
├── table4_alpha_model.py           # One-time script: derives alpha ranges from Table 4 data
├── config/
│   ├── table4_ranges.csv           # ← Source calibration data (model energy measurements by anchor)
│   ├── location_params.csv         # ← Edit to add/update datacenter locations (PUE, WUE)
│   └── baseline_params.csv         # ← Edit to update the Google baseline energy figure
└── alpha_range_data/
    ├── alpha_ranges_full_anchors.json   # Pre-computed per-token energy coefficients (all anchors)
    ├── alpha_ranges.json                # Summary (one anchor per model)
    └── alpha_ranges.csv                 # Summary in CSV form
```

---

## Dashboard

The app has two tabs.

### ⚙️ Manual Input

Enter token counts and model parameters directly for up to three scenarios plus a Google Search baseline. Results are shown as three side-by-side bar charts (IT energy, datacenter electricity, cooling water) with uncertainty intervals, and a downloadable results table.

### 📂 Upload CSV

Upload a single CSV file containing test results for multiple tools. Each row is one query; token columns must follow the naming convention `ToolName_in` / `ToolName_out` (e.g. `Consensus_in`, `Consensus_out`).

**Required columns**

| Column | Description |
|---|---|
| `prompt` | Query text (shown truncated in charts) |
| `ToolName_in` | Input token count for that tool |
| `ToolName_out` | Output token count for that tool |

**Optional columns**

| Column | Description |
|---|---|
| `ID` | Query identifier |
| `T0 (any output)` | Time to first output (s) |
| `T1 (first byte)` | Time to first byte (s) |
| `T2 (full-length)` | Time to full response (s) |
| `citation_total` | Total citations retrieved |
| `citation_used` | Citations actually used in response |
| `word_count_in` | Input word count |
| `word_count_out` | Output word count |

**Example CSV format**

```
ID,prompt,Consensus_in,Consensus_out,Keenious_in,Keenious_out,Scopusai_in,Scopusai_out
1,What are recent advances in X?,16,1249,16,181,16,1393
2,How does Y affect Z?,19,849,19,134,19,869
```

After uploading, select a model per tool (shared anchor and location), then the dashboard generates:

1. **Summary Comparison** — mean IT energy, DC electricity, and cooling water per query for each tool vs. the Google baseline.
2. **Per-Scenario Breakdown** — for each tool: per-query bar charts for all three footprint metrics, plus KPI metrics and a raw data table.
3. **Per-Prompt Cross-Tool Comparison** — select any individual query to see all three footprint metrics side by side across tools.

---

## Calculation Pipeline

### Step 1 — Per-token energy coefficient (alpha)

Alpha (α, in Wh/token) is the energy cost per token for a given model. It is
derived **once** by running `table4_alpha_model.py`, which reads calibration
measurements from Table 4 of the source paper:

```
Energy_IT_Wh ∈ [E_min, E_max]   (from Table 4, measured at a known token count)

α_min = E_min / (input_tokens + output_tokens)
α_max = E_max / (input_tokens + output_tokens)
α_mid = (α_min + α_max) / 2
```

Each model has multiple **calibration anchors** (e.g. `100in_300out`,
`1k_in_1k_out`, `10k_in_1.5k_out`). An anchor is the specific token
configuration used during measurement — it is **not** a constraint on user
workloads; it only determines which alpha value is used.

The results are saved to `alpha_range_data/alpha_ranges_full_anchors.json` and
read at runtime by the dashboard.

### Step 2 — IT equipment energy for a user workload

Given a user-specified workload of `total_tokens = input_tokens + output_tokens`:

```
E_IT (Wh) = α × total_tokens

→ produces an interval [E_IT_lo, E_IT_mid, E_IT_hi]
```

This is location-agnostic: it reflects only the compute energy of the model
itself, not the surrounding datacenter infrastructure.

### Step 3 — Datacenter electricity consumption

The IT energy is scaled by **PUE** (Power Usage Effectiveness) to account for
datacenter overhead (cooling systems, power distribution losses, lighting, etc.):

```
E_DC (Wh) = E_IT_Wh × PUE
```

PUE is dimensionless. A PUE of 1.0 would mean 100% of electricity goes to IT
equipment; real datacenters are always > 1.0.

### Step 4 — On-site cooling water consumption

Datacenter electricity is multiplied by **WUE** (Water Usage Effectiveness) to
estimate direct on-site water evaporated for cooling:

```
Water (L) = E_DC_kWh × WUE
```

WUE is measured in litres per kWh of IT load. It varies significantly by
location based on climate and cooling strategy (e.g. air cooling vs. evaporative
cooling towers).

### Google Search Baseline

The baseline represents a single plain Google web search. It uses a flat
point-estimate (no uncertainty interval):

```
Energy_baseline (Wh) = energy_wh_per_query × number_of_searches
```

The same PUE/WUE scaling (Steps 3–4) is then applied, using the location
selected for the baseline column in the UI.

---

## Configuration Files

All key numeric parameters are stored in CSV files under `config/`. You can
update values there without touching any Python code.

### `config/location_params.csv`

Controls which datacenter locations are available in the dashboard and their
efficiency parameters.

| Column     | Description                                              |
|------------|----------------------------------------------------------|
| `location` | Location key shown in the dashboard dropdown             |
| `pue`      | Power Usage Effectiveness (dimensionless, always ≥ 1.0)  |
| `wue`      | Water Usage Effectiveness (L/kWh, on-site direct water)  |
| `notes`    | Free-text source or description (not used in calculation)|

**Source:** Microsoft Datacenter Sustainability Report (2023–2024).

To add a new location, append a row. To update a value, edit the `pue` or `wue`
column. Changes take effect the next time the dashboard is (re)loaded.

### `config/table4_ranges.csv`

The primary data table. Each row is one measured calibration point for a model
at a specific token configuration (anchor). This is the source of truth for all
energy estimates in the dashboard.

| Column            | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `model`           | Model identifier (must match across files)                         |
| `anchor`          | Calibration configuration label (e.g. `1k_in_1k_out`)             |
| `input_tokens`    | Number of input tokens used in the measurement                     |
| `output_tokens`   | Number of output tokens used in the measurement                    |
| `energy_min_wh`   | Lower bound of measured IT energy (Wh) — source mean − std        |
| `energy_mid_wh`   | Mid-point of measured IT energy (Wh) — source mean                |
| `energy_max_wh`   | Upper bound of measured IT energy (Wh) — source mean + std        |

To add a new model, append rows for each anchor you have measurements for, then
re-run `table4_alpha_model.py` to regenerate the alpha JSON files.

### `config/baseline_params.csv`

Controls the non-AI baseline energy figure used in the Google Search column.

| Column                | Description                                          |
|-----------------------|------------------------------------------------------|
| `name`                | Internal key (must match `"google-search"` for now)  |
| `label`               | Display name shown in the dashboard                  |
| `energy_wh_per_query` | Energy per search query in Wh                        |
| `source`              | Citation source                                      |
| `year`                | Year of the source data                              |
| `notes`               | Free-text description                                |

---

## Updating Model Energy Data

The model alpha ranges (per-token energy coefficients) live in
`alpha_range_data/alpha_ranges_full_anchors.json`. This file is generated by
`table4_alpha_model.py`, which reads its input from `config/table4_ranges.csv`.

To add a new model or update energy measurements:

1. Edit `config/table4_ranges.csv` — add or modify rows for the model, one row
   per calibration anchor, with `energy_min_wh`, `energy_mid_wh`, and
   `energy_max_wh` from the source paper (typically reported as mean ± std).
2. Re-run the derivation script:
   ```bash
   python3 table4_alpha_model.py
   ```
3. The updated JSON files in `alpha_range_data/` are loaded automatically the
   next time the dashboard starts.

---

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Assumptions and Limitations

- **Alpha linearity:** The model assumes energy scales linearly with token count
  using a fixed alpha derived from a single calibration anchor. This is an
  approximation; real inference energy can vary with context length, batching,
  and hardware state.
- **PUE/WUE are averages:** Location parameters are annual averages and do not
  capture seasonal variation or time-of-day effects.
- **Google baseline is a 2009 figure:** The 0.3 Wh/query estimate pre-dates
  modern AI-enhanced search features and should be treated as a lower bound for
  contemporary plain search.
- **On-site water only:** WUE captures direct evaporative cooling at the
  datacenter. Embodied water (manufacturing, construction) and indirect water
  (power generation cooling) are excluded.
- **Per-query distributions with small samples:** The CSV upload mode computes
  footprint for each individual query. With small test sets (e.g. 5 queries),
  summary statistics should be interpreted with caution.
