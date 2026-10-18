# Development Folder — Viva Demonstration

**Project:** Event-Level & Progressive Driver Attribution for Satellite-Detected
Forest Canopy Loss in Sri Lanka's Protected Forests

**Author:** C. H. Mallikarathne (NIBM / Coventry University BSc Computing)

This folder contains every implementation file, organised by pipeline step so
each stage can be run and demonstrated independently during the viva.

---

## Folder structure

```
development/
├── data/                              Shared data files used across steps
├── step1_loss_patch_identification/    Hansen GFC loss-patch extraction & zone assignment
├── step2_time_series_export/          Sentinel-2 NBR time-series export from GEE
├── step3_feature_extraction/          Per-event feature computation in GEE
├── step4_auto_dating/                 Automated loss-date estimation
├── step5_classification/              Two-stage classifier (Stage 1 rules + Stage 2 logistic)
├── step6_event_registry/              Append-only event registry with chain-union matching
├── step7_evaluation/                  E1, E2, E4 evaluation scripts
└── step8_workbench/                   Review workbench (single-page HTML + Leaflet)
```

---

## Step-by-step guide

### data/
Shared input/output data files. Key files:

| File | Description |
|------|-------------|
| `REF_30_confirmed_events.csv` | The 30 confirmed REF evaluation events |
| `REF_all_timeseries.csv` | Full Sentinel-2 NBR time series for all REF events |
| `features_REF_merged.csv` | Extracted features for REF events (input to classifiers) |
| `calibration_batch_record.csv` | Calibration batch record |
| `ref_labelling_record.md` | Full labelling record with per-event decision paths |
| `auto_dating_results_run1.csv` | Auto-dating vs hand-dating comparison |
| `hand_dates_fresh.csv` | Hand-dated FRESH events |
| `features_FRESH_group0/1.csv` | Features for FRESH (unlabelled) events |
| `features_REF_merged_workbench.csv` | Features formatted for the workbench |

### Step 1 — Loss patch identification
**Platform:** Google Earth Engine (JavaScript)

- `count_loss_patches.js` — Identifies Hansen GFC loss patches ≥0.5 ha, applies
  the 60 m water-buffer exclusion, counts patches per protected area.
- `assign_climatic_zone.js` — Assigns each event's climatic zone from the DMS
  precipitation layer.

**To run:** Paste into the GEE Code Editor and execute.

### Step 2 — Time-series export
**Platform:** Google Earth Engine (JavaScript)

- `event_timeseries_v6.js` — Core single-event NBR time-series export.
  Sentinel-2 L2A, cloud-masked, valid_pct filter.
- `REF_batch_timeseries_export.js` — Batch export for all 30 REF events.
- `REF_fresh_timeseries_export.js` — Batch export for FRESH (unlabelled) events.

**To run:** Paste into the GEE Code Editor. Output CSVs go to Google Drive.

### Step 3 — Feature extraction
**Platform:** Google Earth Engine (JavaScript)

- `feature_extraction_v1.js` — Single-event feature extraction. Three windows:
  baseline (12 months before t_forest), Stage 1 (t_loss + 60 days),
  Stage 2 (t_loss + 12 months). Computes: compactness, slope,
  dist_to_prior_loss, early_spectral_change, trend_nbr_per_day, wobble_std,
  cycle_count_provisional. Enforces the window firewall (filterDate + assertion).
- `feature_extraction_batch_REF.js` — Batch wrapper for all REF events.
- `feature_extraction_fresh.js` — Batch wrapper for FRESH events.

**To run:** Paste into the GEE Code Editor. Requires event geometries and
hand-dated t_forest/t_loss.

### Step 4 — Auto-dating
**Platform:** Python

- `auto_dating.py` — Automated loss-date estimation from NBR time series.
  Compares automated dates against hand-labelled dates.

**To run:**
```bash
cd step4_auto_dating
python auto_dating.py
```
Reads from `../data/`.

### Step 5 — Classification
**Platform:** Python

- `classifiers.py` — The two-stage classifier:
  - **Stage 1:** Fixed three-gate rule (compactness < 0.36, slope > 4.43°,
    dist_to_prior_loss > 212 m). All three must pass → candidate_natural;
    otherwise → candidate_human.
  - **Stage 2:** Logistic regression on standardised trend_nbr_per_day,
    wobble_std, cycle_count_provisional. P ≥ 0.5 → permanent_encroachment.
    Includes confidence bands.
- `stage2_loocv.py` — Leave-one-out cross-validation for Stage 2.
  Compares logistic regression against decision-tree baselines.

**To run:**
```bash
cd step5_classification
python stage2_loocv.py
```

### Step 6 — Event registry
**Platform:** Python

- `event_registry.py` — Append-only event registry with chain-union spatial
  matching. Tracks event identity across time steps.
- `test_event_registry.py` — Unit tests for the registry.

**To run:**
```bash
cd step6_event_registry
python -m pytest test_event_registry.py -v
```

### Step 7 — Evaluation
**Platform:** Python (requires scikit-learn, numpy)

- `run_e2.py` — **E2 evaluation**: per-stage and end-to-end classification
  accuracy on the frozen REF set (n=20 resolved events).
  Result: Stage 1 = 19/20, Stage 2 LOOCV = 15/19, End-to-end = 15/20.
- `e1_recheck_and_extras.py` — **E1 recheck**: detailed per-event Stage 1 and
  Stage 2 breakdown with confusion matrix, confidence bands, provenance split.
- `e4_observation_limits.py` — **E4 observation limits**: interval widths,
  clear-observation counts per window, worst-case sliding windows, monthly
  distribution.

**To run:**
```bash
cd step7_evaluation
python run_e2.py
python e1_recheck_and_extras.py
python e4_observation_limits.py
```
All three import from `../step5_classification/classifiers.py` and read data
from `../data/`.

**Dependencies:** `pip install scikit-learn numpy`

### Step 8 — Review workbench
**Platform:** HTML + JavaScript (Leaflet 1.9.4)

- `workbench.html` — The final single-page review workbench. Open in a browser.
- `workbench_template.html` — Template used to generate the workbench.
- `build_workbench_data.py` — Builds the embedded data layer.
- `make_workbench.py` — Assembles the final workbench HTML.
- `workbench_data.js` — Pre-built data for the workbench.

**To run:** Open `workbench.html` in any browser. To regenerate:
```bash
cd step8_workbench
python build_workbench_data.py
python make_workbench.py
```

---

## Dependencies

- **GEE scripts (Steps 1–3):** Google Earth Engine Code Editor (no local install)
- **Python scripts (Steps 4–7):** Python 3.8+, scikit-learn, numpy
- **Workbench (Step 8):** Any modern browser (Leaflet loaded from CDN)

Install Python dependencies:
```bash
pip install scikit-learn numpy
```

---

## Notes

- The GEE scripts (Steps 1–3) run in the browser-based Code Editor, not locally.
- All Python scripts use relative paths and can be run from their own step folder.
- The window firewall is enforced in feature extraction (Step 3): no observation
  from the label window may enter feature extraction.
- Stage 1 thresholds are fixed (in-sample); Stage 2 uses LOOCV for evaluation.
