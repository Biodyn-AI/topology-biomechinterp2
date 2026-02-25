# Executor Iteration Report — iter_0007

## Objective
Run the brainstormer-prioritized metric-matched null calibration test (`N23`) for immune full-layer topology: compare observed geodesic persistence against a degree-preserving rewired-geodesic null, and quantify whether this rescues the fully negative rewiring result from `iter_0006`.

## Command Trace (Reproducible)
1. Main experiment run:
   - `conda run -n subproject40-topology python iterations/iter_0007/run_iter0007_screen.py`
2. Quantitative extraction from generated machine artifacts:
   - `conda run -n subproject40-topology python -c "import pandas as pd; by=pd.read_csv('iterations/iter_0007/h1_immune_metric_matched_by_seed_layer.csv'); layer=pd.read_csv('iterations/iter_0007/h1_immune_metric_matched_layer_summary.csv'); passm=pd.read_csv('iterations/iter_0007/h1_immune_metric_matched_pass_matrix.csv'); print('ROWS',len(by),len(layer)); print('GEO_SIG',int((layer.fisher_p_geodesic_vs_rewire<0.05).sum()),'EUCLID_SIG',int((layer.fisher_p_euclidean_vs_rewire<0.05).sum())); print('MEAN_DELTA_GEO',round(float(layer.mean_h1_delta_geodesic_vs_rewire.mean()),3),'MEAN_DELTA_EU',round(float(layer.mean_h1_delta_euclidean_vs_rewire.mean()),3),'MEAN_SHIFT',round(float(layer.mean_h1_delta_shift_geodesic_minus_euclidean.mean()),3)); print('MIN_P_GEO',float(layer.fisher_p_geodesic_vs_rewire.min())); print('BOTH_SPLITS_SIG_GEO',int(passm.both_splits_sig_geodesic.sum())); print('BRIDGED',int(by.used_component_bridging.sum()),'OF',len(by),'K_MEAN',round(float(by.knn_k.mean()),3));"`
   - `conda run -n subproject40-topology python -c "import pandas as pd; layer=pd.read_csv('iterations/iter_0007/h1_immune_metric_matched_layer_summary.csv'); print('DISTORTION_SIG',int((layer.fisher_p_distortion_lower<0.05).sum()),'MIN_P',float(layer.fisher_p_distortion_lower.min()),'MEAN_DELTA',round(float(layer.mean_distortion_delta_observed_minus_null.mean()),3));"`
3. Paper update compile:
   - `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`

## Experiment Definition
- Script: `iterations/iter_0007/run_iter0007_screen.py`
- Domain: `immune` (seeds: `seed42_main`, `seed43`, `seed44`)
- Layers: all 12 (`0..11`)
- Split regimes:
  - `source_disjoint`
  - `target_disjoint`
- Per test unit:
  - sample `160` genes
  - center embeddings and PCA(`14`)
  - build connected symmetric kNN graph (`k` adaptive in `[12, 30]`, bridge fallback enabled)
  - compute two observed topology metrics:
    - `h1_observed_euclidean` (legacy comparator)
    - `h1_observed_geodesic` (metric-matched comparator)
- Null family:
  - `degree_preserving_rewire_geodesic` (`6` null draws per seed-layer-split)
  - each null draw rewires graph with degree-preserving swaps and re-computes geodesic distances
- Aggregation:
  - empirical permutation p-values per seed-layer-split
  - Fisher-combined p-values across seeds
  - layer-level dual-split pass matrix

## Quantitative Results

### H09 — Metric-matched rewiring null calibration (family: `null_sensitivity`)
Primary artifacts:
- `iterations/iter_0007/h1_immune_metric_matched_layer_summary.csv`
- `iterations/iter_0007/h1_immune_metric_matched_domain_summary.csv`
- `iterations/iter_0007/h1_immune_metric_matched_pass_matrix.csv`

Direct evidence:
- Total tested units: `72` by-seed rows, `24` layer-split aggregates.
- Geodesic-vs-rewire support: `0/24` significant (`Fisher p < 0.05`), minimum geodesic Fisher p = `0.6913`.
- Euclidean-vs-rewire support: `0/24` significant (same minimum p).
- Mean H1 deltas remain strongly negative:
  - geodesic-vs-rewire: `-95.356`
  - euclidean-vs-rewire: `-95.536`
- Dual-split geodesic pass rate: `0/12` layers (`both_splits_sig_geodesic`).
- Split-level domain summary:
  - source: `0/12` geodesic-significant, mean geodesic delta `-105.581`
  - target: `0/12` geodesic-significant, mean geodesic delta `-85.130`

Inference:
- Metric matching does not rescue the rewiring branch. The stronger rewiring null remains decisively non-supportive in immune across all layers and both split regimes.

### H10 — Geodesic-vs-Euclidean calibration shift diagnostic (family: `manifold_distance`)
Primary artifacts:
- `iterations/iter_0007/h1_immune_metric_calibration_shift_summary.csv`
- `iterations/iter_0007/h1_immune_metric_matched_layer_summary.csv`

Direct evidence:
- Mean calibration shift (`delta_geodesic - delta_euclidean`) is positive but small: `+0.180` across layer-split aggregates.
- Layer-split shift direction: positive in `22/24` layer-split aggregates (`91.7%`).
- Split-wise mean shift:
  - source: `+0.150`
  - target: `+0.210`
- Distortion branch (observed lower distortion than rewired null) was not significant:
  - `0/24` significant with minimum Fisher p = `0.0696`
  - mean distortion delta (observed - null) = `+0.105`

Inference:
- There is a directional calibration shift, but its magnitude is negligible relative to the large negative rewiring deltas. This supports a "small mismatch contribution" interpretation, not a primary-cause explanation.

## Diagnostics / Runtime Notes
- Connectivity pressure remains high:
  - bridge fallback used in `61/72` by-seed rows (`84.7%`)
  - mean `k` used: `29.181` (min `13`, max `30`)
- Rewiring draw generation was stable in this run:
  - mean rewiring attempts per row: `6.0`
  - max rewiring attempts per row: `6`

## Limitations
- This iteration is immune-only; no external-lung/lung replication yet for the calibrated null branch.
- Null draw count (`6`) is intentionally bounded for fast screening; very small effects could require more draws for tighter uncertainty bounds.

## Blockers
- No data/runtime blocker in this iteration.

## Decision Summary
- H09 (`null_sensitivity`, metric-matched rewiring calibration): **Negative**.
- H10 (`manifold_distance`, calibration-shift diagnostic): **Inconclusive/neutral** (small directional shift, no null-surviving rescue signal).

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0007/h1_immune_metric_matched_by_seed_layer.csv`
- `iterations/iter_0007/h1_immune_metric_matched_layer_summary.csv`
- `iterations/iter_0007/h1_immune_metric_matched_pass_matrix.csv`
- `iterations/iter_0007/h1_immune_metric_matched_domain_summary.csv`
- `iterations/iter_0007/h1_immune_metric_calibration_shift_summary.csv`
- `iterations/iter_0007/iter0007_screen_summary.json`
