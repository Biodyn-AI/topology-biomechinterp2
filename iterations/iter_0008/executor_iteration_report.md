# Executor Iteration Report — iter_0008

## Objective
Run the brainstormer-prioritized connectivity-aware stronger-null calibration branch (`N35 + N36`) in immune full-layer residual geometry:
1. test whether bridge-conditioned graph construction explains rewiring negativity,
2. test whether edge-length-quantile-constrained rewiring rescues over-adversarial null behavior.

## Command Trace (Reproducible)
1. Main experiment run:
- `conda run -n subproject40-topology python iterations/iter_0008/run_iter0008_screen.py`
2. Quantitative extraction from machine artifacts:
- `conda run -n subproject40-topology python -c "import pandas as pd; by=pd.read_csv('iterations/iter_0008/h1_immune_constrained_rewire_by_seed_layer.csv'); layer=pd.read_csv('iterations/iter_0008/h1_immune_constrained_rewire_layer_summary.csv'); passm=pd.read_csv('iterations/iter_0008/h1_immune_constrained_rewire_pass_matrix.csv'); domain=pd.read_csv('iterations/iter_0008/h1_immune_constrained_rewire_domain_summary.csv'); paired=pd.read_csv('iterations/iter_0008/h1_immune_constrained_rewire_paired_shift_summary.csv'); print('ROWS',len(by),len(layer),len(passm)); print('H1_SIG_UNCON',int(((layer.null_family=='degree_preserving_geodesic_rewire')&(layer.fisher_p_h1_upper<0.05)).sum()),'H1_SIG_QC',int(((layer.null_family=='quantile_constrained_geodesic_rewire')&(layer.fisher_p_h1_upper<0.05)).sum())); print('DUAL_PASS_UNCON',int(passm[passm.null_family=='degree_preserving_geodesic_rewire'].both_splits_sig_h1.sum()),'DUAL_PASS_QC',int(passm[passm.null_family=='quantile_constrained_geodesic_rewire'].both_splits_sig_h1.sum())); print('MEAN_DELTA_UNCON',round(float(layer[layer.null_family=='degree_preserving_geodesic_rewire'].mean_h1_delta_observed_minus_null.mean()),3),'MEAN_DELTA_QC',round(float(layer[layer.null_family=='quantile_constrained_geodesic_rewire'].mean_h1_delta_observed_minus_null.mean()),3)); print('DIST_MINP_UNCON',float(layer[layer.null_family=='degree_preserving_geodesic_rewire'].fisher_p_distortion_lower.min()),'DIST_MINP_QC',float(layer[layer.null_family=='quantile_constrained_geodesic_rewire'].fisher_p_distortion_lower.min())); src=domain[(domain.split_regime=='source_disjoint')].set_index('null_family'); tgt=domain[(domain.split_regime=='target_disjoint')].set_index('null_family'); print('EDGE_L1_SOURCE',round(float(src.loc['degree_preserving_geodesic_rewire','mean_edge_hist_l1_ratio']),4),round(float(src.loc['quantile_constrained_geodesic_rewire','mean_edge_hist_l1_ratio']),4)); print('EDGE_L1_TARGET',round(float(tgt.loc['degree_preserving_geodesic_rewire','mean_edge_hist_l1_ratio']),4),round(float(tgt.loc['quantile_constrained_geodesic_rewire','mean_edge_hist_l1_ratio']),4)); print('PAIRED_SHIFT',paired[['split_regime','mean_h1_delta_shift_constrained_minus_unconstrained','frac_rows_h1_shift_positive','mean_distortion_delta_shift_constrained_minus_unconstrained']].to_dict(orient='records'))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; by=pd.read_csv('iterations/iter_0008/h1_immune_constrained_rewire_by_seed_layer.csv'); u=by[['seed_tag','split_regime','layer','used_component_bridging','knn_k','knn_bucket']].drop_duplicates(); print('UNIQUE_ROWS',len(u)); print('BRIDGE_BY_SPLIT',u.groupby('split_regime')['used_component_bridging'].agg(['sum','count']).to_dict()); print('K_BUCKET_BY_SPLIT',u.groupby(['split_regime','knn_bucket']).size().to_dict())"`
3. Paper compile:
- `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`

## Experiment Definition
- Script: `iterations/iter_0008/run_iter0008_screen.py`
- Domain/runs: immune (`seed42_main`, `seed43`, `seed44`)
- Layers: all `12` (`0..11`)
- Split regimes:
  - `source_disjoint`
  - `target_disjoint`
- Per test unit:
  - sample `150` genes
  - PCA(`14`) after centering
  - connected symmetric kNN graph (`k` adaptive in `[10, 40]`, bridge fallback enabled)
  - observed metrics from geodesic graph distances:
    - `h1_observed_geodesic`
    - `distortion_observed` (mean geodesic/euclidean ratio)
- Null families (metric-matched geodesic comparison):
  - `degree_preserving_geodesic_rewire`
  - `quantile_constrained_geodesic_rewire` (best-of-16 connected rewires per draw, minimizing edge-length-quantile histogram drift)
- Null draws per family per seed-layer-split: `5`

## Quantitative Results

### H11 — Bridge-conditioned rewiring negativity (`N35`; family: `graph_topology`)
Primary artifacts:
- `iterations/iter_0008/h1_immune_constrained_rewire_bridge_gap_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_bridge_k_strata_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_by_seed_layer.csv`

Direct evidence:
- Unique seed-layer-split rows: `72`.
- Bridge usage was highly split-skewed:
  - source: `36/36` bridged, all in `k_gt_30`.
  - target: `2/36` bridged; k-buckets: `k_le_20: 3`, `k_21_30: 13`, `k_gt_30: 20`.
- Raw bridge-gap summary (all rows pooled) did **not** show bridged rows being more negative:
  - unconstrained: bridge minus non-bridge H1 delta gap `+16.147`.
  - quantile-constrained: bridge minus non-bridge H1 delta gap `+17.496`.

Interpretation:
- The directional pattern is opposite to the `N35` expectation, but direct causal interpretation is limited because bridge status is almost perfectly confounded with split regime (source mostly/all bridged, target mostly non-bridged).

### H12 — Edge-length-quantile constrained rewiring calibration (`N36`; family: `null_sensitivity`)
Primary artifacts:
- `iterations/iter_0008/h1_immune_constrained_rewire_layer_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_domain_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_pass_matrix.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_paired_shift_summary.csv`

Direct evidence:
- Tested units: `144` by-seed rows, `48` layer-split-family aggregates, `24` layer-split pass-matrix rows.
- H1 support remained uniformly absent under both nulls:
  - unconstrained significant layer-split tests: `0/24`.
  - quantile-constrained significant layer-split tests: `0/24`.
  - dual-split layer passes for H1: `0/12` in both null families.
- Mean H1 deltas stayed negative and did not improve overall:
  - unconstrained: `-19.244`.
  - quantile-constrained: `-19.532`.
- Distortion lower-tail branch remained non-significant:
  - `0/24` significant in both families.
  - minimum Fisher p for distortion: `0.0964` (both families).
- Edge-length histogram drift from base graph:
  - source split L1 drift mean: `0.3249` (unconstrained) vs `0.3129` (quantile-constrained).
  - target split L1 drift mean: `0.1461` (unconstrained) vs `0.1466` (quantile-constrained).
- Paired shift (quantile-constrained minus unconstrained):
  - source: mean H1 shift `+0.556` (61.1% rows positive), mean distortion shift `+0.00123`.
  - target: mean H1 shift `-1.132` (41.7% rows positive), mean distortion shift `-0.000013`.

Interpretation:
- The constrained null produced only marginal edge-length-histogram calibration change and did not recover positive or significant topology survival under rewiring. The rewiring-survival branch remains non-supportive in immune for this regime.

## Limitations and Blockers
- No data/runtime blocker.
- Identifiability blocker for `N35`: bridge vs non-bridge comparison is confounded by split regime in this run (`source 36/36 bridged`, `target 2/36 bridged`).
- Quantile-constrained rewiring is approximate (best-of-candidates minimization), not exact bin-preserving rewiring.

## Decision Summary
- H11 (`graph_topology`, bridge-conditioned explanation): **Inconclusive/partial** due split-confounded strata; no supportive directional signal observed.
- H12 (`null_sensitivity`, quantile-constrained rewiring rescue): **Negative**.

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0008/h1_immune_constrained_rewire_by_seed_layer.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_layer_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_pass_matrix.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_domain_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_bridge_k_strata_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_bridge_gap_summary.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_paired_shift_by_seed_layer.csv`
- `iterations/iter_0008/h1_immune_constrained_rewire_paired_shift_summary.csv`
- `iterations/iter_0008/iter0008_screen_summary.json`
