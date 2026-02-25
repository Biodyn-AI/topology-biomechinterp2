# Executor Iteration Report — iter_0010

## Objective
Screen a broad, non-retired hypothesis portfolio for geometric/topological structure in scGPT/Geneformer with bounded, reproducible tests and explicit null controls.

## Command Trace (Reproducible)
1. Implemented screen runner:
- `cat > iterations/iter_0010/run_iter0010_screen.py <<'PY' ... PY`
2. Main experiment run:
- `conda run -n subproject40-topology python iterations/iter_0010/run_iter0010_screen.py`
3. Method correction + rerun (H14 null-strength fix):
- `apply_patch` on `iterations/iter_0010/run_iter0010_screen.py` (`n_null: 3 -> 24`)
- `conda run -n subproject40-topology python iterations/iter_0010/run_iter0010_screen.py`
4. Quantitative extraction from machine artifacts:
- `conda run -n subproject40-topology python -c "import pandas as pd; s=pd.read_csv('iterations/iter_0010/h13_manifold_distance_split_summary.csv'); l=pd.read_csv('iterations/iter_0010/h13_manifold_distance_layer_summary.csv'); p=pd.read_csv('iterations/iter_0010/h13_manifold_distance_pass_matrix.csv'); print('H13_SPLIT',s.to_dict(orient='records')); print('H13_DUAL_POS',int(p.both_splits_positive_mean_delta.sum()),'OF',len(p),'DUAL_SIG',int(p.both_splits_fisher_sig_upper.sum())); print('H13_LAYER_DELTA_RANGE',float(l.mean_delta_auc_geodesic_minus_euclidean.min()),float(l.mean_delta_auc_geodesic_minus_euclidean.max()));"`
- `conda run -n subproject40-topology python -c "import pandas as pd; lay=pd.read_csv('iterations/iter_0010/h14_topology_stability_layer_summary.csv'); fil=pd.read_csv('iterations/iter_0010/h14_topology_stability_filtration_layer_summary.csv'); print('H14_LAYER_HEAD',lay[['layer','mean_h1_delta','combined_fisher_p_h1_upper','delta_positive_fraction','mean_cv_h1_delta']].head(12).to_dict(orient='records')); print('H14_ALL_POS_LAYERS',int((lay.mean_h1_delta>0).sum()),'SIG',int((lay.combined_fisher_p_h1_upper<0.05).sum())); print('H14_RANGE_STATS',float(fil.mean_delta_range.mean()),float(fil.mean_delta_range.max()),float(fil.all_settings_positive_fraction.mean()));"`
- `conda run -n subproject40-topology python -c "import pandas as pd; t=pd.read_csv('iterations/iter_0010/h15_cross_model_disagreement_trend.csv'); print('H15_ROWS',t.to_dict(orient='records')); print('NEG_DOMAINS',int((t.spearman_rho_disagreement_vs_positive_rate<0).sum())); print('SIG2SIDED',int((t.p_two_sided<0.05).sum()));"`

## Hypotheses Tested

### H13 — Geodesic Manifold Advantage for Regulatory Edge Discrimination
- Family: `manifold_distance`
- Split regime: `source_disjoint` + `target_disjoint` (dual-axis packet)
- Method:
  - Immune scGPT residual embeddings (`3` seeds, `12` layers), PCA(14), connected kNN geodesics (`k` adaptive in `[10,35]` with bridge fallback).
  - Compared AUROC from `-geodesic_distance` vs `-euclidean_distance` on labeled TF-target edges.
  - Null: label-permutation for delta-AUROC (`200` permutations per seed-layer-split row).
- Primary artifacts:
  - `iterations/iter_0010/h13_manifold_distance_by_seed_layer_split.csv`
  - `iterations/iter_0010/h13_manifold_distance_layer_summary.csv`
  - `iterations/iter_0010/h13_manifold_distance_split_summary.csv`
  - `iterations/iter_0010/h13_manifold_distance_pass_matrix.csv`
- Quantitative results:
  - Rows tested: `72` (3 seeds × 12 layers × 2 splits).
  - Mean geodesic-minus-euclidean AUROC delta:
    - source split: `+0.00519`
    - target split: `+0.01319`
  - Layer significance (Fisher upper-tail p<0.05):
    - source: `7/12`
    - target: `11/12`
  - Dual-split consistency:
    - positive mean delta in both splits: `12/12` layers
    - significant in both splits: `7/12` layers
  - Delta range across layer-split summaries: `[+0.00118, +0.02154]`.
- Interpretation:
  - Evidence: geodesic manifold distances consistently outperform Euclidean distances for immune regulatory-edge discrimination.
  - Inference: non-Euclidean neighborhood structure is functionally relevant to regulatory proximity in this regime.

### H14 — Bootstrap + Filtration Stability of H1 Persistence Signal
- Family: `topology_stability` (new family)
- Split regime: `other` (filtration/bootstrapping perturbation, not edge split)
- Method:
  - Immune scGPT residual embeddings (`3` seeds, `12` layers).
  - Bootstrap persistence packet over settings: sample size `{120,180}` × PCA dim `{10,14}` with `4` bootstraps each.
  - Null: feature-shuffle H1 with `24` draws per bootstrap replicate.
- Primary artifacts:
  - `iterations/iter_0010/h14_topology_stability_bootstrap_records.csv`
  - `iterations/iter_0010/h14_topology_stability_seed_layer_setting_summary.csv`
  - `iterations/iter_0010/h14_topology_stability_layer_summary.csv`
  - `iterations/iter_0010/h14_topology_stability_filtration_sensitivity.csv`
  - `iterations/iter_0010/h14_topology_stability_filtration_layer_summary.csv`
- Quantitative results:
  - Bootstrap rows: `576`; seed-layer-setting summaries: `144`.
  - Mean layer H1 delta (observed minus feature-shuffle null): `+3.870`.
  - Layers with positive mean delta: `12/12`.
  - Layers with combined Fisher upper-tail p<0.05: `12/12`.
  - Filtration robustness:
    - all-settings-positive fraction per layer mean: `1.0`
    - mean delta range across settings: `3.285` (max `4.722`).
- Interpretation:
  - Evidence: topology signal remains directionally stable under bootstrap and filtration perturbations.
  - Inference: previously observed persistent-homology signal is robust rather than a single-setup artifact.

### H15 — Cross-Model Disagreement Trend vs Regulatory Positive Rate
- Family: `cross_model_alignment`
- Split regime: `edge_stratified` (disagreement quantile bins)
- Method:
  - Used cross-model disagreement strata (`10` bins/domain) from cycle15 summaries.
  - Tested Spearman trend between `mean_abs_disagreement` and `positive_rate` per domain.
  - Null: within-domain permutation (`3000` draws/domain).
- Primary artifacts:
  - `iterations/iter_0010/h15_cross_model_disagreement_trend.csv`
  - `iterations/iter_0010/h15_cross_model_disagreement_summary.json`
- Quantitative results:
  - Domain Spearman rho:
    - lung: `-0.9758`, two-sided p `3.33e-4`
    - external_lung: `-0.5030`, p `0.1406`
    - immune: `+0.4012`, p `0.2496`
  - Domains with negative rho: `2/3`.
  - Combined Fisher p-values:
    - two-sided: `8.99e-4`
    - negative-tail: `1.41e-3`
- Interpretation:
  - Evidence: disagreement-versus-positivity trend is domain-heterogeneous; strong negative trend is driven by lung.
  - Inference: cross-model consistency is context-dependent, with no single universal direction across all domains.

## Decision Summary
- `H13` (`manifold_distance`): **promising**.
- `H14` (`topology_stability`): **promising**.
- `H15` (`cross_model_alignment`): **neutral/mixed** (domain-conditional).

## Blockers / Deviations
- No data/runtime blocker.
- Method correction applied mid-iteration: strengthened H14 null draw count (`3 -> 24`) to make empirical p-values decision-capable; reran full screen after fix.

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0010/h13_manifold_distance_by_seed_layer_split.csv`
- `iterations/iter_0010/h13_manifold_distance_layer_summary.csv`
- `iterations/iter_0010/h13_manifold_distance_split_summary.csv`
- `iterations/iter_0010/h13_manifold_distance_pass_matrix.csv`
- `iterations/iter_0010/h14_topology_stability_bootstrap_records.csv`
- `iterations/iter_0010/h14_topology_stability_seed_layer_setting_summary.csv`
- `iterations/iter_0010/h14_topology_stability_layer_summary.csv`
- `iterations/iter_0010/h14_topology_stability_filtration_sensitivity.csv`
- `iterations/iter_0010/h14_topology_stability_filtration_layer_summary.csv`
- `iterations/iter_0010/h15_cross_model_disagreement_trend.csv`
- `iterations/iter_0010/h15_cross_model_disagreement_summary.json`
- `iterations/iter_0010/iter0010_screen_summary.json`
