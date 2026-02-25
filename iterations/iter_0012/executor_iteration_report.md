# Executor Iteration Report — iter_0012

## Objective
Run a breadth-first screening packet with three hypotheses from different method branches:
- `H19` confidence-stratified module anchoring (`module_structure`, new method).
- `H20` cross-model Procrustes/OT transfer (`cross_model_alignment`, new method).
- `H21` local reconstruction-error mechanistic screen (`intrinsic_dimensionality`, new method).

## Command Trace (Reproducible)
1. Installed missing dependencies in the required environment:
- `conda run -n subproject40-topology pip install transformers`
- `conda run -n subproject40-topology pip install torch`
2. Implemented iteration runner:
- `apply_patch` creating `iterations/iter_0012/run_iter0012_screen.py`
3. Syntax verification:
- `conda run -n subproject40-topology python -m py_compile iterations/iter_0012/run_iter0012_screen.py`
4. Artifact-producing run:
- `conda run -n subproject40-topology python iterations/iter_0012/run_iter0012_screen.py`
5. Quantitative extraction from artifacts:
- `conda run -n subproject40-topology python -c "import pandas as pd; d=pd.read_csv('iterations/iter_0012/h19_confidence_community_layer_split_summary.csv'); print(d.to_string(index=False))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; d=pd.read_csv('iterations/iter_0012/h20_cross_model_transfer_by_domain_layer.csv'); print(d.to_string(index=False))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; t=pd.read_csv('iterations/iter_0012/h21_local_reconstruction_trend_summary.csv'); print(t.to_string(index=False))"`
6. Paper compile after TeX update:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex` (run in `paper/`)

## Hypotheses Tested

### H19 — Confidence-Stratified Community Enrichment (N81)
- Family: `module_structure`
- Split regime: `dual_axis_disjoint`
- Method:
  - Built kNN communities on immune scGPT residual embeddings (`3` seeds, `12` layers, source/target-disjoint splits).
  - Stratified edges by DoRothEA confidence tier (`low`, `medium`, `high`) and computed within-tier same-community AUROC.
  - Tested monotonic tier slope with tier-label permutation nulls (`400` permutations per seed-layer-split).
- Artifacts:
  - `iterations/iter_0012/h19_confidence_community_by_seed_layer_split_bin.csv`
  - `iterations/iter_0012/h19_confidence_community_layer_split_summary.csv`
  - `iterations/iter_0012/h19_confidence_community_monotonicity_tests.csv`
- Quantitative results:
  - Mean AUROC slope across tiers was negative in both splits:
    - source: `-0.0771`
    - target: `-0.0627`
  - Positive-slope layers: source `0/12`, target `0/12`.
  - Fisher-significant positive slope layers: source `0/12`, target `0/12`.
  - Mean tier AUROC pattern (source): low `0.5855` > medium `0.5204` > high `0.4313`.
  - Mean tier AUROC pattern (target): low `0.5342` > medium `0.5213` > high `0.4089`.
- Interpretation:
  - The monotonic-positive confidence-scaling claim failed under this tiering definition; evidence is directionally opposite.

### H20 — Cross-Model Procrustes/OT Transfer (N78)
- Family: `cross_model_alignment`
- Split regime: `other`
- Method:
  - Loaded Geneformer token embeddings and aligned matched genes to scGPT layer embeddings for immune/lung/external-lung.
  - Ran two alignment regimes:
    - supervised Procrustes (true matched map),
    - unsupervised OT assignment (Hungarian).
  - Evaluated against random-map nulls (`300` permutations/domain):
    - top-1 retrieval,
    - kNN neighborhood Jaccard transfer,
    - transferred-edge AUROC on regulatory labels.
- Artifacts:
  - `iterations/iter_0012/h20_cross_model_transfer_by_domain_layer.csv`
  - `iterations/iter_0012/h20_cross_model_transfer_null_summary.csv`
  - `iterations/iter_0012/h20_cross_model_transfer_alignment_summary.csv`
- Quantitative results:
  - Domains tested: `3` (immune, lung, external_lung).
  - Procrustes branch:
    - mean top-1 retrieval `0.3954`, significant in `3/3` domains (`p_procrustes_top1_upper < 0.05` each).
    - mean neighborhood Jaccard `0.0631` vs null mean ~`0.01`, significant in `3/3`.
    - mean transferred-edge AUROC `0.5650`, significant in `3/3`.
    - combined Fisher p-values:
      - Jaccard upper-tail: `6.04e-06`
      - transfer AUROC upper-tail: `1.60e-05`
  - OT branch:
    - mean top-1 recovery `0.0024` (not above random; `0/3` significant).
    - mean transferred-edge AUROC `0.5200` (mixed; `1/3` significant).
- Interpretation:
  - Cross-model geometric consistency is positive for map-aware transfer, but unsupervised OT matching is currently ineffective.

### H21 — Local Reconstruction Error Mechanistic Screen (N75)
- Family: `intrinsic_dimensionality`
- Split regime: `dual_axis_disjoint`
- Method:
  - Computed per-gene local linear reconstruction error from neighborhood weights on immune scGPT embeddings (`3` seeds × `12` layers × `2` splits).
  - Built edge features from source/target reconstruction error.
  - Tested edge-level label signal with permutation nulls (`400`/row).
  - Coupled row-level reconstruction metrics to prior geodesic lift from `iter_0010` (`H13`) using seed-wise layer permutation nulls (`3000` draws).
- Artifacts:
  - `iterations/iter_0012/h21_local_reconstruction_edge_features.csv`
  - `iterations/iter_0012/h21_local_reconstruction_trend_summary.csv`
  - `iterations/iter_0012/h21_local_reconstruction_coupling_by_seed.csv`
- Quantitative results:
  - Edge-level predictive effect was split-dependent:
    - source: mean AUROC `0.5331`, Fisher upper-tail `5.64e-23`, positive in `77.8%` rows.
    - target: mean AUROC `0.4780`, Fisher upper-tail ~`1.0`, positive in `22.2%` rows.
  - Coupling to geodesic lift (`H13`) was non-positive:
    - source mean-rho for mean-edge-recon: `-0.2261` (two-sided `p=0.2029`).
    - target mean-rho for mean-edge-recon: `-0.4079` (two-sided `p=0.0190`).
- Interpretation:
  - The intended positive-trend hypothesis failed; target split shows significant inverse coupling.

## Decision Summary
- `H19` (`module_structure` confidence scaling): **negative**.
- `H20` (`cross_model_alignment` Procrustes/OT transfer): **promising (mixed)**.
- `H21` (`intrinsic_dimensionality` local reconstruction screen): **neutral/mixed**.

## Blockers / Deviations
- Blocker encountered: required packages missing in `subproject40-topology` (`transformers`, `torch`).
- Resolution: installed both packages and reran full experiment successfully.
- No remaining hard blockers for this iteration.

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0012/h19_confidence_community_by_seed_layer_split_bin.csv`
- `iterations/iter_0012/h19_confidence_community_layer_split_summary.csv`
- `iterations/iter_0012/h19_confidence_community_monotonicity_tests.csv`
- `iterations/iter_0012/h20_cross_model_transfer_by_domain_layer.csv`
- `iterations/iter_0012/h20_cross_model_transfer_null_summary.csv`
- `iterations/iter_0012/h20_cross_model_transfer_alignment_summary.csv`
- `iterations/iter_0012/h21_local_reconstruction_edge_features.csv`
- `iterations/iter_0012/h21_local_reconstruction_trend_summary.csv`
- `iterations/iter_0012/h21_local_reconstruction_coupling_by_seed.csv`
- `iterations/iter_0012/iter0012_screen_summary.json`
