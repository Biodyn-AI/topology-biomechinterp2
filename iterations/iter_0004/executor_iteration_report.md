# Executor Iteration Report — iter_0004

## Objective
Screen a broader hypothesis space by testing:
1. Cross-domain replication of persistent-homology signal (new domains vs prior lung-only evidence).
2. A distinct manifold-geometry family: intrinsic-dimensionality/local-linearity coupling to topological signal.

## Command Trace (Reproducible)
1. `conda run -n subproject40-topology python iterations/iter_0004/run_iter0004_screen.py`
2. `conda run -n subproject40-topology python -c "import pandas as pd; h1d=pd.read_csv('iterations/iter_0004/scgpt_cross_domain_h1_domain_summary.csv'); print('DOMAIN_SUMMARY'); print(h1d.to_string(index=False)); h1l=pd.read_csv('iterations/iter_0004/scgpt_cross_domain_h1_layer_summary.csv'); print('\\nLAYERS_P_LT_0.01'); print(h1l.groupby('domain').apply(lambda g:int((g.fisher_p<0.01).sum())).to_string()); print('\\nLAYER_DELTA_MEAN_MIN_MAX'); print(h1l.groupby('domain').mean_h1_sum_delta.agg(['mean','min','max']).to_string()); intr=pd.read_csv('iterations/iter_0004/intrinsic_dimensionality_h1_correlation_summary.csv'); print('\\nINTRINSIC_SUMMARY'); print(intr.to_string(index=False))"`
3. `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`

## Hypothesis Tests

### H03 (persistent_homology, cross-domain replication)
Hypothesis: The previously positive H1 persistence signal (lung) generalizes to new scGPT domains (immune and external-lung) under the same null protocol.

Method:
- Inputs: `cycle4_immune_{main,seed43,seed44}` and `cycle7_external_lung_{main,seed43,seed44}` layer embeddings.
- Per seed-layer protocol: sample 350 genes, center, PCA(20), compute ripser H1 lifetime sum.
- Null/control: 20 feature-shuffle null replicates per seed-layer.
- Aggregation: Fisher combine seed-level empirical p-values per domain-layer.

Results (from `iterations/iter_0004/scgpt_cross_domain_h1_domain_summary.csv` and `iterations/iter_0004/scgpt_cross_domain_h1_layer_summary.csv`):
- Immune:
  - Significant layers: `12/12` with Fisher `p < 0.05`; `12/12` with Fisher `p < 0.01`.
  - Mean layer H1 delta: `12.074` (min `4.323`, max `18.311`).
  - Top layer: `L7`, mean delta `18.311`, Fisher `p = 0.0056`.
- External-lung:
  - Significant layers: `12/12` with Fisher `p < 0.05`; `9/12` with Fisher `p < 0.01`.
  - Mean layer H1 delta: `12.482` (min `4.045`, max `20.855`).
  - Top layer: `L0`, mean delta `20.855`, Fisher `p = 0.0056`.

Directional interpretation:
- Positive and reproducible across both newly tested domains; large positive H1 deltas persist across all layers against the null.

### H04 (intrinsic_dimensionality, manifold-geometry coupling)
Hypothesis: Layers with stronger topological signal (H1 delta) show systematic intrinsic-dimensionality/local-linearity behavior.

Method:
- Computed per seed-layer metrics on the same sampled PCA coordinates:
  - `participation_ratio_dim`
  - `linearity_top5_ratio` (variance explained by top-5 PCs; higher implies more locally linear concentration)
  - `mle_intrinsic_dim` (Levina-Bickel kNN estimate)
- Tested layer-wise Spearman correlation between H1 delta and each metric within each seed.
- Null/control: 2,000 layer-permutation replicates per seed; Fisher combined p-values across seeds per domain-metric.

Results (from `iterations/iter_0004/intrinsic_dimensionality_h1_correlation_summary.csv`):
- External-lung:
  - `participation_ratio_dim`: mean rho `+0.508`, Fisher `p = 0.0229` (consistent sign).
  - `linearity_top5_ratio`: mean rho `-0.510`, Fisher `p = 0.0178` (consistent sign).
  - `mle_intrinsic_dim`: mean rho `+0.079`, Fisher `p = 0.936` (null-like).
- Immune:
  - `participation_ratio_dim`: mean rho `+0.266`, Fisher `p = 0.191` (not significant).
  - `linearity_top5_ratio`: mean rho `-0.242`, Fisher `p = 0.147` (not significant).
  - `mle_intrinsic_dim`: mean rho `+0.014`, Fisher `p = 0.353` (not significant).

Directional interpretation:
- Mixed/inconclusive overall. A coherent coupling exists in external-lung but does not yet replicate in immune.

## Limitations / Blockers
- No runtime blocker occurred.
- `mle_intrinsic_dim` produced unstable/undefined local estimates in some neighborhoods (distance ties/degeneracies), reducing confidence in that specific metric; interpretation should rely more on participation-ratio and linearity proxies.
- This iteration did not add direct TRRUST/GO/STRING anchoring for the newly measured intrinsic-dimensionality effects.

## Decision Summary
- H03: **Promising** (strong cross-domain replication under explicit nulls).
- H04: **Neutral/Inconclusive** (domain-specific signal, not yet robustly replicated).

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0004/scgpt_cross_domain_h1_by_seed_layer.csv`
- `iterations/iter_0004/scgpt_cross_domain_h1_layer_summary.csv`
- `iterations/iter_0004/scgpt_cross_domain_h1_domain_summary.csv`
- `iterations/iter_0004/intrinsic_dimensionality_h1_correlation_by_seed.csv`
- `iterations/iter_0004/intrinsic_dimensionality_h1_correlation_summary.csv`
- `iterations/iter_0004/iter0004_screen_summary.json`
