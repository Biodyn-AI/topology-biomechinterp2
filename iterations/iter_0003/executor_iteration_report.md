# Executor Iteration Report — iter_0003

## Objective
Screen for geometric/topological structure with fast null-controlled tests, prioritizing breadth over narrow optimization.

## Command Trace (Reproducible)
1. `conda run -n subproject40-topology python iterations/iter_0003/run_iter0003_screen.py`
2. `conda run -n subproject40-topology python -c "import pandas as pd; df=pd.read_csv('iterations/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv'); print('layers_p_lt_0.01',int((df['fisher_p']<0.01).sum())); print('layers_p_lt_0.05',int((df['fisher_p']<0.05).sum())); print('mean_delta',df['mean_h1_sum_delta'].mean()); print('min_delta',df['mean_h1_sum_delta'].min()); print('max_delta',df['mean_h1_sum_delta'].max());"`
3. `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`

## Hypothesis Tests

### H01 (persistent_homology)
Hypothesis: scGPT lung residual-gene point clouds contain nontrivial loop structure (H1 persistence) exceeding a feature-shuffle null.

- Input embeddings: 3 replicate runs (seed42/43/44) from:
  - `../subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_main/layer_gene_embeddings.npy`
  - `../subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_seed43/layer_gene_embeddings.npy`
  - `../subproject_38_geometric_residual_stream_interpretability/implementation/outputs/cycle6_lung_seed44/layer_gene_embeddings.npy`
- Setup: 350 genes/layer sampled, PCA(20), ripser H1 summaries, 20 feature-shuffle null replicates per seed-layer.

Results (from `iterations/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv`):
- Top layer: L0 with mean H1 delta = `18.603`, mean z = `3.213`, Fisher combined p = `0.0056`.
- Robustness across layers: `11/12` layers with Fisher p `< 0.05`, `9/12` layers with Fisher p `< 0.01`.
- Effect distribution: mean layer delta `10.592` (min `2.859`, max `18.603`).
- Weakest layer: L11 (delta `2.859`, Fisher p `0.064`).

Directional interpretation:
- Strong positive signal: observed scGPT geometry shows persistent H1 structure above a structure-destroying null across most layers.

Artifacts:
- `iterations/iter_0003/scgpt_lung_h1_persistence_by_seed_layer.csv`
- `iterations/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv`

### H02 (cross_model_alignment)
Hypothesis: feature-effect geometry profiles (delta AUROC by geometric feature) align between scGPT and Geneformer across domains.

- Domains: immune, lung, external_lung.
- Features aligned: `centered_cosine`, `dot`, `cosine`.
- Null: exact permutation null over feature index assignments (3! permutations/domain).

Results (from `iterations/iter_0003/cross_model_feature_alignment_by_domain.csv` and `iterations/iter_0003/cross_model_feature_alignment_summary.json`):
- Mean Spearman rho = `0.833`.
- Mean cosine similarity = `0.825` (min `0.730`, max `0.998`).
- Combined permutation p-values:
  - Spearman Fisher combined p = `0.409`
  - Cosine Fisher combined p = `0.349`

Directional interpretation:
- Inconclusive: directional similarity exists, but null-calibrated significance is weak given very low-dimensional feature vectors.

Artifacts:
- `iterations/iter_0003/cross_model_feature_alignment_by_domain.csv`
- `iterations/iter_0003/cross_model_feature_alignment_summary.json`

## Blockers / Limitations
- Residual-level cross-model alignment remains limited by currently accessible Geneformer artifacts in this workspace (feature summaries available; residual token/gene embedding tensors not surfaced in subproject_40 artifacts).
- Fallback executed this iteration: feature-profile alignment with exact permutation null instead of direct residual manifold alignment.

## Decision Summary
- H01: **Promising** (clear positive with null robustness).
- H02: **Inconclusive** (directional but not null-significant).

## Paper/Log Maintenance
- Updated cumulative log: `reports/autoloop_master_log.md`.
- Updated paper source with explicit marker `ITERATION UPDATE: iter_0003`: `paper/autoloop_research_paper.tex`.
- Compiled latest PDF successfully: `paper/autoloop_research_paper.pdf`.

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0003/scgpt_lung_h1_persistence_by_seed_layer.csv`
- `iterations/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv`
- `iterations/iter_0003/cross_model_feature_alignment_by_domain.csv`
- `iterations/iter_0003/cross_model_feature_alignment_summary.json`
- `iterations/iter_0003/iter0003_screen_summary.json`
