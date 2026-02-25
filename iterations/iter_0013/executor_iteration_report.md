# Executor Iteration Report — iter_0013

## Objective
Run a breadth-oriented 3-hypothesis screen with one refinement and two materially new methods:
- `H22` depth-phase split interaction for local reconstruction geometry (`intrinsic_dimensionality`, refinement of H21 with changed protocol).
- `H23` curvature-based graph-topology enrichment (`graph_topology`, new method).
- `H24` CCA-like cross-model manifold consistency with permutation nulls (`cross_model_alignment`, new method).

## Command Trace (Reproducible)
1. Implemented iteration runner:
- `apply_patch` creating `iterations/iter_0013/run_iter0013_screen.py`
2. Syntax check:
- `conda run -n subproject40-topology python -m py_compile iterations/iter_0013/run_iter0013_screen.py`
3. Artifact-producing run:
- `conda run -n subproject40-topology python iterations/iter_0013/run_iter0013_screen.py`
4. Quantitative extraction from machine artifacts:
- `conda run -n subproject40-topology python -c "import pandas as pd; d=pd.read_csv('iterations/iter_0013/h22_phase_transition_model_summary.csv'); print(d.to_string(index=False))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; d=pd.read_csv('iterations/iter_0013/h23_curvature_enrichment_domain_summary.csv'); print(d.to_string(index=False)); print(); s=pd.read_csv('iterations/iter_0013/h23_curvature_enrichment_split_summary.csv'); print(s.to_string(index=False))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; d=pd.read_csv('iterations/iter_0013/h24_cross_model_cca_domain_summary.csv'); print(d.to_string(index=False)); print(); o=pd.read_csv('iterations/iter_0013/h24_cross_model_cca_overall_summary.csv'); print(o.to_string(index=False))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; d=pd.read_csv('iterations/iter_0013/h22_phase_transition_by_seed_layer_split.csv'); s=d.groupby(['domain','phase','split_regime'],as_index=False)['auc_edge_reconstruction'].mean(); print(s.to_string(index=False))"`
5. Updated cumulative log:
- edited `reports/autoloop_master_log.md` with `iter_0013` outcomes and artifact list.
6. Paper update and compile:
- edited `paper/autoloop_research_paper.tex` adding section marker `ITERATION UPDATE: iter_0013`.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex` (run in `paper/`).

## Hypotheses Tested

### H22 — Cross-domain Split×Depth Phase Interaction for Local Reconstruction
- Family: `intrinsic_dimensionality`
- Split regime: `dual_axis_disjoint`
- Method:
  - Computed per-layer edge reconstruction AUROC from local linear reconstruction errors across `3` domains (`immune`, `lung`, `external_lung`), `3` seeds/domain, `12` layers, and source/target-disjoint splits.
  - Formed paired seed-layer differences (`target - source`) and tested phase interactions (`early`, `mid`, `late`) via sign-flip permutation nulls (`4000`) plus bootstrap CIs (`2000`).
- Artifacts:
  - `iterations/iter_0013/h22_phase_transition_by_seed_layer_split.csv`
  - `iterations/iter_0013/h22_phase_transition_phase_means.csv`
  - `iterations/iter_0013/h22_phase_transition_model_summary.csv`
  - `iterations/iter_0013/h22_phase_transition_null_summary.csv`
- Quantitative results:
  - Rows: `216` seed-layer-split rows (`108` paired rows).
  - Immune shows significant late negative split difference:
    - late mean diff (target-source) `-0.02153`, 95% CI `[-0.03137, -0.01272]`, lower-tail `p=0.00075`.
  - Lung shows weaker late negative tendency:
    - late mean diff `+0.00370`, interaction late-vs-early `-0.00918`, lower-tail `p=0.09148`.
  - External-lung shows opposite direction:
    - late mean diff `+0.01123`, lower-tail `p=0.99575`.
  - Promotion gate check (negative late interaction in >=2 domains) was not met (`1/3` domains for late negative mean; `0/3` for significant negative late-vs-early interaction).
- Interpretation:
  - Depth-phase asymmetry is real in immune but not cross-domain robust; this is currently a domain-conditional pattern, not a generalizable mechanism.

### H23 — Forman Curvature Enrichment for Regulatory Edges
- Family: `graph_topology`
- Split regime: `dual_axis_disjoint`
- Method:
  - Built layer-wise kNN graphs (layers `0,3,7,11`) and computed unweighted Forman curvature on graph edges.
  - Tested whether higher negative curvature predicts positive regulatory labels using AUROC and top-vs-bottom curvature-bin positive-rate deltas.
  - Used label-permutation nulls (`300` per seed-layer-split row).
- Artifacts:
  - `iterations/iter_0013/h23_curvature_enrichment_by_seed_layer_split.csv`
  - `iterations/iter_0013/h23_curvature_enrichment_split_summary.csv`
  - `iterations/iter_0013/h23_curvature_enrichment_domain_summary.csv`
- Quantitative results:
  - Rows: `72` seed-layer-split tests.
  - Domain mean AUROC for negative-curvature score was below chance in all domains:
    - immune `0.3406`, lung `0.3894`, external-lung `0.3905`.
  - Top-minus-bottom curvature-bin positive-rate delta was negative in all domains:
    - immune `-0.3062`, lung `-0.1826`, external-lung `-0.1941`.
  - No domain showed supportive Fisher upper-tail significance (`0/3` for AUROC, `0/3` for delta).
- Interpretation:
  - Evidence is decisively opposite the hypothesis in this configuration; negative curvature does not enrich true regulatory edges here.

### H24 — CCA-like Cross-model Geometric Consistency
- Family: `cross_model_alignment`
- Split regime: `other`
- Method:
  - Used matched genes (`n=320` per domain) for scGPT vs Geneformer.
  - Reduced each model with PCA(48), then applied linear CCA-like whitening/SVD alignment (`cca_dim=20`).
  - Measured canonical correlation, pairwise-distance Spearman, kNN neighborhood Jaccard, and top-1 cross-space retrieval.
  - Compared to correspondence-permuted nulls (`160` permutations/domain) and to a no-CCA PCA baseline.
- Artifacts:
  - `iterations/iter_0013/h24_cross_model_cca_domain_summary.csv`
  - `iterations/iter_0013/h24_cross_model_cca_null_summary.csv`
  - `iterations/iter_0013/h24_cross_model_cca_overall_summary.csv`
- Quantitative results:
  - Domains tested: `3/3`.
  - Mean canonical correlation: `0.7968`.
  - Distance structure agreement (CCA): mean Spearman `0.7466` (`3/3` significant; combined Fisher `p=3.17e-05`).
  - kNN neighborhood overlap (CCA): mean Jaccard `0.1714` (`3/3` significant; combined Fisher `p=3.17e-05`).
  - Top-1 retrieval (CCA): mean `0.7229` (`3/3` significant; combined Fisher `p=3.17e-05`).
  - CCA improved over PCA baseline in all domains for all three transfer metrics (distance/Jaccard/top-1 deltas all positive).
- Interpretation:
  - Strong, reproducible cross-model geometric consistency signal with non-trivial effect size and robust permutation separation.

## Decision Summary
- `H22` (`intrinsic_dimensionality` depth-phase interaction): **neutral** (domain-conditional, fails >=2-domain robustness gate).
- `H23` (`graph_topology` curvature enrichment): **negative**.
- `H24` (`cross_model_alignment` CCA-like consistency): **promising**.

## Blockers / Deviations
- No hard data/runtime blocker.
- Runtime note: Geneformer model loading emits expected checkpoint compatibility warnings (`UNEXPECTED` head/prediction params, `MISSING` pooler params) but embedding extraction completed and all outputs were produced.

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0013/h22_phase_transition_by_seed_layer_split.csv`
- `iterations/iter_0013/h22_phase_transition_phase_means.csv`
- `iterations/iter_0013/h22_phase_transition_model_summary.csv`
- `iterations/iter_0013/h22_phase_transition_null_summary.csv`
- `iterations/iter_0013/h23_curvature_enrichment_by_seed_layer_split.csv`
- `iterations/iter_0013/h23_curvature_enrichment_split_summary.csv`
- `iterations/iter_0013/h23_curvature_enrichment_domain_summary.csv`
- `iterations/iter_0013/h24_cross_model_cca_domain_summary.csv`
- `iterations/iter_0013/h24_cross_model_cca_null_summary.csv`
- `iterations/iter_0013/h24_cross_model_cca_overall_summary.csv`
- `iterations/iter_0013/iter0013_screen_summary.json`
