# Executor Iteration Report — iter_0011

## Objective
Run a breadth-first screening packet with one new family and two materially changed methods: community/module topology anchoring, cross-model feature transfer, and intrinsic-dimensionality coupling to manifold gain.

## Command Trace (Reproducible)
1. Implemented screen runner:
- `apply_patch` creating `iterations/iter_0011/run_iter0011_screen.py`
2. Initial execution:
- `conda run -n subproject40-topology python iterations/iter_0011/run_iter0011_screen.py`
3. Reproducibility fix:
- `apply_patch` on `iterations/iter_0011/run_iter0011_screen.py` replacing non-deterministic `hash(...)` RNG seeding with fixed seed lookup in H18.
4. Final execution (artifact-producing run):
- `conda run -n subproject40-topology python iterations/iter_0011/run_iter0011_screen.py`
5. Quantitative extraction from machine artifacts:
- `conda run -n subproject40-topology python -c "import pandas as pd; s=pd.read_csv('iterations/iter_0011/h16_module_structure_split_summary.csv'); print(s.to_dict(orient='records'))"`
- `conda run -n subproject40-topology python -c "import pandas as pd, json; d=pd.read_csv('iterations/iter_0011/h17_cross_model_transfer_domain_summary.csv'); print(d.to_dict(orient='records')); print(json.load(open('iterations/iter_0011/h17_cross_model_transfer_summary.json')))"`
- `conda run -n subproject40-topology python -c "import pandas as pd; s=pd.read_csv('iterations/iter_0011/h18_intrinsic_geodesic_coupling_summary.csv'); print(s.to_dict(orient='records'))"`
6. Paper compile after TeX update:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex` (run in `paper/`)

## Hypotheses Tested

### H16 — Community Co-membership Enrichment for Regulatory Edges
- Family: `module_structure`
- Split regime: `dual_axis_disjoint`
- Method:
  - Immune scGPT residual embeddings (`3` seeds, `12` layers, source/target-disjoint splits).
  - Built kNN graphs (`k=20`) on PCA(14) gene embeddings per seed-layer-split.
  - Detected graph communities (`greedy_modularity_communities`) and scored each edge by same-community membership.
  - Primary metrics: AUROC of same-community indicator for edge label; positive-rate gap (same-community minus cross-community).
  - Null control: label permutation (`300` draws per seed-layer-split).
- Primary artifacts:
  - `iterations/iter_0011/h16_module_structure_by_seed_layer_split.csv`
  - `iterations/iter_0011/h16_module_structure_layer_summary.csv`
  - `iterations/iter_0011/h16_module_structure_split_summary.csv`
- Quantitative results:
  - Tested rows: `72` (3 seeds × 12 layers × 2 splits).
  - Mean AUROC (same-community indicator):
    - source split: `0.5387`
    - target split: `0.5413`
  - Mean positive-rate gap (same minus different community):
    - source split: `+0.0727`
    - target split: `+0.0778`
  - Layer robustness:
    - AUROC > 0.5 in `12/12` layers for each split.
    - Fisher-combined upper-tail significance in `12/12` layers for both AUROC and rate-gap in each split.
  - Mean graph modularity:
    - source `0.3893`, target `0.2971`.
- Interpretation:
  - Evidence supports biologically anchored community structure: TRRUST-labeled positives are consistently concentrated within learned geometric communities.

### H17 — Cross-model Feature-Ranking Transfer (Rescue with Changed Method)
- Family: `cross_model_alignment`
- Split regime: `other`
- Method:
  - Compared scGPT vs Geneformer feature deltas on shared geometric features (`centered_cosine`, `dot`, `cosine`) across immune/lung/external_lung.
  - Domain-level metric: Spearman rank correlation of feature deltas.
  - Transfer check: whether the top scGPT feature matches top Geneformer feature.
  - Null control: exact feature-index permutation.
    - Per-domain exact null over `3! = 6` permutations.
    - Global exact null over independent per-domain permutations (`6^3 = 216` combinations).
- Primary artifacts:
  - `iterations/iter_0011/h17_cross_model_transfer_domain_summary.csv`
  - `iterations/iter_0011/h17_cross_model_transfer_global_null.csv`
  - `iterations/iter_0011/h17_cross_model_transfer_summary.json`
- Quantitative results:
  - Domain Spearman rho: external_lung `1.0`, immune `0.5`, lung `1.0`.
  - Positive-rho domains: `3/3`.
  - Top-feature transfer: `centered_cosine` matched in `3/3` domains.
  - Exact global significance:
    - mean-rho upper-tail p: `0.0369`
    - top-match-count upper-tail p: `0.0415`
- Interpretation:
  - Cross-model geometric ranking consistency is positive under an exact null, though the shared feature set is small (`3`) and should be expanded before strong promotion.

### H18 — Intrinsic-Dimensionality Coupling with Geodesic Gain
- Family: `intrinsic_dimensionality`
- Split regime: `dual_axis_disjoint`
- Method:
  - Computed per seed-layer-split intrinsic metrics from immune scGPT embeddings on the split-specific gene set:
    - participation-ratio dimension,
    - top-5 local-linearity ratio.
  - Merged with prior manifold-gain outcome (`iter_0010` H13 delta AUROC geodesic minus Euclidean).
  - Tested seed-wise Spearman coupling and mean-rho significance by split using within-seed layer permutation null (`3000` draws per split-metric).
- Primary artifacts:
  - `iterations/iter_0011/h18_intrinsic_geodesic_metrics_by_seed_layer_split.csv`
  - `iterations/iter_0011/h18_intrinsic_geodesic_coupling_by_seed.csv`
  - `iterations/iter_0011/h18_intrinsic_geodesic_coupling_summary.csv`
- Quantitative results:
  - Rows merged: `72`.
  - Source split coupling (mean seed rho):
    - local_linearity_top5: `+0.2354`, two-sided p `0.1646` (not significant)
    - participation_ratio_dim: `-0.2354`, two-sided p `0.1759` (not significant)
  - Target split coupling (mean seed rho):
    - local_linearity_top5: `+0.4079`, two-sided p `0.0143`
    - participation_ratio_dim: `-0.4079`, two-sided p `0.0190`
  - Directionality across seeds:
    - target split: `3/3` positive for local_linearity and `3/3` negative for participation-ratio.
- Interpretation:
  - Coupling is split-conditional: target-disjoint shows significant structure, source-disjoint does not.

## Decision Summary
- `H16` (`module_structure`): **promising**.
- `H17` (`cross_model_alignment`): **promising** (tentative due only 3 shared features).
- `H18` (`intrinsic_dimensionality`): **neutral/mixed**.

## Blockers / Deviations
- No hard data/runtime blocker.
- Reproducibility deviation corrected during iteration: deterministic RNG seeding fix in H18, followed by full rerun.

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0011/h16_module_structure_by_seed_layer_split.csv`
- `iterations/iter_0011/h16_module_structure_layer_summary.csv`
- `iterations/iter_0011/h16_module_structure_split_summary.csv`
- `iterations/iter_0011/h17_cross_model_transfer_domain_summary.csv`
- `iterations/iter_0011/h17_cross_model_transfer_global_null.csv`
- `iterations/iter_0011/h17_cross_model_transfer_summary.json`
- `iterations/iter_0011/h18_intrinsic_geodesic_metrics_by_seed_layer_split.csv`
- `iterations/iter_0011/h18_intrinsic_geodesic_coupling_by_seed.csv`
- `iterations/iter_0011/h18_intrinsic_geodesic_coupling_summary.csv`
- `iterations/iter_0011/iter0011_screen_summary.json`
