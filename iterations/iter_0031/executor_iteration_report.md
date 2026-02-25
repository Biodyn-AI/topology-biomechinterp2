# Executor Iteration Report — iter_0031

## Scope
Executed a breadth-oriented 3-hypothesis packet aligned to the `N382/N379/N376` brief:
- `H76` (`module_structure`, refinement / rescue-once major change): coexpression-aware support-concordance interaction v2 for the `H70` triangle-defect branch.
- `H77` (`cross_model_alignment`, rescue-once major change): non-edge relational rank agreement endpoint (Spearman/top-k overlap), replacing edge-utility transfer.
- `H78` (`manifold_distance`, new method): geodesic detour-elasticity screen under neighborhood-size perturbation (`k={8,12,16}`).

Portfolio balance this iteration:
- Carry-over refinement: `H76` only.
- Materially changed/new methods: `H77`, `H78`.

## Environment
- Python environment: `subproject40-topology`
- Package installation: none required

## Command Trace
```bash
# syntax check
conda run -n subproject40-topology \
  python -m py_compile iterations/iter_0031/run_iter0031_screen.py

# execute 3-slot screening packet
conda run -n subproject40-topology \
  python iterations/iter_0031/run_iter0031_screen.py

# metric extraction checks used in this report
conda run -n subproject40-topology python -c "import pandas as pd; d='iterations/iter_0031'; h76=pd.read_csv(f'{d}/h76_coexpression_support_interaction_by_seed_layer_split.csv'); h77=pd.read_csv(f'{d}/h77_relational_rank_agreement_by_domain_layer_split.csv'); h78=pd.read_csv(f'{d}/h78_geodesic_detour_elasticity_by_domain_split_layer.csv'); print(len(h76), len(h77), len(h78))"
```

Primary runner:
- `iterations/iter_0031/run_iter0031_screen.py`

Primary machine-readable outputs:
- `iterations/iter_0031/iter0031_screen_summary.json`
- `iterations/iter_0031/h76_coexpression_support_interaction_by_seed_layer_split.csv`
- `iterations/iter_0031/h76_coexpression_support_interaction_domain_summary.csv`
- `iterations/iter_0031/h76_coexpression_support_interaction_null_summary.csv`
- `iterations/iter_0031/h77_relational_rank_agreement_by_domain_layer_split.csv`
- `iterations/iter_0031/h77_relational_rank_agreement_domain_summary.csv`
- `iterations/iter_0031/h77_relational_rank_agreement_null_summary.csv`
- `iterations/iter_0031/h78_geodesic_detour_elasticity_by_domain_split_layer.csv`
- `iterations/iter_0031/h78_geodesic_detour_elasticity_domain_summary.csv`
- `iterations/iter_0031/h78_geodesic_detour_elasticity_null_summary.csv`

## Results

### H76 — Coexpression-Aware Support-Concordance Interaction v2 (`module_structure`, `N382`)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Kept H70-style baseline + triangle-defect augmentation.
- Added coexpression-aware stratification with strata `geodesic x degree x coexpression_proxy x ontology_overlap`.
- Controls (`24` each): support shuffle within strata, matched random support within strata, label shuffle within geodesic bins.

Direct evidence:
- Mean `delta_AUROC(triangle-baseline) = +0.02323` (`27/36` rows positive).
- Mean interaction (high-support minus low-support lift) `= +0.00041` (`18/36` rows positive).
- Interaction `null_gap_q95 > 0` in `11/36` rows, but only `1/6` domain-split means were null-surviving.
- Domain-split means were mixed:
  - strongest positive: `external_lung/source_disjoint = +0.07245` interaction, `mean null_gap=+0.00580`.
  - strongest negative: `external_lung/target_disjoint = -0.05754`, `mean null_gap=-0.10613`.
- Immune/source remained weak for the anchor objective (`mean delta_AUROC(triangle-baseline) = -0.00460`).

Gate check versus brief (`N382`):
- Global interaction mean `> 0`: **pass (marginal)**.
- Immune/source `delta_AUROC > 0`: **fail** (`-0.00460`).
- Interaction `null_gap_q95 > 0` in `>=3/6` domain-splits: **fail** (`1/6`).

Interpretation:
- The geometric lift remains present, but the coexpression-aware anchoring interaction is still not robust enough for promotion.

### H77 — Cross-Model Relational Rank Agreement Endpoint (`cross_model_alignment`, `N379`)
Design:
- Seed42 held-out-domain pilot, layers `{7,11}`, splits `{source,target}` (`12` rows).
- Learned source-domain orthogonal map from Geneformer spectral signatures to scGPT signatures.
- Endpoint changed from edge-transfer AUROC to non-edge relational agreement:
  - primary: `delta_spearman(mapped_vs_sc - baseline_vs_sc)` on pairwise distance ranks,
  - secondary: `delta_topk_overlap` for closest-pair overlap.
- Controls (`24` each): symbol permutation, random orthogonal basis, signature-basis destroy.

Direct evidence:
- Mean `delta_spearman = +5.65e-06` (effectively zero).
- Mean `delta_topk_overlap = +3.56e-05` (effectively zero).
- Mean `null_gap_q95 = -0.01897`; rows with positive null-gap: `0/12`.
- `delta_spearman > 0` in `5/12` rows, but no domain-split had positive mean null-gap.

Gate check versus brief (`N379`):
- Immune above null q95: **fail**.
- At least one non-immune domain/split above null q95: **fail**.

Interpretation:
- The non-edge relational endpoint does not recover cross-model consistency under null calibration; this rescue remains negative.

### H78 — Geodesic Detour-Elasticity Screen (`manifold_distance`, `N376`)
Design:
- Seed42 broad screen over all domains, both disjoint splits, and layers `{0,3,7,11}` (`24` rows).
- Computed edge detour ratios using geodesic distances at neighborhood scales `k={8,12,16}`.
- Added elasticity descriptors (span, CV, mean detour) on top of geodesic/support baseline.
- Controls (`24` each): endpoint swap within geodesic bins, elasticity-feature shuffle within bins, label shuffle.

Direct evidence:
- Mean `delta_AUROC(elasticity-baseline) = +0.00193` (`14/24` rows positive).
- Positive mean delta in `4/6` domain-splits.
- `p_best < 0.05` in `11/24` rows and Fisher-significant in `4/6` domain-splits.
- However, mean `null_gap_q95` remained negative in all domain-splits (`0/6` positive).

Gate check versus brief (`N376`):
- Non-negative global mean delta: **pass**.
- Positive mean in `>=4/6` domain-splits: **pass** (`4/6`).
- At least one permutation-significant split: **pass**.
- Strict null-gap robustness: **fail** (`0/6` positive mean null-gap).

Interpretation:
- Detour-elasticity is directionally promising as a cheap screen but remains below robust null thresholds; classify as inconclusive, not promotable yet.

## Decision Summary
- `H76`: **inconclusive** (interaction objective not robust; one null-surviving domain-split only).
- `H77`: **negative** (new endpoint still null-gap negative across all rows).
- `H78`: **inconclusive** (passes directional breadth gates but fails null-gap robustness).

## Blockers and Fallbacks
- No data/runtime blocker.
- Method-level blocker: cross-model branch still collapses under null-gap despite endpoint change.

Fallback action used this iteration:
- Preserved breadth with `H78` cheap manifold perturbation screen while keeping one explicit cross-model rescue and one biologically anchored refinement.

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with marker `ITERATION UPDATE: iter_0031`.
- Compile command (run from `paper/`):
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.
