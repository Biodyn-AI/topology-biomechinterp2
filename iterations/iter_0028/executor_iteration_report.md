# Executor Iteration Report — iter_0028

## Scope
Executed the brainstormer-prioritized 3-slot screening packet with materially changed methods:
- `H67` (`persistent_homology`, `N329`): rank-based multiparameter persistence surface.
- `H68` (`cross_model_alignment`, `N338`): cycle-consistent utility-regularized cross-model mapping.
- `H69` (`manifold_distance`, `N335`): multiscale geodesic triangle-defect spectrum.

This iteration intentionally used two rescue-once directions (`H67`, `H68`) with major method changes, plus one cheap broad-screen novel geometry test (`H69`).

## Environment
- Python environment: `subproject40-topology`
- Package installation: none required

## Command Trace
```bash
# syntax check
conda run -n subproject40-topology \
  python -m py_compile iterations/iter_0028/run_iter0028_screen.py

# execute 3-slot screening packet
conda run -n subproject40-topology \
  python iterations/iter_0028/run_iter0028_screen.py
```

Primary runner:
- `iterations/iter_0028/run_iter0028_screen.py`

Primary machine-readable outputs:
- `iterations/iter_0028/iter0028_screen_summary.json`
- `iterations/iter_0028/h67_rank_surface_by_seed_layer_split.csv`
- `iterations/iter_0028/h67_rank_surface_domain_summary.csv`
- `iterations/iter_0028/h67_rank_surface_null_summary.csv`
- `iterations/iter_0028/h68_cycle_utility_ot_by_domain_layer_split.csv`
- `iterations/iter_0028/h68_cycle_utility_ot_domain_summary.csv`
- `iterations/iter_0028/h68_cycle_utility_ot_null_summary.csv`
- `iterations/iter_0028/h69_triangle_defect_by_seed_layer_split.csv`
- `iterations/iter_0028/h69_triangle_defect_domain_summary.csv`
- `iterations/iter_0028/h69_triangle_defect_null_summary.csv`

Execution runtime:
- End-to-end screen command wall time: `~103s`.

## Results

### H67 — Rank-Based Multiparameter Persistence Surface (`persistent_homology`, `N329`)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Compared rank-surface topology score vs directed baseline and one-axis ablation.
- Controls: axis-rank permutation within degree bins + label permutation (`12` each).

Direct evidence:
- Mean `delta_AUROC(rank_surface - baseline) = -0.03048` (std `0.01348`).
- Positive rows: `1/36`; positive mean domain-splits: `0/6`.
- Failure slices remained negative:
  - `lung/source_disjoint = -0.03243`
  - `external_lung/source_disjoint = -0.03453`
- One-axis comparison: mean `delta_AUROC(rank_surface - one_axis) = +0.00402` (small), indicating no practical rescue vs baseline.
- Domain-split Fisher-significant aggregates: `1/6` (immune/target), but in negative direction.

Interpretation:
- Major-change rank-bifiltration still fails utility and failure-slice rescue gates.

### H68 — Cycle-Consistent Utility-Regularized Cross-Model Mapping (`cross_model_alignment`, `N338`)
Design:
- Coverage: seed42 pilot, held-out-domain transfer across `3 domains x 2 disjoint splits x 2 layers = 12` rows.
- Method: utility-weighted cycle-consistent GF↔scGPT mapping, then mapped-GF transfer scoring against GF baseline.
- Controls: random linear map, anchor-pair shuffle, signature-destroy permutation (`12` each).

Direct evidence:
- Mean `delta_AUROC(transfer - baseline) = -0.30464` (std `0.08177`).
- Mean `null_gap_q95 = -0.03847`.
- Alignment quality remained weak in held-out targets:
  - mean mapped-to-sc cosine `= +0.00560`
  - immune mapped cosine was negative (`-0.06643` mean across splits/layers).
- Immune gate failed: mean immune `null_gap_q95 = -0.10658` and immune mean transfer delta `= -0.34478`.
- Non-immune condition failed: rows with `(null_gap_q95 > 0 and delta >= 0)` were `0/8`.

Interpretation:
- This major redesign did not rescue cross-model utility; both transfer utility and null-gap criteria fail.

### H69 — Multiscale Geodesic Triangle-Defect Spectrum (`manifold_distance`, `N335`)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Score: geodesic-support baseline plus multiscale triangle-defect features (`k={8,12,16}`).
- Controls: endpoint swap within geodesic bins, matched random-third-node controls, label permutation (`8` each).

Direct evidence:
- Mean `delta_AUROC(triangle_defect - baseline) = +0.02617` (std `0.02707`).
- Positive rows: `30/36`; positive mean domain-splits: `6/6`.
- Domain-split mean deltas:
  - `external_lung/source_disjoint = +0.02514`
  - `external_lung/target_disjoint = +0.05049`
  - `immune/source_disjoint = +0.00162`
  - `immune/target_disjoint = +0.00869`
  - `lung/source_disjoint = +0.01797`
  - `lung/target_disjoint = +0.05308`
- Null contrast by family (global):
  - endpoint-swap null mean `-0.01721` (q95 `+0.01533`)
  - label-permutation null mean `-0.01907` (q95 `+0.01244`)
  - matched-random-third null mean `+0.01350` (q95 `+0.03147`)

Interpretation:
- H69 is the only branch with broad positive utility direction this iteration.
- Evidence is not yet row-level significant under current permutation budget (`p_best` floor `0.111`), and matched-random-third null remains a meaningful challenge; this is promising but not yet definitive.

## Decision Summary
- `H67`: **negative** (major-change rescue failed across all domain-splits and known failure slices).
- `H68`: **negative** (major-change rescue failed both utility and null-gap gates).
- `H69`: **promising** (consistent positive deltas across all domain-splits; requires stronger null-permutation resolution and robustness expansion before promotion to strong claim).

## Branch Retirement Actions
- Retire current rank-surface persistence utility endpoint (`H67`) for this loop; it is a second negative/inconclusive utility outcome after prior persistence utility failures.
- Retire current cross-model utility-transfer endpoint (`H68`) for this loop after the major-change rescue failed immune and non-immune gates.
- Keep `H69` active for one follow-up robustness iteration with higher null resolution and split-seed expansion.

## Blockers and Fallbacks
- No hard data/runtime blocker.
- Methodological caveat: `H69` used `8` permutations per null family (coarse p-value floor), so row-level significance is under-resolved; fallback is straightforward (`>=32` permutations) and computationally feasible.

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with marker `ITERATION UPDATE: iter_0028`.
- Compile command (run from `paper/`):
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.
