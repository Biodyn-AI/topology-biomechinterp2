# Executor Iteration Report — iter_0029

## Scope
Executed a breadth-first 3-hypothesis packet aligned to the prior brainstormer brief:
- `H70` (`manifold_distance`, refinement of `H69`): hard-null robustness expansion for multiscale geodesic triangle-defect features.
- `H71` (`cross_model_alignment`, major-change rescue): topology-signature distillation transfer from Geneformer to scGPT signature space.
- `H72` (`topology_stability`, new method): edge trajectory motif class scan across layers.

Portfolio balance this iteration:
- Carry-over refinement: `H70` (single refinement slot).
- Materially novel methods: `H71`, `H72`.

## Environment
- Python environment: `subproject40-topology`
- Package installation: none required

## Command Trace
```bash
# syntax check
conda run -n subproject40-topology \
  python -m py_compile iterations/iter_0029/run_iter0029_screen.py

# execute 3-slot screening packet
conda run -n subproject40-topology \
  python iterations/iter_0029/run_iter0029_screen.py
```

Primary runner:
- `iterations/iter_0029/run_iter0029_screen.py`

Primary machine-readable outputs:
- `iterations/iter_0029/iter0029_screen_summary.json`
- `iterations/iter_0029/h70_triangle_defect_robust_by_seed_layer_split.csv`
- `iterations/iter_0029/h70_triangle_defect_robust_domain_summary.csv`
- `iterations/iter_0029/h70_triangle_defect_robust_null_summary.csv`
- `iterations/iter_0029/h71_topology_signature_distill_by_domain_layer_split.csv`
- `iterations/iter_0029/h71_topology_signature_distill_domain_summary.csv`
- `iterations/iter_0029/h71_topology_signature_distill_null_summary.csv`
- `iterations/iter_0029/h72_edge_trajectory_motif_by_domain_split.csv`
- `iterations/iter_0029/h72_edge_trajectory_motif_domain_summary.csv`
- `iterations/iter_0029/h72_edge_trajectory_motif_null_summary.csv`

Execution runtime:
- End-to-end screen command wall time: `~10 minutes`.

## Results

### H70 — Triangle-Defect Hard-Null Robustness Expansion (`manifold_distance`, `N343`)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Same feature recipe as `H69` (`k={8,12,16}`) to preserve comparability.
- Null budget increased to `48` permutations per family:
  - endpoint swap within geodesic bins,
  - matched random third node,
  - label permutation.
- Added bootstrap CIs for domain-split mean deltas.

Direct evidence:
- Mean `delta_AUROC(triangle_defect - baseline) = +0.02637` (std `0.02687`).
- Positive rows: `29/36`.
- Rows with `p_best < 0.05`: `36/36` (new p-value floor from 48-permutation budget).
- Mean matched-random-third null gap: `+0.01010`; row-level positive matched-random gap in `24/36` rows.
- Domain-split mean deltas were positive in `6/6`:
  - minimum: immune/source-disjoint `+0.00543` (CI crosses zero),
  - maximum: lung/target-disjoint `+0.05334`.
- Domain-split mean matched-random-third null gaps were positive in `6/6`.

Gate check versus brief (`N343`):
- Global mean delta `>= +0.015`: **pass** (`+0.02637`).
- Positive mean in `>=5/6` domain-splits: **pass** (`6/6`).
- Matched-random-third null-gap positive in `>=3/6` domain-splits: **pass** (`6/6` by mean gap).

Interpretation:
- The `H69` signal survives stronger null calibration and remains directionally consistent across all domain-splits.
- Weakest effects remain in immune splits; these should be stressed with stronger biological and coexpression controls before claiming mechanism-level specificity.

### H71 — Cross-Model Topology-Signature Distillation (`cross_model_alignment`, `N350`)
Design:
- Seed42 held-out-domain pilot across immune/lung/external-lung, splits `{source,target}` and layers `{7,11}` (`12` rows).
- Major method change vs prior cross-model branches:
  - weighted ridge distillation `GF signatures -> scGPT topology signatures`,
  - source-domain token-affinity teacher in scGPT signature space,
  - held-out-domain transfer utility test versus Geneformer baseline.
- Null controls (`24` permutations each):
  - random teacher-signature assignment,
  - anchor-label shuffle,
  - signature-destroy permutation.

Direct evidence:
- Mean `delta_AUROC(transfer - baseline) = -0.42758` (std `0.17687`).
- Positive rows: `0/12`.
- Mean `null_gap_q95 = -0.14795`.
- Mean mapped-to-sc cosine alignment: `+0.00634` (near zero).
- Domain-split mean deltas ranged from `-0.57268` (immune/target) to `-0.28866` (lung/source).

Interpretation:
- This major-change rescue decisively failed the transfer utility endpoint and did not establish meaningful cross-model alignment.
- Cross-model alignment in this loop remains non-promotable and should be retired in this endpoint family.

### H72 — Edge Trajectory Motif Class Screen (`topology_stability`, `N355`)
Design:
- Cheap pilot on `seed42_main` across 3 domains and both disjoint splits (`6` rows).
- Built edge trajectories over layers `{0,3,7,11}` from baseline + triangle-defect channels.
- Clustered trajectories into motif classes (`K=3` effective), then tested motif-augmented edge scoring vs baseline.
- Controls (`12` permutations each):
  - layer-order permutation,
  - motif-label shuffle within degree bins.

Direct evidence:
- Mean `delta_AUROC(motif - baseline) = +0.00008` (std `0.00593`).
- Positive rows: `4/6`, but with very small magnitudes.
- Rows with `p_best < 0.05`: `0/6`.
- Rows with motif enrichment `p < 0.05`: `0/6`.
- Mean best motif enrichment (train split): `0.445`, but null-calibrated significance not achieved.

Interpretation:
- Motif classes are descriptively separable but do not yet produce robust incremental predictive utility.
- This pilot is inconclusive; evidence is insufficient for promotion.

## Decision Summary
- `H70`: **promising** and promoted as active branch; hard-null robustness gate passed.
- `H71`: **negative**; major-change rescue failed utility and alignment criteria.
- `H72`: **inconclusive**; cheap pilot completed, no robust null-surviving gain.

## Branch Retirement Actions
- Retire current cross-model topology-signature distillation utility endpoint (`H71`) for this loop.
- Keep `H70` active for targeted robustness/biological-anchor follow-up.
- Keep `H72` neutral (not retired yet), but do not allocate large budget without stronger seed/null support.

## Blockers and Fallbacks
- No hard data/runtime blocker.
- Statistical caveat for `H72`: only `12` permutations per null family (p-value floor `0.0769`), so confirmatory follow-up should increase to `>=48` and add multiseed coverage.

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with marker `ITERATION UPDATE: iter_0029`.
- Compile command (run from `paper/`):
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.
