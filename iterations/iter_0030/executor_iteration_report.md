# Executor Iteration Report — iter_0030

## Scope
Executed a breadth-oriented 3-hypothesis packet with one carry-over refinement and two materially new methods:
- `H73` (`module_structure`, refinement of `H70`): support-concordance biological anchoring for triangle-defect lift.
- `H74` (`cross_model_alignment`, explicit rescue-once): relational spectral alignment pilot for held-out-domain transfer.
- `H75` (`manifold_distance`, new method): geodesic curvature-acceleration dynamic screen.

Portfolio balance this iteration:
- Carry-over refinement: `H73` only.
- Materially novel methods: `H74`, `H75`.

## Environment
- Python environment: `subproject40-topology`
- Package installation: none required

## Command Trace
```bash
# syntax check
conda run -n subproject40-topology \
  python -m py_compile iterations/iter_0030/run_iter0030_screen.py

# execute 3-slot screening packet
conda run -n subproject40-topology \
  python iterations/iter_0030/run_iter0030_screen.py
```

Primary runner:
- `iterations/iter_0030/run_iter0030_screen.py`

Primary machine-readable outputs:
- `iterations/iter_0030/iter0030_screen_summary.json`
- `iterations/iter_0030/h73_support_concordance_by_seed_layer_split.csv`
- `iterations/iter_0030/h73_support_concordance_domain_summary.csv`
- `iterations/iter_0030/h73_support_concordance_null_summary.csv`
- `iterations/iter_0030/h74_relational_spectral_alignment_by_domain_layer_split.csv`
- `iterations/iter_0030/h74_relational_spectral_alignment_domain_summary.csv`
- `iterations/iter_0030/h74_relational_spectral_alignment_null_summary.csv`
- `iterations/iter_0030/h75_curvature_acceleration_by_domain_split.csv`
- `iterations/iter_0030/h75_curvature_acceleration_domain_summary.csv`
- `iterations/iter_0030/h75_curvature_acceleration_null_summary.csv`

## Results

### H73 — Support-Concordance Anchoring of Triangle-Defect Lift (`module_structure`, `N368`)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Kept H70 triangle-defect score fixed for comparability.
- Tested interaction proxy `delta_high_support - delta_low_support` using support-concordance `support - 0.4 * directional_margin`.
- Controls (`32` each):
  - support shuffle within matched geodesic/support strata,
  - matched random support within strata,
  - label shuffle within geodesic bins.

Direct evidence:
- Mean `delta_AUROC(triangle - baseline) = +0.02498` (still H70-like positive).
- Mean interaction proxy `= -0.00032` (near zero / slightly negative).
- Interaction-positive rows: `19/36`; rows with `p_best < 0.05`: `7/36`.
- Mean interaction `null_gap_q95 = -0.08390`; null-surviving interaction mean in domain-splits: `0/6`.
- Immune/source slice remained weak: mean `delta_AUROC = +0.00026`.

Gate check versus brief (`N368`):
- Global interaction mean `> 0`: **fail** (`-0.00032`).
- Immune/source mean `delta_AUROC > 0`: **pass but minimal** (`+0.00026`).
- Null-surviving interaction in `>=3/6` domain-splits: **fail** (`0/6`).

Interpretation:
- The underlying geometry signal persists, but this support-concordance interaction formulation does not explain/resolve weak slices and does not survive null calibration.

### H74 — Relational Spectral Cross-Model Alignment (`cross_model_alignment`, `N365`)
Design:
- Seed42 held-out-domain pilot, layers `{7,11}`, splits `{source,target}` (`12` rows).
- New method: spectral embeddings of scGPT/Geneformer topology signatures + orthogonal Procrustes map learned on source domains.
- Endpoint: held-out transfer utility (`delta_AUROC transfer - baseline`) using source-trained logistic edge model.
- Controls (`24` each): eigenspectrum permutation, random orthogonal basis, signature-destroy permutation.

Direct evidence:
- Mean `delta_AUROC = +0.01136` (mixed sign across rows).
- Mean `null_gap_q95 = -0.09881` (non-robust globally).
- Immune failed decisively: mean `delta_AUROC = -0.16020`, mean `null_gap_q95 = -0.14682`.
- Positive row-level null gap in only `2/12` rows.
- Mean mapped-to-sc cosine alignment `= +0.00017` (no practical alignment).

Gate check versus brief (`N365`):
- Immune `delta_AUROC >= 0` and `null_gap_q95 > 0`: **fail**.
- Non-immune support does not rescue immune gate failure.

Interpretation:
- This explicit rescue-once cross-model redesign remains non-promotable for edge-utility transfer; only a narrow lung/target slice is null-gap positive.

### H75 — Curvature-Acceleration Broad Screen (`manifold_distance`, `N361`)
Design:
- Seed42 broad scan over all domains and both disjoint splits (`6` rows), layers `{0,3,7,11}`.
- Built edge curvature trajectories from multiscale triangle-defect proxies; added slope/acceleration smoothness terms over geodesic baseline.
- Controls (`32` each): layer-order permutation, curvature shuffle within geodesic bins, label shuffle.

Direct evidence:
- Mean `delta_AUROC(curvature - baseline) = +0.00210`.
- Positive rows: `3/6`; rows with `p_best < 0.05`: `2/6`.
- Positive mean domain-splits: `3/6`.
- `null_gap_q95` was negative in all domain-splits (`0/6` positive).

Gate check versus brief (`N361`):
- Non-negative global mean delta: **pass** (`+0.00210`).
- Positive mean in `>=4/6` domain-splits: **fail** (`3/6`).
- At least one permutation-significant split: **pass** (`2/6`).

Interpretation:
- The dynamic-curvature formulation shows weak directional signal but fails robustness/null-gap requirements, so it remains inconclusive.

## Decision Summary
- `H73`: **neutral** (interaction hypothesis not supported; keep core H70 branch, not this anchoring formula).
- `H74`: **negative** (explicit rescue-once failed immune + null-gap gate; retire this cross-model transfer endpoint).
- `H75`: **inconclusive** (weak mixed signal, non-robust under null-gap criterion).

## Blockers and Fallbacks
- No data/runtime blocker.
- Methodological limitation encountered: current edge dataset lacks explicit coexpression columns, so `H73` used degree+geodesic-matched controls instead of degree+coexpression matching.

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with marker `ITERATION UPDATE: iter_0030`.
- Compile command (run from `paper/`):
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.
