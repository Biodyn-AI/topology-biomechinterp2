# Brainstormer Next Iteration Brief — iter_0016 -> iter_0017

## Gate Status
- Current validation passed: `passed_min_research_gate=true`.
- Run a normal 3-hypothesis packet.

## Minimal Executable Plan for iter_0017

### 1) High-probability discovery run
- Candidate: `N141` (multi-seed convexity/detour incremental test).
- Objective: verify `H32` robustness and isolate unique signal beyond diffusion.
- Protocol:
1. Re-run convexity/detour features for seeds `42/43/44`, domains `immune/lung/external_lung`, layers `0/3/7/11`, both disjoint splits.
2. Fit nested models: `geodesic+diffusion` vs `geodesic+diffusion+convexity+detour`.
3. Report AUROC delta, log-loss delta, and calibration delta per row and per domain/split aggregate.
4. Calibrate with degree x coexpression x geodesic matched permutations.
- Required artifacts:
  - `h34_convexity_detour_multiseed_by_seed_layer_split.csv`
  - `h34_convexity_detour_multiseed_domain_summary.csv`
  - `h34_convexity_detour_multiseed_null_summary.csv`
- Promotion gate:
  - Positive incremental delta in `>=2/3` domains and both split regimes.

### 2) Cheap broad-screen run
- Candidate: `N147` (local-linearity breakpoint screen).
- Objective: quickly test whether split asymmetry is a depth-phase mechanism.
- Protocol:
1. Use existing local linearity/reconstruction features and fit piecewise depth models per domain/split.
2. Estimate breakpoint depth and slope pre/post breakpoint.
3. Test breakpoint differences between source- and target-disjoint splits.
4. Validate with layer-order permutation controls.
- Required artifacts:
  - `h35_linearity_breakpoint_by_seed_domain_split.csv`
  - `h35_linearity_breakpoint_summary.csv`
  - `h35_linearity_breakpoint_null_summary.csv`
- Promotion gate:
  - Reproducible split-specific breakpoint shift in at least two domains.

### 3) High-risk/high-reward run
- Candidate: `N149` (anchor-regularized utility-optimized spectral alignment).
- Objective: rescue cross-model transfer with a utility-targeted objective.
- Protocol:
1. Build CCA-whitened spectral embeddings per domain pair.
2. Add soft TRRUST anchor regularization and optimize mapping for held-out transfer utility.
3. Evaluate edge-transfer AUROC, top-1 retrieval, and cycle-return as secondary.
4. Compare against CCA-only and `H33` baselines with random-anchor nulls.
- Required artifacts:
  - `h36_anchor_spectral_alignment_domain_summary.csv`
  - `h36_anchor_spectral_alignment_map_quality.csv`
  - `h36_anchor_spectral_alignment_null_summary.csv`
- Promotion gate:
  - Transfer AUROC gain over both baselines in `>=2/3` domains.

## Execution Order
1. Run `N141` first (highest expected yield with strongest prior evidence).
2. Run `N147` second (low-cost mechanism discriminator).
3. Run `N149` third (highest upside, highest tuning risk).

## Scope Guardrails
1. Do not reopen GW-primary methods.
2. Do not spend on cycle-consistency variants that lack a transfer-utility objective term.
3. Require every run to output row-level tables and null summaries.

## Contingency if a Future Gate Fails
1. Execute only `N141` on `immune` and `lung` with seed42 and both split regimes.
2. Emit mandatory narrative files plus one machine summary CSV.
3. Resume full packet after validation returns to pass state.
