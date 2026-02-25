# Brainstormer Next Iteration Brief - iter_0029

## Status and Objective
- Gate is green (`passed_min_research_gate=true`), so run a full 3-slot packet.
- Objective: convert `H70` into biologically anchored evidence, take one final high-upside cross-model attempt, and run one cheap dynamic-geometry screen.

## Suggested Slot Mapping (next executor loop)
- `H73` <- `N368` (high-probability discovery).
- `H74` <- `N365` (high-risk/high-reward).
- `H75` <- `N361` (cheap broad-screen).

## Slot A (Primary): N368 - Support-Concordance Anchoring of H70
- Hypothesis: triangle-defect lift is strongest on biologically convergent edges and this explains weak immune slices.
- Scope:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: `{7,11}`.
- Design:
  - Keep `H70` features fixed for comparability.
  - Build support-concordance score from TRRUST + STRING + GO agreement.
  - Fit defect x support interaction and stratified utility models with degree+coexpression matching.
- Controls:
  - Support-score shuffle within degree+coexpression bins.
  - Matched random-support controls.
  - Label permutation.
- Keep gate:
  - Global interaction coefficient mean `> 0`.
  - Immune/source mean `delta_AUROC > 0`.
  - Null-surviving interaction support in `>=3/6` domain-splits.

## Slot B (High-Risk): N365 - Relational Spectral Cross-Model Alignment
- Hypothesis: cross-model transfer works when aligning relational spectra of topology signatures instead of direct token mappings.
- Scope:
  - Pilot on `seed42`, held-out-domain setup, layers `{7,11}`.
  - Expand to multiseed only if pilot gate passes.
- Design:
  - Compute topology-signature matrices in scGPT and Geneformer.
  - Align Gram/Laplacian spectra with orthogonal map learned on source domains.
  - Evaluate held-out transfer utility vs baseline and report `null_gap_q95`.
- Controls:
  - Eigen-spectrum permutation.
  - Random orthogonal basis controls.
  - Signature-destroy permutation.
- Keep gate:
  - Immune: `delta_AUROC >= 0` and `null_gap_q95 > 0`.
  - Non-immune: at least one domain-split with same condition.

## Slot C (Cheap Breadth): N361 - Curvature-Acceleration Screen
- Hypothesis: real regulatory edges show smoother (lower-acceleration) geodesic curvature trajectories across depth.
- Scope:
  - Start with `seed42`, domains all, splits both, layers `{0,3,7,11}`.
- Design:
  - Compute edge-level curvature per layer.
  - Derive slope and acceleration terms and test incremental utility over geodesic baseline.
- Controls:
  - Layer-order permutation.
  - Curvature shuffle within degree bins.
- Keep gate:
  - Non-negative global mean `delta_AUROC`.
  - Positive mean in `>=4/6` domain-splits.
  - At least one domain-split with permutation significance.

## Required Outputs
- Per slot:
  - by-row CSV,
  - domain-summary CSV,
  - null-summary CSV.
- Packet summary:
  - `iterations/<next_iter>/iterXXXX_screen_summary.json`.
- Mandatory executor artifacts:
  - `iterations/<next_iter>/executor_iteration_report.md`
  - `iterations/<next_iter>/executor_hypothesis_screen.json`
  - `iterations/<next_iter>/executor_next_steps.md`

## Fast-Fail Rules
1. Stop Slot B expansion beyond pilot if immune fails either gate condition.
2. Stop Slot A expansion if all immune rows remain interaction-negative after half permutations.
3. Always finish Slot C because it is cheap and keeps breadth coverage.

## Contingency (only if a future gate turns red)
1. Run Slot C only on `seed42` with 24 permutations and produce one compact summary.
2. Run reduced Slot A on immune only (`layers {7,11}`) with matched support shuffle.
3. Emit valid machine-readable outputs (`executor_hypothesis_screen.json` and `iterXXXX_screen_summary.json`) before any further expansion.
