# Brainstormer Next Iteration Brief - iter_0027

## Status and Objective
- Current gate status is green (`passed_min_research_gate=true`), so run a full 3-slot experiment packet in the next loop.
- Objective: recover practical utility while preserving null robustness by prioritizing new topology/geometry formulations and one final cross-model major redesign.

## Suggested Slot Mapping (next executor loop)
- `H67` <- `N329` (high-probability discovery).
- `H68` <- `N338` (high-risk/high-reward).
- `H69` <- `N335` (cheap broad-screen).

## Slot A (Primary): N329 - Rank-Based Multiparameter Persistence Surface
- Hypothesis: rank-bifiltration topology features recover source-disjoint failure slices that thresholded two-axis features missed.
- Scope:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: start `{7,11}`, expand to `{0,3,7,11}` only if pilot clears fast-fail checks.
- Design:
  - Build persistence landscapes/images on rank axes `(distance rank, directed-support-margin rank)`.
  - Compare `directed/signed baseline` vs `baseline + N329 features`.
  - Report utility (`delta_AUROC`, calibration slope, ECE).
- Controls:
  - Axis-rank permutation within degree bins.
  - One-axis ablation (distance-only rank filtration).
  - Label permutation.
- Keep gate:
  - `lung/source_disjoint` mean delta `>= 0`.
  - `external_lung/source_disjoint` mean delta `>= 0`.
  - Positive mean delta in `>=4/6` domain-splits.

## Slot B (High-Risk): N338 - Cycle-Consistent Utility-Regularized Cross-Model OT
- Hypothesis: adding cycle consistency and direct utility regularization can produce both robust null-gap and non-negative transfer.
- Scope:
  - Pilot on `seed42`, layers `{7,11}`, held-out-domain design.
  - Expand only if pilot gate passes.
- Design:
  - Learn bidirectional scGPT↔Geneformer barycentric transport on two domains.
  - Optimize joint objective: cycle loss + transfer utility + entropy regularization.
  - Evaluate held-out domain transfer vs baseline and null-gap.
- Controls:
  - Random transport matrix.
  - Anchor-label shuffle.
  - Signature-destroy permutation.
- Keep gate:
  - Immune mean `null_gap_q95 > 0` and transfer delta `>= 0`.
  - At least one non-immune split with `null_gap_q95 > 0` and transfer delta `>= 0`.

## Slot C (Cheap Breadth): N335 - Multiscale Geodesic Triangle-Defect Spectrum
- Hypothesis: multiscale triangle-defect features provide lightweight geometric lift missed by prior anisotropy and ID variants.
- Scope:
  - Same domains/seeds/splits as Slot A.
  - Layers `{7,11}` first; full-depth expansion optional.
- Design:
  - For each edge, sample third-node triangles at `k={8,12,16}`.
  - Compute defect quantiles/tails/skew features and add to directed baseline.
- Controls:
  - Endpoint swap within geodesic bins.
  - Matched random third-node controls.
  - Label permutation.
- Keep gate:
  - Global mean delta `>= 0`.
  - Positive mean delta in `>=4/6` domain-splits.

## Required Outputs
- Per slot:
  - by-row CSV,
  - domain-summary CSV,
  - null-summary CSV.
- Packet summary JSON:
  - `iterations/<next_iter>/iterXXXX_screen_summary.json`.
- Mandatory executor artifacts:
  - `iterations/<next_iter>/executor_iteration_report.md`
  - `iterations/<next_iter>/executor_hypothesis_screen.json`
  - `iterations/<next_iter>/executor_next_steps.md`

## Fast-Fail Rules
1. Stop Slot A expansion beyond `{7,11}` if both source-disjoint failure slices are below `-0.003` after pilot.
2. Stop Slot B expansion immediately if immune fails either condition (`null_gap_q95 <= 0` or transfer delta `< 0`).
3. Always complete Slot C because it is low-cost and broad-coverage.

## Minimal Recovery Plan (only if a future gate turns red)
1. Run Slot C only on `seed42`, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
2. Produce one valid machine summary JSON and one valid hypothesis screen JSON.
3. Defer Slot A/B until the research gate is restored.
