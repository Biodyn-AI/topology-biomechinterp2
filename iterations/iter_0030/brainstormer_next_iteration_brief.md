# Brainstormer Next Iteration Brief - iter_0030

## Status and Objective
- Gate status is green (`passed_min_research_gate=true`), so run a full 3-slot packet.
- Objective: keep the robust `H70` lineage, test one final cross-model idea with a non-edge endpoint, and keep one cheap geometry breadth probe.

## Suggested Slot Mapping (next executor loop)
- `H76` <- `N382` (high-probability discovery).
- `H77` <- `N379` (high-risk/high-reward).
- `H78` <- `N376` (cheap broad-screen).

## Slot A (Primary): N382 - Coexpression-aware Support-Concordance Interaction v2
- Hypothesis: topology lift (`H70`) is biologically meaningful only in coexpression/ontology-concordant strata.
- Scope:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: `{7,11}`.
- Design:
  - Keep `H70` geometry score fixed for comparability.
  - Build support-concordance from TRRUST/STRING/GO, then stratify by geodesic x degree x coexpression x ontology-tier.
  - Evaluate interaction: `delta_AUROC(high_support) - delta_AUROC(low_support)`.
- Controls:
  - Support shuffle within full strata.
  - Matched random-support sampling within strata.
  - Label shuffle within geodesic bins.
- Keep gate:
  - Global interaction mean `> 0`.
  - Immune/source `delta_AUROC > 0`.
  - Interaction `null_gap_q95 > 0` in `>=3/6` domain-splits.

## Slot B (High-Risk): N379 - Cross-Model Relational Rank Agreement (Non-edge Endpoint)
- Hypothesis: cross-model shared structure exists at relational-rank level even when edge-transfer AUROC fails.
- Scope:
  - Pilot on `seed42` first.
  - Domains: held-out-domain setup over `immune`, `lung`, `external_lung`.
  - Layers: `{7,11}`.
- Design:
  - Build per-model topology score matrices/signatures.
  - Learn source-domain relational alignment (orthogonal/spectral map).
  - Evaluate held-out-domain rank concordance (`Kendall tau`, `Spearman`, top-k overlap), not edge utility transfer.
- Controls:
  - Symbol permutation.
  - Spectral basis randomization.
  - Signature-destroy permutation.
- Keep gate:
  - Immune concordance metric above null q95.
  - At least one non-immune domain/split also above null q95.

## Slot C (Cheap Breadth): N376 - Geodesic Detour Elasticity Screen
- Hypothesis: true regulatory edges are more geodesically robust to local neighborhood perturbation.
- Scope:
  - `seed42`.
  - Domains all, splits both.
  - Layers `{0,3,7,11}`.
- Design:
  - Apply fixed local neighbor-dropout schedules around endpoint neighborhoods.
  - Recompute edge detour ratio under perturbations.
  - Add elasticity descriptors to geodesic baseline and test incremental lift.
- Controls:
  - Matched endpoint randomization with same dropout schedule.
  - Label permutation.
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
1. Stop Slot B after pilot if immune fails the null-q95 gate.
2. Stop Slot A expansion if interim interaction mean remains <= 0 after half permutation budget.
3. Always finish Slot C to preserve breadth coverage.

## Minimal Recovery Plan (if a future gate turns red)
1. Run Slot C only (`N376`) on `seed42` with 24 permutations and one compact summary CSV.
2. Run reduced Slot A (`N382`) on immune/lung at layers `{7,11}` with stratified support shuffle only.
3. Emit valid `executor_hypothesis_screen.json` and `iterXXXX_screen_summary.json` before any expansion.
