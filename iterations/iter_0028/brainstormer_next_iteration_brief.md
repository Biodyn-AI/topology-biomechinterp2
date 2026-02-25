# Brainstormer Next Iteration Brief - iter_0028

## Status and Objective
- Gate is green (`passed_min_research_gate=true`), so run a full 3-slot packet.
- Objective: convert the `H69` directional signal into null-robust evidence, while taking one final high-upside cross-model shot and one cheap mechanism screen.

## Suggested Slot Mapping (next executor loop)
- `H70` <- `N343` (high-probability discovery).
- `H71` <- `N350` (high-risk/high-reward).
- `H72` <- `N355` (cheap broad-screen).

## Slot A (Primary): N343 - H69 Robustness Expansion with Hard-Null Resolution
- Hypothesis: triangle-defect lift is real and survives stronger null calibration.
- Scope:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: `{7,11}`.
- Design:
  - Re-run `H69` with identical feature recipe (`k={8,12,16}`) to preserve comparability.
  - Increase null draws to `>=48` for each null family:
    - endpoint swap within geodesic bins,
    - matched random third node,
    - label permutation.
  - Add bootstrap CIs per domain-split and aggregate matched-random-third `null_gap_q95`.
- Keep gate:
  - Global mean `delta_AUROC >= +0.015`.
  - Positive mean in `>=5/6` domain-splits.
  - Matched-random-third `null_gap_q95 > 0` in `>=3/6` domain-splits.

## Slot B (High-Risk): N350 - Cross-Model Topology-Signature Distillation
- Hypothesis: transfer improves when aligning to topology signatures rather than global embedding maps.
- Scope:
  - Pilot `seed42`, held-out-domain design, layers `{7,11}`.
  - Expand to more seeds only if pilot gate passes.
- Design:
  - Build scGPT topology-signature targets from directed + triangle-defect features.
  - Train GF-to-signature mapping on two domains; evaluate held-out domain transfer utility vs GF baseline.
  - Report both utility (`delta_AUROC`) and robustness (`null_gap_q95`).
- Controls:
  - Random teacher-signature assignment.
  - Anchor-label shuffle.
  - Signature-destroy permutation.
- Keep gate:
  - Immune: `delta_AUROC >= 0` and `null_gap_q95 > 0`.
  - Non-immune: at least one domain-split with `delta_AUROC >= 0` and `null_gap_q95 > 0`.

## Slot C (Cheap Breadth): N355 - Edge Trajectory Motif Screen
- Hypothesis: layerwise geometry/topology trajectories partition edges into mechanistically meaningful motif classes.
- Scope:
  - All domains/seeds/splits using layers `{0,3,7,11}` if available; otherwise reuse `{7,11}` with transition proxies.
- Design:
  - Build per-edge trajectories from directed score + defect statistics across layers.
  - Cluster trajectories into motif classes (small fixed `K`, e.g., `4-6`).
  - Test enrichment of positives by motif class and incremental utility from motif class feature.
- Controls:
  - Layer-order permutation.
  - Class-label shuffle within degree bins.
- Keep gate:
  - At least one motif class enrichment survives permutation (`p<0.05`).
  - Incremental model with motif class has non-negative global mean utility delta.

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
1. Stop Slot A early if matched-random-third `null_gap_q95` is negative in all `6/6` domain-splits after half of planned permutations.
2. Stop Slot B expansion beyond pilot if immune fails either gate condition.
3. Always complete Slot C because it is low-cost and broadens mechanistic coverage.

## Contingency (only if a future gate turns red)
1. Run Slot C only on `seed42`, layers `{7,11}` with one permutation family.
2. Produce valid machine summary JSON and valid hypothesis screen JSON.
3. Defer Slot A/B until research gate is restored.
