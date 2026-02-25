# Brainstormer Structured Feedback - iter_0045

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0045/executor_research_validation.json`).
- Recovery-only mode is not required for this iteration.

## Iteration Evidence Snapshot
- `H118` (`module_structure`) is the only strong branch this round.
  - mean `delta_vs_h70 = +0.09885` across 18 rows
  - direction positive in `18/18`
  - positive mean null-gap in only `3/6` domain-splits
  - split-level weak points are narrow and systematic (negative mean null-gap in `immune/source_disjoint`, `lung/source_disjoint`, `external_lung/target_disjoint`)
- `H119` (`cross_model_alignment`) is a clean negative endpoint.
  - mean `delta_vs_h70 = +0.00060`
  - positive mean null-gap in `1/6` domain-splits
- `H120` (`manifold_distance`) is directional but asymmetric and only partially robust.
  - mean `delta_vs_h70 = +0.03854`, positive in `12/12` rows
  - positive mean null-gap in `3/6` domain-splits
  - all three `source_disjoint` domain means have negative null-gap, while `target_disjoint` means are positive

## Stale Direction Triage
1. Disagreement-gated transfer utility endpoint (`H119` and similar transfer-AUROC gating formulations) -> `retire_now`.
Reason: repeated cross-model rescue attempts keep collapsing at null calibration and this variant is near-zero effect.

2. Additive PH rescue lineage (`H100/H101/H103/H106/H110/H113` style additive barcode/filtration tweaks) -> `retire_now`.
Reason: repeated `0/6` or near-`0/6` null-gap domain support under materially different constructions.

3. Coarse finite-state motif grammar lineage (`H104/H107/H111/H112`) -> `rescue_once_with_major_change`.
Required major change: move from coarse quantile states to biologically anchored, occupancy-matched mechanistic states with dwell-time statistics.

4. Standalone/additive intrinsic-dimension lineage (`H98/H114` style) -> `retire_now`.
Reason: long negative tail with no robust promotion signal.

5. Tangent-acceleration endpoint (`H115`) -> `retire_now`.
Reason: negative direction and no robustness support in the tested formulation.

6. Geodesic curvature-drift branch (`H120`) -> `rescue_once_with_major_change`.
Required major change: model directional asymmetry explicitly (`source->target` vs `target->source`) instead of pooled curvature descriptors.

7. Signed motif-community branch (`H118`) -> `keep_active_high_priority`.
Reason: strongest multiseed effect in the current window; needs hardening, not replacement.

## Portfolio Steering for Next Loop
- Spend one slot hardening `H118` with stronger nulls and a harder split regime.
- Spend one slot on a genuinely new topological object (not another additive PH scalar tweak).
- Spend one cheap slot on source-target asymmetry geometry, since `H120` exposes a clear directional failure pattern.

## Minimal Recovery Plan (only if a future gate flips to false)
1. Run the cheap asymmetry screen first with reduced null budget (`>=12`) to restore valid machine artifacts quickly.
2. Run a narrowed `H118` hardening pass on one domain (`lung`) and one layer (`11`) with full nulls.
3. Defer expensive cross-model topology alignment until gate returns `true`.
