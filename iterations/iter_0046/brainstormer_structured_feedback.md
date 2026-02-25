# Brainstormer Structured Feedback - iter_0046

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0046/executor_research_validation.json`).
- Full hypothesis packet planning is valid; no recovery-only restriction is required.

## Iteration Evidence Snapshot
- `H123` (`module_structure`) is the clear lead branch.
  - mean `delta_vs_h70 = +0.09351` across 22 rows
  - positive direction in `22/22` rows
  - positive mean null-gap in `8/8` observed domain-splits
  - remaining weakness is coverage, not robustness: `lung/dual_axis_disjoint` was missing after filtering/sampling.
- `H121` (`manifold_distance`) is a real but fragile positive.
  - mean `delta_vs_h70 = +0.03273`, positive in `11/12` rows
  - positive mean null-gap in `2/6` domain-splits
  - source-disjoint slices remain the stress point (especially external_lung and lung).
- `H122` (`cross_model_alignment`) is decisively negative.
  - mean `transport_score_neg_mse = -100.54990`
  - positive mean null-gap in `0/6` domain-splits
  - this endpoint should remain retired.

## Stale Direction Triage
1. Unanchored cross-model transport/gating resets (`H96/H99/H102/H109/H119/H122`) -> `retire_now`.
Reason: repeated objective rewrites still collapse under null-gap criteria, including the latest major reset.

2. Additive scalar persistent-homology rescue chain (`H100/H101/H103/H106/H110/H113`) -> `retire_now`.
Reason: long run of near-zero or negative robustness outcomes across multiple filtrations and controls.

3. Standalone additive intrinsic-dimension utilities (`H98/H114` line) -> `retire_now`.
Reason: repeated non-robust behavior; no evidence that additive ID terms are the right object.

4. Coarse discrete grammar lineage (`H104/H107/H111/H112`) -> `rescue_once_with_major_change`.
Required change: replace coarse tokenization with biologically anchored states and explicit dwell/entropy controls.

5. Pooled geodesic-curvature drift endpoint (`H120`) -> `retire_now`.
Reason: directional asymmetry variant (`H121`) dominates it and provides a cleaner geometric target.

6. Directional geodesic asymmetry (`H121`) -> `rescue_once_with_major_change`.
Required change: focus on source-disjoint hard slices, layer-conditioned features, and higher null budgets.

7. Signed motif-community hardening (`H118/H123`) -> `keep_active_high_priority`.
Reason: strongest and most stable branch, with remaining work concentrated in coverage completion.

## Strategic Pivot
- Keep one exploitation slot on `H123`-line hardening and biological conditioning.
- Use one exploratory slot for a genuinely different cross-model alignment objective that is anchor-constrained from the start.
- Use one cheap geometry screen built directly on the `H121` infrastructure to map where source-disjoint failure comes from.

## Minimal Recovery Plan (if a future gate flips to `false`)
1. Run a fast `H121`-line screen first (single seed, layers `{7,11}`, null budget `>=12`) to re-establish valid machine artifacts.
2. Run a narrowed `H123` confirmation on `lung` layer 11 with forced `dual_axis_disjoint` inclusion and full nulls.
3. Defer expensive cross-model experiments until gate returns to `true`.
