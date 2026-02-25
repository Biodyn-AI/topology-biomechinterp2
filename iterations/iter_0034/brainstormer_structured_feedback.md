# Brainstormer Structured Feedback - iter_0034

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0034/executor_research_validation.json`).
- Recovery-only mode is not required for this iteration.

## Iteration Readout
- `H87` is the strongest live signal in the packet.
  - Mean `delta_auc_descriptor_blend_minus_h70 = +0.08035`.
  - Positive rows `24/24`; positive mean delta in `6/6` domain-splits.
  - Positive mean null-gap in `4/6` domain-splits; weakest slices are `immune/source_disjoint` and `external_lung/source_disjoint`.
- `H85` is a near-miss and should not be promoted as-is.
  - Mean `delta_auc_local_dual_filtration_plus_h70_minus_h70 = +0.00360`.
  - Positive mean delta in `5/6` domain-splits, but positive mean null-gap only `3/6` (gate was `>=4/6`).
  - Failure pattern is concentrated in `external_lung/*` and `lung/target_disjoint`, with instability tied to local-gap calibration.
- `H86` is decisively negative.
  - Mean `barcode_ot_depth_alignment_score = 0.58192`, but positive null-gap domains `0/3`.
  - This is another cross-model null-gap failure after multiple endpoint changes.

## Stale Direction Triage
- `cross_model_alignment` global utility/order endpoints -> `retire_now`.
  - Evidence: recent chain `H74/H77/H80/H83/H86` is uniformly non-promotable with null-gap failure.
  - Reopen only once with a major structural reset and strict preflight controls.
- Standalone additive `graph_topology` scoring line -> `retire_now`.
  - Evidence: repeated negatives (`H61`, `H84`) with no robust rescue.
- Standalone intrinsic-dimension utility-lift line -> `retire_now`.
  - Evidence: repeated negatives (`H54`, `H60`, `H63`, `H66`) and low rescue potential.
- Global detour/dropout elasticity utility forms -> `retire_now`.
  - Evidence: `H78` inconclusive then `H81` negative with null-gap collapse.
- `H85` dual-filtration local witness line -> `rescue_once_with_major_change`.
  - Required change: adaptive, split/layer-conditioned bifiltration calibration and event-type decomposition; otherwise retire.
- H70-line global interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
  - Required change: stratum-local biological conditioning instead of global interaction coefficients.

## Navigation for iter_0035
- Keep `H87` as the primary backbone and test whether its descriptor signal is stable across seeds, not just strong in one seed.
- Allocate only one high-risk cross-model slot and enforce immediate fast-fail.
- Use one cheap orthogonal manifold screen each loop to prevent overfitting to the H70/H87 family.
