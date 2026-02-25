# Brainstormer Structured Feedback - iter_0033

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0033/executor_research_validation.json`).
- No recovery-only mode required.

## Iteration Readout
- `H82` is the only promotable branch in this packet.
  - Mean `delta_auc_local_cycle_plus_h70_minus_h70 = +0.01595`.
  - Positive mean delta in `6/6` domain-splits.
  - Positive mean null-gap in `4/6` domain-splits.
  - Weak slices are concentrated in `immune/source_disjoint` and `external_lung/target_disjoint` (negative mean null-gap).
- `H83` is decisively negative.
  - Mean trajectory Spearman `-0.07043`.
  - Positive null-gap domains `0/3`.
- `H84` is decisively negative.
  - Mean `delta_auc_sbc_index_minus_baseline = -0.02803`.
  - Positive mean delta and null-gap domain-splits `0/6` and `0/6`.

## Stale Direction Triage
- `cross_model_alignment` utility/invariance endpoints in current form -> `retire_now`.
  - Evidence: recent chain `H71/H74/H77/H80/H83` is consistently negative with null-gap failures, and cumulative family count is dominated by non-promotable outcomes.
  - Reopen only once with a materially different topology-level objective and strict domain-level null-gap gate.
- Standalone additive `graph_topology` scores (curvature/SBC style) -> `retire_now`.
  - Evidence: `H61` and `H84` are uniformly negative; earlier direct-curvature variants were also negative.
- Standalone intrinsic-dimension utility-lift branch -> `retire_now`.
  - Evidence: `H54/H60/H63/H66` are consecutive negatives with low rescue yield.
- H70/H82 biological overlay formulations in current global-interaction form -> `rescue_once_with_major_change`.
  - Evidence: base geometry is positive, but interaction-style overlays (`H73/H76/H79`) repeatedly miss null-gap robustness.

## Active Signal Backbone
- Keep `H70 -> H82` as the core discovery lane.
- Prioritize local/topological mechanism tests over global additive features.
- Use biological anchoring as stratification/enrichment diagnostics, not as broad global reweighting.

## Immediate Navigation Rules
- Allocate exactly one slot to H82-line confirmation/extension each loop until null-gap failures are localized or resolved.
- Reserve one slot for a single high-risk cross-model redesign (topology-level, not transfer-utility).
- Reserve one slot for a cheap orthogonal broad-screen to avoid local overfitting to H82.
