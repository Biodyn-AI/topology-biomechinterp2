# Brainstormer Structured Feedback - iter_0039

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0039/executor_research_validation.json`).
- Full 3-slot execution is allowed next loop.

## Iteration Evidence Snapshot
- `H100` (`persistent_homology`) is negative.
  - mean `delta_auc_relative_ph_minus_h93 = -0.00188`
  - positive mean null-gap domain-splits: `0/6`
  - decision in screen: retired.
- `H101` (`persistent_homology`) is directional but non-robust.
  - mean `delta_auc_persistence_derivative_minus_h70 = +0.00621`
  - positive mean domain-splits: `4/6`
  - positive mean null-gap domain-splits: `0/6`
  - only one row cleared positive null-gap (`lung`, `target_disjoint`, `layer=0`, `+0.0019`).
- `H102` (`cross_model_alignment`) failed rescue gate.
  - mean concordance `+0.57065`
  - mean null-gap(q95) `-0.09697`
  - positive null-gap domains: `0/3`
  - decision in screen: retired.

## Cumulative Pattern (Master Log + Paper + Hypothesis Screens)
- Current robust in-model anchors remain `H91` and `H93`.
- Cross-model branch is stale in its current objective family: `11` consecutive negatives (`H65` -> `H102`).
- Additive standalone intrinsic-dimension lineage is stale: `6` consecutive negatives at tail (`H42/H54/H60/H63/H66/H89/H98` with no robust promotion in recent cycles).
- Additive bridge-curvature lineage (`H95/H97`) is a repeated directional-but-non-robust pattern (`0/6` positive mean null-gap domain-splits across two runs).
- Recent additive persistent-homology extensions (`H100`, additive `H101`) do not clear robustness gates.

## Stale Direction Triage
1. Cross-model unsupervised concordance/OT-depth endpoints (`H65/H68/H71/H74/H77/H80/H83/H86/H96/H99/H102`) -> `retire_now`.
Reason: repeated domain-level null fragility (`0/3` in latest resets) despite major endpoint changes.

2. Exact `H100` relative-background formulation -> `retire_now`.
Reason: under baseline and null-gap negative in all domain-splits.

3. Additive `H101` derivative-spectrum formulation -> `rescue_once_with_major_change`.
Required change: interaction-only design on top of `H91/H93` (no standalone additive blend).

4. Additive bridge-curvature utility lineage (`H95/H97`) -> `retire_now`.
Reason: persistent `0/6` null-gap support after stricter rewiring controls.

5. Standalone/additive intrinsic-dimension utility lineage (`H54/H60/H63/H66/H89/H98`) -> `retire_now`.
Reason: repeated null failure and no recent robust support.

6. Flat global pooled biological overlays (`H73/H76/H79` style) -> `rescue_once_with_major_change`.
Required change: hierarchical local partial pooling (GO/Cell Ontology/module level), not global pooled coefficients.

7. Standalone additive topology-stability trajectory forms (`H90/H92`) -> `retire_now`.
Reason: weak directional effects repeatedly collapse under null controls.

## Navigation for Next Loop
- Use `2/3` slots for in-model geometry/topology hypotheses explicitly anchored to `H91/H93`.
- Allow at most `1/3` slot for cross-model, only with a fundamentally new shared anchor and a hard fast-fail (`>=2/3` domains must clear null-gap).
- Prefer contrastive/interaction objectives over additive overlays.
- Raise null resolution for any rescue candidate before promotion claims.

## Contingency Recovery Plan (only if a future gate flips false)
1. Run one cheap broad screen first (motif grammar or interaction-only derivative rescue) with reduced null budget (`16`) to maintain valid machine outputs.
2. Run one high-probability in-model candidate (`H91/H93`-anchored) on `layers {7,11}` only.
3. Skip cross-model until gate returns `true`.
