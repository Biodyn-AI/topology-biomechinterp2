# Brainstormer Structured Feedback - iter_0040

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0040/executor_research_validation.json`).
- Full 3-slot execution is valid next loop.

## Iteration Evidence Snapshot
- `H103` (`persistent_homology`, interaction-only derivative rescue) is negative.
  - mean `delta_auc_interaction_derivative_minus_h91_h93 = -0.00304`
  - positive mean null-gap domain-splits: `0/6`
- `H104` (`manifold_distance`, depth motif grammar) is negative.
  - mean `delta_auc_motif_grammar_minus_h70 = -0.00908`
  - positive mean null-gap domain-splits: `0/6`
- `H105` (`null_sensitivity`, STRING-conditioned null rescue) is negative for rescue intent.
  - mean conditioned-minus-unconditioned null-gap `= -0.05125`
  - positive conditioned-gain domain-splits: `0/6`

## Cumulative Pattern Readout
- Active robust anchors remain `H91` and `H93`; no new robust branch emerged in `iter_0037` through `iter_0040`.
- Recent packet trend (`iter_0037`-`iter_0040`): `10` negatives, `2` inconclusive, `0` promising.
- Cross-model static concordance lineage remains stale despite repeated endpoint resets.
- Additive/overlay formulations repeatedly show directional lift without null-gap survival.

## Stale Direction Triage
1. Cross-model static concordance/OT-depth endpoint lineage (`H65/H68/H71/H74/H77/H80/H83/H86/H96/H99/H102`) -> `retire_now`.
Reason: repeated `0/3` domain-level null-gap support under materially changed objectives.

2. Derivative-spectrum additive/interactions-on-same-backbone (`H101/H103`) -> `retire_now`.
Reason: two consecutive failures with `0/6` positive mean null-gap domain-splits.

3. Conditioned-null-as-rescue objective (`H105`) -> `retire_now`.
Reason: conditioned calibration consistently tightens nulls and reduces margins; keep only as control.

4. Bridge-curvature additive utility lineage (`H95/H97`) -> `retire_now`.
Reason: repeated strong directionality with persistent robustness failure (`0/6`).

5. Standalone/additive intrinsic-dimension utility lineage (`H54/H60/H63/H66/H89/H98`) -> `retire_now`.
Reason: long negative tail with no robust rescue.

6. Rewiring-survival-as-primary-objective lineage (`H07`-`H12`) -> `retire_now`.
Reason: repeated non-supportive results after calibration and constrained-null variants.

7. Flat pooled module-support overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Required change: hierarchical pooling by GO/Cell Ontology with partial pooling priors, not global pooled interactions.

8. Discrete motif grammar as tested (`H104`) -> `rescue_once_with_major_change`.
Required change: continuous state-space motif model (HMM/DFA likelihood ratio with matched controls), not coarse token counts.

## Navigation for iter_0041
- Allocate exactly one slot to high-probability, anchor-adjacent innovation (`H93`-adjacent, biologically anchored).
- Allocate exactly one slot to high-risk cross-model hypothesis with a hard fast-fail gate.
- Allocate one cheap broad-screen slot in algorithmic/motif space.
- Enforce hard gate: no promotion without positive mean null-gap in at least half of tested domain-splits.

## Minimal Recovery Plan If Gate Fails Later
1. Run one cheap broad-screen (`N537`-style motif automata) with reduced null budget (`16`) to keep machine-output cadence valid.
2. Run one anchor-adjacent low-cost hypothesis (`N538`-style STRING triad weighted filtration) on layers `{7,11}` only.
3. Skip cross-model slot until `passed_min_research_gate` returns `true`.
