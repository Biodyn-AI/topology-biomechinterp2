# Brainstormer Structured Feedback - iter_0041

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0041/executor_research_validation.json`).
- Full 3-slot execution packet is valid next loop.

## Iteration Evidence Snapshot
- `H106` (`persistent_homology`, `N538`) is decisively negative.
  - mean `delta_auc_string_triad_weighted_minus_h93 = -0.00118`
  - positive mean null-gap domain-splits: `0/6`
- `H107` (`topology_stability`, `N537`) is directional but non-robust.
  - mean `delta_auc_dfa_motif_minus_h70 = +0.04563`
  - positive mean null-gap domain-splits: `0/6`
  - strongest direction (`immune/source_disjoint`) still below q95 null.
- `H108` (`cross_model_alignment`, `N531`) is the only branch clearing the keep gate this loop.
  - mean `module_response_rank_spearman = +0.73159`
  - positive domain null-gap count: `2/3` (fails in immune)

## Cumulative Pattern Readout
- Repeated negatives are concentrated in additive filtration tweaks, static cross-model concordance endpoints, and standalone intrinsic-dimension utilities.
- Repeated directional-but-non-robust results suggest the nulls are defeating coarse featureization, not necessarily the family-level biological signal.
- Current strongest route to near-term discovery remains perturbation-conditioned cross-model structure (`H108` lineage), with one domain-specific failure mode to isolate.

## Stale Direction Triage
1. Additive weighted-filtration rescues on top of `H93` (`H94/H100/H103/H106`) -> `retire_now`.
Reason: repeated `0/6` null-gap survival under materially different additive constructions.

2. Static/role-graph/OT-depth cross-model concordance lineage (`H96/H99/H102` and earlier static endpoints) -> `retire_now`.
Reason: repeated domain-level null-gap failure despite endpoint resets.

3. Conditioned-null-as-rescue objective (`H105`) -> `retire_now`.
Reason: acts as stricter control; no evidence it rescues weak slices.

4. Standalone intrinsic-dimension additive utility lineage (`H89/H98` plus prior `H60/H63`) -> `retire_now`.
Reason: long negative tail with no robust keep-gate passes.

5. Coarse tokenized motif grammar lineage (`H104/H107`) -> `rescue_once_with_major_change`.
Required change: biologically anchored state alphabet + transition model that controls for state occupancy and module size.

6. `H108` immune failure mode -> `rescue_once_with_major_change` (domain-targeted only).
Required change: immune-specific perturbation panel balancing and module-size/variance matched nulls; no global redesign until immune is diagnosed.

## Portfolio Guidance
- Push novelty toward perturbation-topology coupling, multiparameter PH, geodesic sensitivity geometry, and biologically anchored finite-state mechanisms.
- Avoid another iteration of simple additive blends that already repeatedly failed null-gap gates.
- Keep one cheap broad-screen slot to maintain exploration breadth.

## Minimal Recovery Plan (only if gate flips false next loop)
1. Run the cheap broad-screen candidate first with reduced null budget (`16`) to restore valid machine artifacts quickly.
2. Run one anchor-adjacent in-model topology candidate on layers `{7,11}` only.
3. Defer heavy cross-model packets until gate returns `true`.
