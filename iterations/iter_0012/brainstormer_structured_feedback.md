# Brainstormer Structured Feedback — iter_0012

## Inputs Inspected
- `iterations/iter_0012/executor_iteration_report.md`
- `iterations/iter_0012/executor_hypothesis_screen.json`
- `iterations/iter_0012/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0012/`:
  - `h19_confidence_community_by_seed_layer_split_bin.csv`
  - `h19_confidence_community_layer_split_summary.csv`
  - `h19_confidence_community_monotonicity_tests.csv`
  - `h20_cross_model_transfer_alignment_summary.csv`
  - `h20_cross_model_transfer_by_domain_layer.csv`
  - `h20_cross_model_transfer_null_summary.csv`
  - `h21_local_reconstruction_edge_features.csv`
  - `h21_local_reconstruction_trend_summary.csv`
  - `h21_local_reconstruction_coupling_by_seed.csv`
  - `iter0012_screen_summary.json`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery patch is required for this iteration.

## Iteration Signal Assessment
1. `H19` failed in a directional way, not just by power.
- Mean confidence-tier slope is negative in both splits (source `-0.0771`, target `-0.0627`).
- Positive-slope layers are `0/12` in both splits.
- Tier prevalence is extremely skewed (`high` tier positive-rate ~`0.96`), so raw monotonic AUC slope is confounded by saturation and should not be rerun unchanged.

2. `H20` is the strongest active branch.
- Map-aware transfer is significant in all domains (`3/3`) with mean transferred-edge AUROC `0.5650` and mean Procrustes top-1 `0.3954`.
- Unsupervised Hungarian OT is effectively collapsed (mean top-1 `0.0024`, `0/3` significant).
- One OT metric (external-lung transfer AUROC) looks positive despite collapsed top-1, so future unsupervised tests must include map-quality gates, not only downstream AUROC.

3. `H21` reveals a useful mechanism shift.
- Source-disjoint edge signal is positive (mean AUC `0.5331`), target-disjoint is negative (mean AUC `0.4780`).
- Target coupling to `H13` geodesic lift is inverse and significant (mean seed rho `-0.4079`, two-sided `p=0.0190`).
- Late layers (9-11) are the strongest negative zone in both splits, suggesting a depth-phase effect rather than pure noise.

## Stale Direction Triage

### `retire_now`
1. Rewiring-survival primary branch (`H07/H09/H12` lineage).
- Repeated calibrated negatives across `iter_0006`-`iter_0008`, no rescue signal.

2. Distortion-lower-tail rescue for rewiring branch.
- Repeated non-significance and no directional correction.

3. Plain Hungarian OT as a standalone unsupervised alignment method.
- Collapsed top-1 recovery in `iter_0012` indicates this exact method should not be repeated.

### `rescue_once_with_major_change`
1. Confidence-monotonic module anchoring (`H19` current form).
- Allow one rescue only with independent priors + prevalence-adjusted modeling.

2. Intrinsic branch framed as "positive coupling" (`H21` current hypothesis sign).
- Recast as inverse-coupling and test cross-domain replication before deciding promotion/retirement.

3. Coarse disagreement-bin cross-model trend (`H15` style).
- Only re-open with per-edge controls and explicit domain interaction terms.

4. Bridge-conditioned rewiring explanation (`H11`).
- Only if a fixed-k design creates bridge/non-bridge strata within each split.

## Navigation Guidance for iter_0013
1. Spend most budget on cross-model structure transfer (`H20` successor) and biologically anchored reinterpretation of `H19`/`H21`, not additional rewiring stress tests.
2. Require dual gates for cross-model unsupervised alignment: map-quality (top-1 / cycle-consistency) and biological transfer (edge AUROC / enrichment).
3. Treat split asymmetry as signal, not nuisance: explicitly model source vs target interactions and depth-phase transitions.
4. Keep one cheap broad-screen in each packet to maintain throughput and avoid single-point failure.

## Minimal Recovery Plan (Contingency Only)
- Not needed this iteration because gate passed.
- If a future gate fails, run one compact packet with:
1. A single low-cost split-aware test (`H21` inverse-coupling screen).
2. One explicit null/permutation control.
3. Mandatory artifacts (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, at least one machine CSV).
