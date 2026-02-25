# Brainstormer Structured Feedback — iter_0016

## Inputs Inspected
- `iterations/iter_0016/executor_iteration_report.md`
- `iterations/iter_0016/executor_hypothesis_screen.json`
- `iterations/iter_0016/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0016/`:
  - `h31_diffusion_incremental_by_seed_layer_split.csv`
  - `h31_diffusion_incremental_domain_summary.csv`
  - `h31_diffusion_incremental_null_summary.csv`
  - `h32_convexity_detour_by_seed_layer_split.csv`
  - `h32_convexity_detour_domain_summary.csv`
  - `h32_convexity_detour_null_summary.csv`
  - `h33_cycle_consistent_alignment_domain_summary.csv`
  - `h33_cycle_consistent_alignment_map_quality.csv`
  - `h33_cycle_consistent_alignment_null_summary.csv`
  - `iter0016_screen_summary.json`
  - `run_iter0016_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery patch is required in this iteration.

## Iteration Signal Assessment
1. `H32` is the current exploitation branch and should get first compute.
- Mean delta vs geodesic baseline is strong (`+0.01706`) with `4/6` domain-split Fisher-significant groups.
- Lung and immune carry most of the signal; external-lung source is flat/negative.
- Immediate implication: prioritize multi-seed replication plus incremental-vs-diffusion testing before broad expansion.

2. `H31` is real but constrained, not broad.
- All domain-split means are positive, but significance concentrates in immune and one external-lung split (`3/6` significant groups).
- External-lung target and both lung splits are weak under current confound adjustment.
- Immediate implication: continue only with a materially harder confound design or ontology-stratified framing.

3. `H33` remains utility-negative despite better structural consistency.
- Cycle-return improved (`+0.0269`, `p=0.0062`) but edge-transfer AUROC did not (`delta ~ -4.55e-05`, `0/3` domains significant).
- Null edge-AUROC means are effectively identical to observed, indicating the current objective is optimizing the wrong target for downstream utility.
- Immediate implication: block further spend unless objective includes biological/utility anchors.

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| GW-primary correspondence recovery (`H27/H29`) | Two controlled failures on both correspondence and transfer utility | `retire_now` |
| Rewiring-null survival lineage (`H07/H09/H12`) | Repeated multi-iteration negatives with no trend toward rescue | `retire_now` |
| Raw Forman-curvature enrichment (`H23` form) | Opposite-direction and below-chance behavior across domains | `retire_now` |
| Triangle-thinness/hyperbolicity edge score (`H30` form) | Mostly below chance and below geodesic baseline | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why not retired | Required major change | Action |
|---|---|---|---|
| `H33` cycle-consistent alignment as currently optimized | Structural consistency improves, but utility does not | Optimize directly for transfer utility with anchor regularization and holdout-anchor validation | `rescue_once_with_major_change` |
| Diffusion as pooled global claim (`H28/H31` framing) | Direction stays positive but robustness is domain-concentrated | Shift to ontology-stratified mixed models with stronger nuisance controls and interaction terms | `rescue_once_with_major_change` |
| Universal-sign intrinsic coupling (`H04/H18/H21/H22`) | Persistent domain/split sign heterogeneity | Model heterogeneity explicitly using ID-shape and anisotropy, not mean-only coupling | `rescue_once_with_major_change` |
| Pooled geometry x prior interaction (`H26` form) | Calibration hints exist but interaction is unstable | Use consensus prior tiers (TRRUST/GO/STRING) and per-ontology strata with prevalence control | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Exploit `H32` immediately, but force an incremental test against diffusion-adjusted baselines to avoid rediscovering the same manifold signal twice.
2. Treat `H31` as a targeted rescue branch only; do not run another broad pooled rerun without a major framing change.
3. Continue cross-model work only on objectives tied to downstream transfer, not cycle quality alone.
4. Increase biological anchoring density in every new branch to reduce ambiguous geometry-only wins.

## Minimal Recovery Plan (Only if a Future Gate Fails)
1. Run one seed42-only `H32` replication packet on `immune` and `lung`, both disjoint splits, layers `0/3/7/11`.
2. Emit mandatory files plus one machine summary (`h32_quick_recovery_summary.csv`).
3. Resume full 3-hypothesis packet after gate recovery.
