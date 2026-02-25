# Brainstormer Structured Feedback - iter_0018

## Inputs Inspected
- `iterations/iter_0018/executor_iteration_report.md`
- `iterations/iter_0018/executor_hypothesis_screen.json`
- `iterations/iter_0018/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0018/`:
  - `h37_consensus_tier_geometry_by_seed_layer_split.csv`
  - `h37_consensus_tier_geometry_domain_summary.csv`
  - `h37_consensus_tier_geometry_null_summary.csv`
  - `h38_id_distribution_moments_by_seed_layer_split.csv`
  - `h38_id_distribution_moments_fit_by_seed_split.csv`
  - `h38_id_distribution_moments_domain_summary.csv`
  - `h38_id_distribution_moments_null_summary.csv`
  - `h39_ph_feature_shuffle_by_seed_layer_split.csv`
  - `h39_ph_feature_shuffle_domain_summary.csv`
  - `h39_ph_feature_shuffle_null_summary.csv`
  - `iter0018_screen_summary.json`
  - `run_iter0018_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`

## Research Gate
- `passed_min_research_gate=true`.
- No recovery patch is required for this iteration.

## Iteration Signal Assessment
1. `H37` is decisively negative in the tested formulation.
- Mean high-tier minus low-tier uplift gap: `-0.00801`.
- Row/domain significance: `0/24` row-level, `0/6` domain-split Fisher.
- Structural issue: tier prevalence is imbalanced (several domain-splits have undefined tier-gap rows), so this exact binning is not worth another direct rerun.

2. `H38` has strong directional fit gain but weak null survival.
- Mean `delta R^2`: `+0.35673`, positive in `18/18` seed-split fits.
- Significance: only `1/18` row-level significant; `0/6` domain-split Fisher significant.
- Interpretation: likely overfitting to in-sample layer trajectories; requires out-of-sample tests before promotion.

3. `H39` is underpowered directional signal, not promotable yet.
- Mean H1 z-score: `+0.34579`, with `5/6` domain-splits positive mean z.
- Significance: `0/24` row-level and `0/6` domain-split Fisher.
- Method risk: single-seed packet and low PH null budget (`n=20`) make false directional positives likely.

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| Rewiring-null survival lineage (`H07/H09/H11/H12`) | Repeated calibrated failures across multiple null constructions | `retire_now` |
| GW-primary correspondence recovery (`H27/H29`) | Two direct failures on mapping quality and transfer utility | `retire_now` |
| Raw curvature/thinness direct-score variants (`H23/H30`) | Consistent below-baseline behavior | `retire_now` |
| Confidence-monotonicity module claim (`H19`) | Opposite-direction slope in both splits | `retire_now` |
| Current consensus-tier gap formulation (`H37`) | Negative effect plus sparse/imbalanced tier buckets | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why rescue is still plausible | Required major change | Action |
|---|---|---|---|
| Intrinsic-dimension mechanism (`H35/H38`) | Consistent directional structure exists but fails null robustness | Out-of-sample prediction (leave-layer/leave-seed out) plus stricter null | `rescue_once_with_major_change` |
| PH excess under feature-shuffle (`H39`) | Directional z-signal appears in most domain-splits | Multi-seed rerun + bootstrap + larger null budget + filtration variants | `rescue_once_with_major_change` |
| Anchor-regularized alignment causality (`H36`) | Utility gains are large but mechanism is unresolved | Anchor mismatch/dropout causal controls with fixed objective | `rescue_once_with_major_change` |
| Diffusion incremental branch (`H28/H31`) | Small positive means persist | Adaptive-time and biology-stratified interaction tests instead of pooled reruns | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Shift from coarse biological tiers to continuous support scoring (TRRUST/DoRothEA/STRING/GO) with interaction models.
2. Keep one topology-heavy high-risk experiment to avoid local optimization traps.
3. Use one cheap out-of-sample mechanism screen to quickly accept or kill `H38`-style claims.
4. Avoid another packet that spends a slot on rewiring/GW/raw-curvature retests.

## Minimal Recovery Plan (Only if a future gate fails)
1. Run seed42-only `N179` (continuous support interaction) on layers `{0,3,7,11}` for all domains/splits.
2. Run seed42-only `N174` (out-of-sample ID moments) with reduced permutations (`<=100`).
3. Emit mandatory report files plus one machine summary CSV/JSON, then resume full 3-hypothesis breadth packet.
