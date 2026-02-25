# Brainstormer Structured Feedback — iter_0017

## Inputs Inspected
- `iterations/iter_0017/executor_iteration_report.md`
- `iterations/iter_0017/executor_hypothesis_screen.json`
- `iterations/iter_0017/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0017/`:
  - `h34_convexity_detour_multiseed_by_seed_layer_split.csv`
  - `h34_convexity_detour_multiseed_domain_summary.csv`
  - `h34_convexity_detour_multiseed_null_summary.csv`
  - `h35_linearity_breakpoint_by_seed_domain_split.csv`
  - `h35_linearity_breakpoint_summary.csv`
  - `h35_linearity_breakpoint_null_summary.csv`
  - `h36_anchor_spectral_alignment_domain_summary.csv`
  - `h36_anchor_spectral_alignment_map_quality.csv`
  - `h36_anchor_spectral_alignment_null_summary.csv`
  - `iter0017_screen_summary.json`
  - `run_iter0017_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`

## Research Gate
- `passed_min_research_gate=true`.
- No emergency gate-recovery patch is required for this iteration.

## Iteration Signal Assessment
1. `H34` is directionally reproducible but low-effect.
- Mean incremental AUROC is positive in all `6/6` domain-split groups, but magnitude is small (`+0.00153` overall).
- Strongest support is concentrated in target-disjoint immune/lung only.
- Implication: keep active only if next run is biology-stratified (consensus-tier interactions), not another pooled rerun.

2. `H35` reveals a mechanism candidate but not robust yet.
- Piecewise depth structure is significant in all domains/splits.
- Split-specific breakpoint relocation is significant only in external-lung (`+4.33` layers, `p=0.0465`).
- Implication: treat as heterogeneity mechanism branch; prioritize variance/shape diagnostics and targeted replication.

3. `H36` is utility-positive but mechanism-ambiguous.
- Target AUROC gain is large and consistent (`+0.2008`, `3/3` domains positive).
- Label permutation is significant (`3/3`), but random-anchor null is invariant (`p=1.0`), so anchor causality is unresolved.
- Implication: one decisive causal-anchor test packet (dropout/mismatch) should determine promote vs retire.

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| GW-primary correspondence recovery (`H27/H29`) | Two controlled failures in correspondence and transfer utility | `retire_now` |
| Rewiring-null survival lineage (`H07/H09/H12`) | Repeated multi-iteration negatives across calibration variants | `retire_now` |
| Raw curvature/thinness direct edge scoring (`H23/H30` formulations) | Below-baseline or opposite-direction results | `retire_now` |
| Confidence-monotonicity module variant (`H19`) | Directional failure with no partial rescue signal | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why not retired yet | Required major change | Action |
|---|---|---|---|
| Anchor-regularized alignment (`H36` form) | Strong utility gains exist, but anchor-specific attribution failed | Run anchor-dropout and anchor-mismatch causal controls with fixed lambda and same eval split | `rescue_once_with_major_change` |
| Universal pooled intrinsic-coupling framing (`H04/H18/H21/H22/H35`) | Repeated domain/split heterogeneity | Model distribution-shape effects (ID variance/skew, tangent anisotropy), not pooled means | `rescue_once_with_major_change` |
| Pooled diffusion incremental claim (`H28/H31`) | Persistent weak positive means but limited robustness | Biology-tier and ontology-stratified interaction models with stronger nuisance controls | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Exploit the low-cost likely-win branch: biological-tier stratification of `H34` features (TRRUST/GO/STRING consensus).
2. Run one cheap mechanism screen around `H35` using ID variance/skew and tangent structure to test whether external-lung is a true regime shift.
3. Spend the high-risk slot on a decisive `H36` anchor-causality packet; if mismatch/dropout controls do not reduce utility, retire anchor claims.
4. Maintain breadth with at least one topology-filtration variant experiment in the same loop.

## Minimal Recovery Plan (Only if a Future Gate Fails)
1. Seed42-only quick packet on `immune` and `lung`:
- `N165` consensus-tier interaction test.
- `N160` ID variance/skew breakpoint screen.
2. Emit mandatory narrative files and one machine summary JSON/CSV.
3. Resume full 3-hypothesis breadth packet immediately after gate recovery.
