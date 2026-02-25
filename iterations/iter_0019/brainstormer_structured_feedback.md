# Brainstormer Structured Feedback - iter_0019

## Inputs Inspected
- `iterations/iter_0019/executor_iteration_report.md`
- `iterations/iter_0019/executor_hypothesis_screen.json`
- `iterations/iter_0019/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0019/`:
  - `h40_support_interaction_by_seed_layer_split.csv`
  - `h40_support_interaction_domain_summary.csv`
  - `h40_support_interaction_null_summary.csv`
  - `h41_zigzag_persistence_by_seed_layer_split.csv`
  - `h41_zigzag_persistence_domain_summary.csv`
  - `h41_zigzag_persistence_null_summary.csv`
  - `h42_id_oos_by_seed_split.csv`
  - `h42_id_oos_domain_summary.csv`
  - `h42_id_oos_null_summary.csv`
  - `iter0019_screen_summary.json`
  - `run_iter0019_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery-only patch is required for this iteration.

## Iteration Signal Assessment
1. `H40` is the only clear positive branch this round.
- Mean interaction coefficient is positive (`+0.1317`) with Fisher-significant interaction support in `4/6` domain-split groups.
- External-lung and immune carry the effect; lung remains weak/non-supportive.
- Immediate implication: push this branch with better priors (true STRING scores) and stricter out-of-domain calibration.

2. `H41` remains method-limited and inconclusive.
- Direction is mostly positive (`5/6` domain-splits) but no domain-split survives layer-permutation Fisher aggregation (`0/6`).
- Since the run used a proxy (no true zigzag package), this line is not retired yet; it gets one rescue with a real zigzag implementation.

3. `H42` is negative for the broad H38 mechanism claim.
- Overall observed OOS `delta R^2` is strongly negative (`-10.70`) with instability dominated by extreme leave-seed-out failures.
- Interpretation: current pooled ID-moment framing is not transportable as-is and should not be promoted.

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| Rewiring-null survival lineage (`H07/H09/H11/H12`) | Repeated calibrated negatives across multiple rewiring controls | `retire_now` |
| GW-first correspondence recovery (`H27/H29`) | Two controlled failures on correspondence quality and edge-transfer utility | `retire_now` |
| Raw curvature/thinness direct scoring (`H23/H30`) | Below-baseline and wrong-direction behavior in repeated tests | `retire_now` |
| Coarse tiered biological stratification (`H19/H37` forms) | Opposite-direction or non-identifiable tier effects | `retire_now` |
| Broad pooled H38 mechanism claim in current metric form (`H42`) | OOS instability and negative mean holdout gain | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why still plausible | Required major change | Action |
|---|---|---|---|
| Split-zigzag branch (`H41`) | Directional positives persist despite proxy limitations | Install/use true zigzag persistence and repeat full controls | `rescue_once_with_major_change` |
| Anchor-causal interpretation (`H36`) | Large utility gains exist but mechanism attribution is unresolved | Adversarial anchor corruption/dropout controls with matched objective | `rescue_once_with_major_change` |
| Diffusion incremental branch (`H28/H31`) | Small positive effects recur but dilute when pooled | Adaptive-time plus ontology-stratified modeling instead of pooled averages | `rescue_once_with_major_change` |
| Cycle-consistency alignment (`H33`) | Structural consistency improved before utility decoupled | Topology-regularized objective tied directly to transfer AUROC | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Exploit `H40` momentum with stronger biological priors and heterogeneity modeling, not another pooled rerun.
2. Spend exactly one high-risk slot on true zigzag persistence to test genuinely new topology.
3. Use one cheap robust OOS screen to rapidly kill or rescue ID/local-linearity mechanisms without R2 blow-up artifacts.
4. Do not spend new slots on rewiring survival, GW-first mapping, or coarse tier binning.

## Minimal Recovery Plan (only if a future gate fails)
1. Run seed42-only `H43` (STRING + ontology support interaction) on layers `{0,3,7,11}` with reduced permutations (`<=80`).
2. Run seed42-only `H45` (robust OOS ID screen with MAE + winsorized `R^2`) with reduced null draws (`<=80`).
3. Emit required executor artifacts (`report`, `next_steps`, `hypothesis_screen`) plus one machine summary JSON, then return to full 3-slot packet.
