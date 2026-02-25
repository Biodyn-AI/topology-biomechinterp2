# Brainstormer Next Iteration Brief - iter_0020

## Objective
Run one focused packet that converts current structural positives into decision-grade utility evidence while opening one new high-upside topology axis.

## Required Execution Packet (3 slots)

### Slot A - High-probability discovery (`N198`)
- Hypothesis: support-weighted zigzag excess predicts utility uplift better than unweighted zigzag excess.
- Scope: domains `immune/lung/external_lung`, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `0,3,7,11`.
- Outputs:
  - `iterations/iter_0021/h46_weighted_zigzag_by_seed_layer_split.csv`
  - `iterations/iter_0021/h46_weighted_zigzag_domain_summary.csv`
  - `iterations/iter_0021/h46_weighted_zigzag_null_summary.csv`
- Null/control: target permutation preserving degree x support-quantile strata (`>=160` permutations/row).

### Slot B - High-risk/high-reward (`N200`)
- Hypothesis: bifiltration (`distance x support`) detects robust topological structure absent in single-parameter PH.
- Scope: start with seed42 across all 3 domains and both disjoint splits; expand to multi-seed only if pilot is directional.
- Outputs:
  - `iterations/iter_0021/h47_bifiltration_by_domain_layer_split.csv`
  - `iterations/iter_0021/h47_bifiltration_domain_summary.csv`
  - `iterations/iter_0021/h47_bifiltration_null_summary.csv`
- Null/control: support-label shuffle within degree/coexpression bins plus distance-only ablation.

### Slot C - Cheap broad-screen (`N208`)
- Hypothesis: cross-model top-k edge sets share enriched 3-node motifs beyond degree-preserving chance.
- Scope: top-k sweep (`50/100/200`) by domain and late layers (`7,11`) first; add early layer if runtime allows.
- Outputs:
  - `iterations/iter_0021/h48_cross_model_motif_overlap_by_domain_layer.csv`
  - `iterations/iter_0021/h48_cross_model_motif_overlap_summary.csv`
  - `iterations/iter_0021/h48_cross_model_motif_overlap_null_summary.csv`
- Null/control: degree-preserving random edge sets with matched k.

## Promotion Gates
1. `H46` promotion gate: weighted zigzag excess has positive held-out utility coupling in `>=2/3` domains and permutation-calibrated `p<0.05` after correction.
2. `H47` continuation gate: bifiltration signal is directional and significant in at least `2` domain-split units; otherwise stop this branch early.
3. `H48` keep gate: motif-overlap enrichment significant in `>=2` domains with robustness across at least two k values.

## Stop / Retire Rules
- Retire current intrinsic-dimension delta-R2 formulation if no redesigned target is introduced next iteration.
- Do not allocate a slot to rewiring-survival or GW-first correspondence in iter_0021.

## Minimal Recovery Plan (only if next executor gate unexpectedly fails)
1. Run Slot C only (`H48`) with one domain and one k (`100`) to guarantee machine artifact generation.
2. Emit mandatory files (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, one non-narrative CSV).
3. Re-run Slot A on a reduced grid (seed42, layers `0/11`) to re-establish valid comparative evidence quickly.
