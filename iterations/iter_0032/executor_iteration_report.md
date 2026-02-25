# Executor Iteration Report - iter_0032

## Scope
This iteration executed a 3-hypothesis breadth screen mapped to brainstormer priorities:
- `H79` (`N395`): TF-module conditioned biological anchoring rescue.
- `H80` (`N392`): cross-model pathway-centroid alignment (edge-free).
- `H81` (`N389`): neighbor-dropout detour elasticity v2.

## Command Trace
All experiments were run in the required environment:

```bash
conda run --no-capture-output -n subproject40-topology python -m py_compile iterations/iter_0032/run_iter0032_screen.py
conda run --no-capture-output -n subproject40-topology python iterations/iter_0032/run_iter0032_screen.py
```

No additional package installation was required.

## Quantitative Results

### H79 - TF-Module Conditioned Support Calibration (`module_structure`, rescue-once major change)
- Data scope: immune/lung/external_lung; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{7,11}`.
- Rows tested: `36`.
- Primary metric: `delta_auc_module_minus_defect`.
- Global mean: `+0.03458`.
- Immune/source mean delta: `+0.03198`.
- Null robustness: domain-splits with positive mean `null_gap_q95` = `1/6`.
- Interaction diagnostic: global mean `interaction_delta_high_minus_low = -0.07111` (direction opposite rescue target).
- Interpretation: utility lift is directionally positive, but the null-gap and interaction gates are not met.
- Artifacts:
  - `iterations/iter_0032/h79_tf_module_conditioned_by_seed_layer_split.csv`
  - `iterations/iter_0032/h79_tf_module_conditioned_domain_summary.csv`
  - `iterations/iter_0032/h79_tf_module_conditioned_null_summary.csv`

### H80 - Pathway-Centroid Cross-Model Alignment (`cross_model_alignment`, major method change)
- Data scope: seed42 pilot; domains immune/lung/external_lung; layers `{7,11}`.
- Rows tested: `6`.
- Primary metric: `spearman_centroid_distance`.
- Global mean Spearman: `+0.15032`.
- Secondary metrics: mean `CKA=0.21263`, mean top-1 profile retrieval `0.38889`.
- Null robustness: positive mean `null_gap_q95_spearman` in `0/3` domains.
- Interpretation: pathway-level similarity is detectable, but it does not clear null-q95 robustness.
- Artifacts:
  - `iterations/iter_0032/h80_pathway_centroid_alignment_by_domain_layer.csv`
  - `iterations/iter_0032/h80_pathway_centroid_alignment_domain_summary.csv`
  - `iterations/iter_0032/h80_pathway_centroid_alignment_null_summary.csv`

### H81 - Neighbor-Dropout Detour Elasticity v2 (`manifold_distance`, major-change rescue)
- Data scope: seed42; all three domains; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_dropout_minus_baseline`.
- Global mean: `-0.01199`.
- Positive rows: `1/24`.
- Domain-splits with positive mean delta: `0/6`.
- Domain-splits with positive mean `null_gap_q95`: `0/6`.
- Interpretation: decisive negative result for this dropout-elasticity formulation.
- Artifacts:
  - `iterations/iter_0032/h81_neighbor_dropout_detour_elasticity_by_domain_split_layer.csv`
  - `iterations/iter_0032/h81_neighbor_dropout_detour_elasticity_domain_summary.csv`
  - `iterations/iter_0032/h81_neighbor_dropout_detour_elasticity_null_summary.csv`

## Iteration Summary Decision
- `H79`: **inconclusive** (positive directional lift, but fails robustness/interaction rescue gates).
- `H80`: **negative** (edge-free endpoint still fails null-gap robustness).
- `H81`: **negative** (utility and robustness both fail).

## Machine-Readable Packet Summary
- `iterations/iter_0032/iter0032_screen_summary.json`

## Blockers
- No data/runtime blocker this iteration.
- Method-level blocker remains for cross-model alignment: repeated null-gap failure despite endpoint redesign.
