# Executor Iteration Report - iter_0033

## Scope
This iteration executed a breadth-oriented 3-hypothesis packet aligned to the `N399/N407/N412` brief:
- `H82` (`N399`): local witness-cycle persistence on H70-like hotspot edges.
- `H83` (`N407`): cross-model pathway trajectory invariance (scGPT depth vs Geneformer token-rank strata).
- `H84` (`N412`): shortcut-bridge competition index broad screen.

## Command Trace
All experiment commands were run in the required environment:

```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0033/run_iter0033_screen.py
conda run -n subproject40-topology python iterations/iter_0033/run_iter0033_screen.py
```

No additional package installation was required.

## Quantitative Results

### H82 - Local Witness-Cycle Persistence on H70 Hotspots (`persistent_homology`, major-change rescue)
- Data scope: immune/lung/external_lung; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{7,11}`.
- Rows tested: `36`.
- Primary metric: `delta_auc_local_cycle_plus_h70_minus_h70`.
- Global mean primary metric: `+0.01595`.
- Domain-split means positive: `6/6`.
- Robustness metric: mean `null_gap_q95_local_cycle_hotspot_gap` positive in `4/6` domain-splits.
- Notable domain-split values:
  - `external_lung/source_disjoint`: mean delta `+0.02005`, mean null-gap `+0.02023`.
  - `lung/target_disjoint`: mean delta `+0.02058`, mean null-gap `+0.29375`.
  - `immune/source_disjoint`: mean delta `+0.02117`, mean null-gap `-0.10573` (still negative).
- Interpretation: this localized cycle-persistence formulation clears the pre-registered keep gate (`delta>0` in `6/6` and null-gap positive in `4/6`) and is currently the strongest branch in this packet.
- Artifacts:
  - `iterations/iter_0033/h82_local_witness_cycle_by_seed_layer_split.csv`
  - `iterations/iter_0033/h82_local_witness_cycle_domain_summary.csv`
  - `iterations/iter_0033/h82_local_witness_cycle_null_summary.csv`

### H83 - Cross-Model Pathway Trajectory Invariance (`cross_model_alignment`, rescue with changed endpoint)
- Data scope: seed42 pilot; domains immune/lung/external_lung; scGPT layers `{0,3,7,11}`; Geneformer token-rank quartile strata.
- Rows tested: `3` (one per domain).
- Primary metric: `trajectory_spearman_mean`.
- Global mean primary metric: `-0.07043`.
- Domains with positive null-gap (`trajectory_spearman - q95(null)`): `0/3`.
- Domain values:
  - `external_lung`: `-0.01872` (null-gap `-0.08154`).
  - `immune`: `-0.08615` (null-gap `-0.23832`).
  - `lung`: `-0.10641` (null-gap `-0.19283`).
- Secondary metrics: mean `trajectory_cka` is weak (`~0.077` across 3 domains), top-1 retrieval mixed (`0.36875` to `0.55`).
- Interpretation: negative pilot; invariance did not emerge above controls.
- Artifacts:
  - `iterations/iter_0033/h83_pathway_trajectory_invariance_by_domain.csv`
  - `iterations/iter_0033/h83_pathway_trajectory_invariance_domain_summary.csv`
  - `iterations/iter_0033/h83_pathway_trajectory_invariance_null_summary.csv`

### H84 - Shortcut-Bridge Competition Index (`graph_topology`, cheap broad screen)
- Data scope: seed42; immune/lung/external_lung; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_sbc_index_minus_baseline`.
- Global mean primary metric: `-0.02803`.
- Domain-split means positive: `0/6`.
- Domain-split mean null-gap (`delta - q95(null)`) positive: `0/6`.
- Range of mean deltas by domain-split: `-0.01935` to `-0.03553`.
- Interpretation: decisive negative for SBC as a standalone additive channel in this formulation.
- Artifacts:
  - `iterations/iter_0033/h84_shortcut_bridge_competition_by_domain_split_layer.csv`
  - `iterations/iter_0033/h84_shortcut_bridge_competition_domain_summary.csv`
  - `iterations/iter_0033/h84_shortcut_bridge_competition_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0033/iter0033_screen_summary.json`

## Iteration Decision
- `H82`: **promising**.
- `H83`: **negative**.
- `H84`: **negative**.

## Blockers
- No runtime or data blocker in this iteration.
- Method-level blocker persists for cross-model alignment objectives: this new invariance endpoint remained null-gap negative in all tested domains.
