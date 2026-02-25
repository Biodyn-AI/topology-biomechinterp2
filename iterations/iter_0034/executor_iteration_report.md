# Executor Iteration Report - iter_0034

## Scope
This iteration executed the 3-slot breadth packet from the prior brainstormer brief:
- `H85` (`N420`): dual-filtration local witness persistence (carry-over refinement from `H82`).
- `H86` (`N429`): cross-model barcode OT depth alignment (major-change rescue pilot).
- `H87` (`N433`): sparse descriptor blend breadth screen.

## Command Trace
All experiment commands were run in the required environment:

```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0034/run_iter0034_screen.py
conda run -n subproject40-topology python iterations/iter_0034/run_iter0034_screen.py
```

No additional package installation was required.

## Quantitative Results

### H85 - Dual-Filtration Local Witness Persistence (`persistent_homology`, refinement)
- Data scope: immune/lung/external_lung; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{7,11}`.
- Rows tested: `36`.
- Primary metric: `delta_auc_local_dual_filtration_plus_h70_minus_h70`.
- Global mean primary metric: `+0.00360`.
- Positive rows: `20/36`.
- Positive mean domain-splits: `5/6`.
- Robustness metric: positive mean `null_gap_q95_local_dual_filtration_hotspot_gap` in `3/6` domain-splits (keep gate required `>=4/6`).
- Interpretation: directional lift persists but robustness is incomplete; this misses the pre-registered null-gap gate by one domain-split.
- Artifacts:
  - `iterations/iter_0034/h85_dual_filtration_witness_by_seed_layer_split.csv`
  - `iterations/iter_0034/h85_dual_filtration_witness_domain_summary.csv`
  - `iterations/iter_0034/h85_dual_filtration_witness_null_summary.csv`

### H86 - Barcode OT Depth Alignment (`cross_model_alignment`, major-change rescue)
- Data scope: seed42 pilot across immune/lung/external_lung; scGPT layers `{0,3,7,11}` vs Geneformer token-rank quartile strata.
- Rows tested: `3` (one per domain).
- Primary metric: `barcode_ot_depth_alignment_score`.
- Global mean primary metric: `0.58192`.
- Control result: positive `null_gap_q95_barcode_ot_depth_alignment_score` in `0/3` domains.
- Domain null-gaps: external_lung `-0.05486`, immune `-0.06371`, lung `-0.08940`.
- Interpretation: despite moderate raw alignment score, it does not clear null controls in any tested domain; rescue fails keep gate.
- Artifacts:
  - `iterations/iter_0034/h86_barcode_ot_depth_alignment_by_domain.csv`
  - `iterations/iter_0034/h86_barcode_ot_depth_alignment_domain_summary.csv`
  - `iterations/iter_0034/h86_barcode_ot_depth_alignment_null_summary.csv`

### H87 - Sparse Descriptor Blend Breadth Screen (`manifold_distance`, new method)
- Data scope: seed42 breadth run across immune/lung/external_lung; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_descriptor_blend_minus_h70`.
- Global mean primary metric: `+0.08035`.
- Positive rows: `24/24`.
- Positive mean domain-splits: `6/6`.
- Robustness metric: positive mean `null_gap_q95_delta_auc` in `4/6` domain-splits (keep gate required `>=2/6`).
- Interpretation: strongest signal in this iteration; broad directional and null-gap support across splits, though row-level `p_best` remains conservative due low null budget.
- Artifacts:
  - `iterations/iter_0034/h87_sparse_descriptor_blend_by_domain_split_layer.csv`
  - `iterations/iter_0034/h87_sparse_descriptor_blend_domain_summary.csv`
  - `iterations/iter_0034/h87_sparse_descriptor_blend_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0034/iter0034_screen_summary.json`

## Iteration Decision
- `H85`: **neutral** (directional but misses null-gap keep gate).
- `H86`: **negative** (null-gap failure in all domains).
- `H87`: **promising** (passes breadth keep gate with strong effect size).

## Blockers
- No data/runtime blocker.
- Method-level blocker persists for cross-model alignment: the major-change OT rescue still failed null-gap criteria in all pilot domains.
