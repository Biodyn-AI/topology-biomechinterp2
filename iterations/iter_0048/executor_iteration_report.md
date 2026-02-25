# Executor Iteration Report: iter_0048

## Objective
Run one breadth-oriented screening packet with one carry-over refinement and two materially changed/novel slots:
- `H127` (`module_structure`, refinement of `H124` / `N641`-style): signed motif-community + STRING hardening with explicit GO co-membership interactions and GO-membership null.
- `H128` (`graph_topology`, novel family): graph curvature/community surrogate descriptors over residual kNN graphs.
- `H129` (`manifold_distance`, changed-method rescue of `H126` / `N634`-style): multi-scale torsion spectrum (`k={8,12,16}`) with scale-order null.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0048/run_iter0048_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0048/run_iter0048_screen.py
conda run -n subproject40-topology python iterations/iter_0048/run_iter0048_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary metric | Result | Null robustness | Decision |
|---|---|---|---:|---:|---|
| H127 | module_structure | mean `delta_vs_h70` | `+0.13222` (9 rows; seed42 x 3 domains x 3 splits) | positive mean null-gap domain-splits: `2/9`; lung dual-axis mean null-gap `-0.00596` | neutral |
| H128 | graph_topology | mean `delta_vs_h70` | `+0.00753` (12 rows; 3 domains x 2 splits x 2 layers) | positive mean null-gap domain-splits: `0/6` | negative |
| H129 | manifold_distance | mean `delta_vs_h70` | `+0.02100` (12 rows; 3 domains x 2 splits x 2 layers) | positive mean null-gap domain-splits: `0/6` | negative |

### H127 details (`module_structure`, GO-augmented hardening)
- Directionality remained strong (`9/9` positive row deltas; `9/9` positive mean delta domain-splits).
- Null-calibrated robustness remained limited (`2/9` positive mean null-gap domain-splits; row-level positive null-gap `2/9`).
- Hard slice stayed unresolved: `lung/dual_axis_disjoint` mean null-gap `-0.00596`.
- Added null family this iteration: `go_membership_permutation_within_degree_strata` (in addition to TF-sign, motif-decoy, STRING-bin, label nulls; 32 each).
- Artifacts:
  - `iterations/iter_0048/h127_signed_string_go_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0048/h127_signed_string_go_hardening_domain_summary.csv`
  - `iterations/iter_0048/h127_signed_string_go_hardening_null_summary.csv`

### H128 details (`graph_topology`, novel family)
- Weak directional signal (`mean delta +0.00753`; `7/12` positive rows; `4/6` positive mean-delta domain-splits).
- Robustness failed decisively (`0/6` positive mean null-gap domain-splits; row-level positive null-gap `2/12`).
- Tested nulls: curvature shuffle within degree bins, community/topology feature shuffle, degree-bin edge-feature rewiring, label permutation (24 each).
- Interpretation: this surrogate endpoint adds little beyond H70 under strict controls; retire this formulation unless topology features are redefined.
- Artifacts:
  - `iterations/iter_0048/h128_graph_topology_surrogate_by_domain_split_layer.csv`
  - `iterations/iter_0048/h128_graph_topology_surrogate_domain_summary.csv`
  - `iterations/iter_0048/h128_graph_topology_surrogate_null_summary.csv`

### H129 details (`manifold_distance`, multi-scale torsion spectrum)
- Directional lift remained broad (`mean delta +0.02100`; `11/12` positive rows; `5/6` positive mean-delta domain-splits).
- Strict-null robustness failed (`0/6` positive mean null-gap domain-splits; row-level positive null-gap `1/12`).
- New null added this iteration: `scale_order_permutation` (with path-reversal, endpoint-swap, label nulls; 24 each).
- Interpretation: multi-scale torsion captures directional structure but still collapses under null calibration in this seed42 packet.
- Artifacts:
  - `iterations/iter_0048/h129_multiscale_torsion_by_domain_split_layer.csv`
  - `iterations/iter_0048/h129_multiscale_torsion_domain_summary.csv`
  - `iterations/iter_0048/h129_multiscale_torsion_null_summary.csv`

## Interpretation
- `H127` stays the strongest active branch in this iteration, but GO augmentation did not solve the hard null-gap bottleneck; promotion criteria remain unmet.
- `H128` provides negative evidence for this graph-topology surrogate endpoint.
- `H129` provides negative evidence for this multi-scale torsion endpoint under current nulls, despite directional lift.

## Blockers / Runtime Notes
- No missing-data or package blockers.
- High-volume scikit-learn warning spam (`penalty`/`l1_ratio`) occurred during logistic-CV loops; runs completed and outputs were written.

## Machine-Readable Summary
- `iterations/iter_0048/iter0048_screen_summary.json`

## Paper Update
- Updated `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0048`.
- Compiled `paper/autoloop_research_paper.pdf` via `latexmk`.
