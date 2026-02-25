# Executor Iteration Report - iter_0035

## Scope
This iteration executed a breadth-oriented 3-slot packet with one carry-over refinement and two materially new methods:
- `H88` (`N448`, refinement): multiseed sparse-descriptor consensus robustness on top of the `H87` backbone.
- `H89` (`N441`, new method): local linearity phase-boundary manifold screen.
- `H90` (`N438`, new method): perturbation topology-stability screen.

To respect retirement guidance, this packet avoided another heavy cross-model alignment endpoint and instead prioritized robustness + orthogonal geometry/topology probes.

## Command Trace
All experiment commands were run in the required environment:

```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0035/run_iter0035_screen.py
conda run -n subproject40-topology python iterations/iter_0035/run_iter0035_screen.py
```

No package installation was required.

Paper/log maintenance command:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Quantitative Results

### H88 - Multiseed Sparse-Descriptor Consensus (`split_robustness`, refinement)
- Data scope: immune/lung/external_lung; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
- Rows tested: `72`.
- Primary metric: `delta_auc_sparse_descriptor_blend_minus_h70`.
- Mean primary metric: `+0.07603`; positive rows `72/72`.
- Positive mean domain-splits: `6/6`.
- Robustness: positive mean `null_gap_q95_delta_auc` in `5/6` domain-splits.
- Descriptor stability: mean nonzero-set Jaccard `0.49263` (median `0.48611`; `>=0.6` in `33.3%` layer-slices), mean sign agreement `0.99074`.
- Weakest slice: `immune/source_disjoint` mean null-gap `-0.00264`.
- Interpretation: strong reproducible utility signal survives nulls in most slices; mechanism-level descriptor-set stability is moderate rather than high, so this is promising but not yet a locked mechanistic core.
- Artifacts:
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_by_seed_split_layer.csv`
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_domain_summary.csv`
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_null_summary.csv`
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_stability.csv`

### H89 - Local Linearity Phase-Boundary Screen (`intrinsic_dimensionality`, new method)
- Data scope: seed42 breadth run across immune/lung/external_lung, splits `source_disjoint/target_disjoint`, layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_phase_boundary_minus_h70`.
- Mean primary metric: `+0.01676`; positive rows `20/24`; positive mean domain-splits `6/6`.
- Null robustness: positive mean `null_gap_q95_delta_auc` in `0/6` domain-splits (best split still negative: `lung/source_disjoint = -0.00970`).
- Interpretation: directional lift is small and collapses under controls (layer-order/feature/label nulls), giving decisive negative evidence for this phase-boundary formulation as a promotable objective.
- Artifacts:
  - `iterations/iter_0035/h89_phase_boundary_by_domain_split_layer.csv`
  - `iterations/iter_0035/h89_phase_boundary_domain_summary.csv`
  - `iterations/iter_0035/h89_phase_boundary_null_summary.csv`

### H90 - Perturbation Topology-Stability Screen (`topology_stability`, new method)
- Data scope: seed42 breadth run across immune/lung/external_lung, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
- Rows tested: `12`.
- Primary metric: `delta_auc_stability_blend_minus_h70`.
- Mean primary metric: `+0.00449`; positive rows `7/12`; positive mean domain-splits `4/6`.
- Stability trend: mean `stability_pos_minus_neg = +0.00793` (`8/12` rows positive).
- Null robustness: positive mean `null_gap_q95_delta_auc` in `0/6` domain-splits (best split `lung/source_disjoint = -0.01431`).
- Interpretation: perturbation-stability carries weak directional biology-consistent trend but does not survive null controls as a predictive utility feature; current formulation is negative.
- Artifacts:
  - `iterations/iter_0035/h90_topology_stability_by_domain_split_layer.csv`
  - `iterations/iter_0035/h90_topology_stability_domain_summary.csv`
  - `iterations/iter_0035/h90_topology_stability_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0035/iter0035_screen_summary.json`

## Iteration Decision
- `H88`: **promising** (strong multiseed utility and mostly positive null-gap support; descriptor-core stability is moderate and needs tightening).
- `H89`: **negative** (null-gap failure in all domain-splits).
- `H90`: **negative** (null-gap failure in all domain-splits).

## Blockers
- No data/runtime blocker.
- Non-blocking sklearn warnings about deprecating `penalty` argument were emitted during logistic fits; results were generated successfully.
