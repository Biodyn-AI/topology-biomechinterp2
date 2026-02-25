# Executor Iteration Report - iter_0036

## Scope
This iteration executed a breadth-oriented 3-slot packet with one carry-over refinement and two materially new methods:
- `H91` (`N449`, refinement): stability-selected sparse-descriptor consensus on the `H88` branch.
- `H92` (`N452`, new method): scale-space lifetime trajectory descriptors.
- `H93` (`N458`, new method): confidence/sign-weighted filtration rescue.

Brainstormer carry-forward was addressed by executing `N449` and `N452` directly; the suggested cross-model `N456` slot was replaced with `N458` to avoid re-concentrating effort in a repeatedly negative/retired cross-model endpoint family during this bounded iteration.

## Command Trace
All experiment commands were run in the required environment:

```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0036/run_iter0036_screen.py
PYTHONWARNINGS=ignore conda run --no-capture-output -n subproject40-topology python iterations/iter_0036/run_iter0036_screen.py
```

No package installation was required.

Paper/log maintenance command:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Quantitative Results

### H91 - Stability-Selected Sparse Descriptor Consensus (`split_robustness`, refinement)
- Data scope: immune/lung/external_lung; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
- Rows tested: `72`.
- Primary metric: `delta_auc_stability_selected_blend_minus_h70`.
- Mean primary metric: `+0.07424`; positive rows `72/72`.
- Positive mean domain-splits: `6/6`.
- Robustness: positive mean `null_gap_q95_delta_auc` in `6/6` domain-splits.
- Stability: mean nonzero-set Jaccard `0.65046` (sign agreement `1.00000`), meeting the pre-registered `>=0.65` stability target.
- Random-subset control: stability-selected blend beat matched random feature subsets in `72/72` rows.
- Interpretation: this carry-over branch upgraded from moderate to high structural stability while preserving strong utility lift; null-gap support now clears all domain-splits.
- Artifacts:
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_by_seed_split_layer.csv`
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_domain_summary.csv`
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_null_summary.csv`
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_stability.csv`

### H92 - Scale-Space Lifetime Trajectory Descriptors (`topology_stability`, new method)
- Data scope: seed42 breadth run across immune/lung/external_lung, splits `source_disjoint/target_disjoint`, layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_scale_trajectory_minus_h70`.
- Mean primary metric: `+0.00386`; positive mean domain-splits `5/6`.
- Null robustness: positive mean `null_gap_q95_delta_auc` in `0/6` domain-splits.
- Strongest row still failed robustness: max row null-gap `-0.00350`.
- Interpretation: small directional lift exists but fully collapses under scale-order/feature/label controls; this exact trajectory-shape additive utility form is negative.
- Artifacts:
  - `iterations/iter_0036/h92_scale_space_lifetime_by_domain_split_layer.csv`
  - `iterations/iter_0036/h92_scale_space_lifetime_domain_summary.csv`
  - `iterations/iter_0036/h92_scale_space_lifetime_null_summary.csv`

### H93 - Confidence/Sign-Weighted Filtration Rescue (`persistent_homology`, new method)
- Data scope: seed42 breadth run across immune/lung/external_lung, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
- Rows tested: `12`.
- Primary metric: `delta_auc_weighted_filtration_minus_h70`.
- Mean primary metric: `+0.08443`; positive rows `12/12`.
- Positive mean domain-splits: `6/6`.
- Robustness: positive mean `null_gap_q95_delta_auc` in `6/6` domain-splits (`9/12` row-level positive null-gap).
- Interpretation: biologically weighted filtration is a high-upside positive branch with broad directional and null-gap support; current p-value resolution is coarse because null draws were intentionally bounded.
- Artifacts:
  - `iterations/iter_0036/h93_confidence_sign_weighted_filtration_by_domain_split_layer.csv`
  - `iterations/iter_0036/h93_confidence_sign_weighted_filtration_domain_summary.csv`
  - `iterations/iter_0036/h93_confidence_sign_weighted_filtration_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0036/iter0036_screen_summary.json`

## Iteration Decision
- `H91`: **promising** (strong utility + full domain-split null-gap support + stability target reached).
- `H92`: **negative** (null-gap failure in all domain-splits).
- `H93`: **promising** (strong directional lift and all-domain null-gap support; needs higher-resolution null follow-up).

## Blockers
- No data/runtime blocker.
- Non-blocking note: sklearn logistic warnings were suppressed for the main execution to keep command output machine-readable.
