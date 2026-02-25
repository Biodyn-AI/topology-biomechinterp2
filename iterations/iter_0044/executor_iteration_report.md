# Executor Iteration Report - iter_0044

## Scope
Executed a breadth screen with two materially new hypotheses from the `iter_0042` roadmap that were not run in `iter_0043`:
- `H115` (`N558` style): layerwise tangent-subspace acceleration signal.
- `H116` (`N563` style): TRRUST-sign motif interaction with geometric baseline.

## Command Trace
```bash
# 1) Implemented experiment runner
#    iterations/iter_0044/run_iter0044_screen.py

# 2) Executed bounded screen in dedicated env
conda run -n subproject40-topology python iterations/iter_0044/run_iter0044_screen.py

# 3) Extracted split/domain summaries for reporting
conda run -n subproject40-topology python -c "import pandas as pd; b='iterations/iter_0044/';
for f in ['h115_tangent_acceleration_domain_summary.csv','h116_trrust_sign_motif_domain_summary.csv','h115_tangent_acceleration_by_domain_split.csv','h116_trrust_sign_motif_by_domain_split.csv']:
 df=pd.read_csv(b+f); print('=='+f+'=='); print(df.to_string(index=False)); print()"
```

## Quantitative Results From Artifacts

### H115 - Tangent-Subspace Acceleration (`family=manifold_distance`)
Primary metric: `delta_vs_h70 = AUROC(H70+tangent_accel_features) - AUROC(H70)`.

From `iterations/iter_0044/h115_tangent_acceleration_domain_summary.csv`:
- Mean delta across domain-splits: `-0.00622`.
- Mean null-gap across domain-splits: `+0.00156`.
- Direction pass: `2/6` domain-splits.
- Null-gap pass: `4/6` domain-splits.
- Worst split: lung/source-disjoint `delta=-0.02787`, `null_gap=-0.01547`.

Interpretation: this formulation does not show robust positive additive utility; classify as negative for promotion.

### H116 - TRRUST Sign-Motif Interaction (`family=module_structure`)
Primary metric: `delta_vs_h70 = AUROC(H70+motif_interactions) - AUROC(H70)`.

From `iterations/iter_0044/h116_trrust_sign_motif_domain_summary.csv`:
- Mean delta across domain-splits: `+0.07810`.
- Mean null-gap across domain-splits: `+0.06989`.
- Direction pass: `6/6` domain-splits.
- Null-gap pass: `6/6` domain-splits.
- Split deltas range: `+0.03165` to `+0.10294`.
- Motif coverage by split: `0.263` to `0.533` (`iterations/iter_0044/h116_trrust_sign_motif_by_domain_split.csv`).

Interpretation: strong and consistent positive signal with null survival in all domain-splits; promising branch.

## Artifacts Generated This Iteration
- `iterations/iter_0044/run_iter0044_screen.py`
- `iterations/iter_0044/iter0044_screen_summary.json`
- `iterations/iter_0044/h115_tangent_acceleration_by_domain_split.csv`
- `iterations/iter_0044/h115_tangent_acceleration_domain_summary.csv`
- `iterations/iter_0044/h115_tangent_acceleration_null_summary.csv`
- `iterations/iter_0044/h116_trrust_sign_motif_by_domain_split.csv`
- `iterations/iter_0044/h116_trrust_sign_motif_domain_summary.csv`
- `iterations/iter_0044/h116_trrust_sign_motif_null_summary.csv`

## Blockers / Notes
- No data/runtime blockers.
- Non-blocking warning flood from sklearn logistic deprecation (`penalty='l1'`) in this environment.
