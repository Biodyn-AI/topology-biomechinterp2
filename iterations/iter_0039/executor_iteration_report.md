# Executor Iteration Report - iter_0039

## Scope
This iteration executed a 3-slot breadth packet with two new persistent-homology methods and one rescue-once cross-model reset:
- `H100` (`persistent_homology`, new method): relative persistence contrast vs matched background complexes (`N493`-style).
- `H101` (`persistent_homology`, new method): persistence derivative-spectrum descriptors (`N497`-style).
- `H102` (`cross_model_alignment`, major-change rescue): OT + monotone depth warp on GO-module role manifolds (`N501`-style fast-fail pilot).

## Command Trace
All experiment commands were run in the required environment.

```bash
conda run --no-capture-output -n subproject40-topology python -m py_compile iterations/iter_0039/run_iter0039_screen.py
PYTHONWARNINGS=ignore conda run --no-capture-output -n subproject40-topology python iterations/iter_0039/run_iter0039_screen.py
```

A null-quality issue was found in `H102` (`module_membership_permutation` produced `NaN` because the permutation builder exhausted symbols under overlapping module sizes). The null generator was patched and the full screen rerun:

```bash
conda run --no-capture-output -n subproject40-topology python -m py_compile iterations/iter_0039/run_iter0039_screen.py
PYTHONWARNINGS=ignore conda run --no-capture-output -n subproject40-topology python iterations/iter_0039/run_iter0039_screen.py
```

Metric extraction command used for this report:

```bash
conda run --no-capture-output -n subproject40-topology python - <<'PY'
import json
from pathlib import Path
import pandas as pd
b=Path('iterations/iter_0039')
h100=pd.read_csv(b/'h100_relative_persistence_contrast_by_domain_split_layer.csv')
h100s=pd.read_csv(b/'h100_relative_persistence_contrast_domain_summary.csv')
h101=pd.read_csv(b/'h101_persistence_derivative_spectrum_by_domain_split_layer.csv')
h101s=pd.read_csv(b/'h101_persistence_derivative_spectrum_domain_summary.csv')
h102=pd.read_csv(b/'h102_ot_monotone_depth_warp_by_domain.csv')
h102s=pd.read_csv(b/'h102_ot_monotone_depth_warp_domain_summary.csv')
print(json.dumps({
  'h100_mean_delta': float(h100['delta_auc_relative_ph_minus_h93'].mean()),
  'h100_pos_null_domain_splits': int((h100s['mean_null_gap_q95_delta_auc']>0).sum()),
  'h101_mean_delta': float(h101['delta_auc_persistence_derivative_minus_h70'].mean()),
  'h101_pos_null_domain_splits': int((h101s['mean_null_gap_q95_delta_auc']>0).sum()),
  'h102_mean_concordance': float(h102['module_persistence_ot_concordance'].mean()),
  'h102_pos_domains': int((h102s['mean_null_gap_q95_concordance']>0).sum())
}, indent=2))
PY
```

Paper compile command used this iteration:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

No package installation was required.

## Quantitative Results

### H100 - Relative Persistence Contrast (`persistent_homology`, new method)
- Data scope: seed42, domains `immune/lung/external_lung`, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
- Rows tested: `12`.
- Primary metric: `delta_auc_relative_ph_minus_h93`.
- Mean primary metric: `-0.00188` (positive rows `3/12`; positive mean domain-splits `2/6`).
- Robustness: mean `null_gap_q95_delta_auc = -0.02384`; positive null-gap rows `0/12`; positive mean null-gap domain-splits `0/6`.
- Interpretation: relative-persistence contrast in this formulation did not improve over the H93 backbone and failed all null-gap gates.
- Artifacts:
  - `iterations/iter_0039/h100_relative_persistence_contrast_by_domain_split_layer.csv`
  - `iterations/iter_0039/h100_relative_persistence_contrast_domain_summary.csv`
  - `iterations/iter_0039/h100_relative_persistence_contrast_null_summary.csv`

### H101 - Persistence Derivative Spectrum (`persistent_homology`, new method)
- Data scope: seed42, domains `immune/lung/external_lung`, splits `source_disjoint/target_disjoint`, layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_persistence_derivative_minus_h70`.
- Mean primary metric: `+0.00621` (positive rows `12/24`; positive mean domain-splits `4/6`).
- Robustness: mean `null_gap_q95_delta_auc = -0.01682`; positive null-gap rows `1/24`; positive mean null-gap domain-splits `0/6`.
- Interpretation: directional signal exists in subsets (strongest in lung), but robustness fails globally under quantile-order, derivative-sign, and label nulls.
- Artifacts:
  - `iterations/iter_0039/h101_persistence_derivative_spectrum_by_domain_split_layer.csv`
  - `iterations/iter_0039/h101_persistence_derivative_spectrum_domain_summary.csv`
  - `iterations/iter_0039/h101_persistence_derivative_spectrum_null_summary.csv`

### H102 - OT + Monotone Depth Warp (`cross_model_alignment`, rescue-once major change)
- Data scope: seed42 pilot, domains `immune/lung/external_lung`, one row per domain (joint layers `{7,11}`).
- Rows tested: `3`.
- Primary metric: `module_persistence_ot_concordance`.
- Mean primary metric: `+0.57065`.
- Robustness: mean `null_gap_q95_concordance = -0.09697`; positive null-gap domains `0/3`.
- Auxiliary: mean warped OT transport cost `1.69768`; mean top-module overlap Jaccard `0.32322`; mean alignment RMSE `0.85326`.
- Interpretation: despite moderate raw concordance, the cross-model branch still fails domain-level q95 null-gap criteria and does not clear fast-fail rescue gate.
- Artifacts:
  - `iterations/iter_0039/h102_ot_monotone_depth_warp_by_domain.csv`
  - `iterations/iter_0039/h102_ot_monotone_depth_warp_domain_summary.csv`
  - `iterations/iter_0039/h102_ot_monotone_depth_warp_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0039/iter0039_screen_summary.json`

## Iteration Decision
- `H100`: **negative**.
- `H101`: **inconclusive** (directional but non-robust).
- `H102`: **negative** (cross-model rescue gate failed at `0/3` domains).

## Blockers
- No runtime/data blocker.
- Method-level blocker remains for cross-model alignment: repeated major endpoint resets continue to miss domain-level null-gap robustness.
