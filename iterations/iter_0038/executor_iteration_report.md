# Executor Iteration Report - iter_0038

## Scope
This iteration executed a bounded 3-slot hypothesis packet with one carry-over rescue and two materially changed methods:
- `H97` (`graph_topology`, refinement of `H95`): calibrated bridge-curvature blend with stricter structure-matched rewiring nulls.
- `H98` (`intrinsic_dimensionality`, new method): multi-radius intrinsic-dimension heterogeneity entropy features.
- `H99` (`cross_model_alignment`, new method / structural reset): cross-model GO-module role-graph alignment.

The packet matches the exploration policy (2-3 hypotheses; at most one carry-over refinement).

## Command Trace
All research commands were run in the required environment:

```bash
conda run --no-capture-output -n subproject40-topology python -m py_compile iterations/iter_0038/run_iter0038_screen.py
PYTHONWARNINGS=ignore conda run --no-capture-output -n subproject40-topology python iterations/iter_0038/run_iter0038_screen.py
```

Metric extraction for report values:

```bash
conda run --no-capture-output -n subproject40-topology python - <<'PY'
import json
from pathlib import Path
import pandas as pd
b=Path('iterations/iter_0038')
h97=pd.read_csv(b/'h97_graph_bridge_calibrated_by_domain_split_layer.csv')
h97s=pd.read_csv(b/'h97_graph_bridge_calibrated_domain_summary.csv')
h98=pd.read_csv(b/'h98_id_entropy_by_domain_split_layer.csv')
h98s=pd.read_csv(b/'h98_id_entropy_domain_summary.csv')
h99=pd.read_csv(b/'h99_cross_model_role_graph_by_domain_layer.csv')
h99s=pd.read_csv(b/'h99_cross_model_role_graph_domain_summary.csv')
print(json.dumps({
 'h97_mean_delta': float(h97['delta_auc_graph_bridge_calibrated_minus_h70'].mean()),
 'h97_positive_domain_splits_null_gap': int((h97s['mean_null_gap_q95_delta_auc']>0).sum()),
 'h98_mean_delta': float(h98['delta_auc_id_entropy_minus_h70'].mean()),
 'h99_mean_concordance': float(h99['module_role_graph_concordance'].mean()),
}, indent=2))
PY
```

Paper update and compile command:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

No package installation was required.

## Quantitative Results

### H97 - Calibrated Bridge-Curvature Rewiring Rescue (`graph_topology`, refinement)
- Data scope: seed42, immune/lung/external_lung, source/target-disjoint splits, layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_graph_bridge_calibrated_minus_h70`.
- Mean primary metric: `+0.07852` (`+` rows: `24/24`; positive mean domain-splits: `6/6`).
- Null robustness: mean `null_gap_q95_delta_auc = -0.00867`; positive null-gap rows `2/24`; positive mean null-gap domain-splits `0/6`.
- Auxiliary: mean baseline AUC `0.87039`, calibrated blend mean AUC `0.94890`.
- Interpretation: directional utility remains strong, but stricter null calibration still fails the robustness gate (`0/6` positive mean null-gap domain-splits).
- Artifacts:
  - `iterations/iter_0038/h97_graph_bridge_calibrated_by_domain_split_layer.csv`
  - `iterations/iter_0038/h97_graph_bridge_calibrated_domain_summary.csv`
  - `iterations/iter_0038/h97_graph_bridge_calibrated_null_summary.csv`

### H98 - Multi-Radius ID Heterogeneity Entropy (`intrinsic_dimensionality`, new method)
- Data scope: seed42, immune/lung/external_lung, source/target-disjoint splits, layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_id_entropy_minus_h70`.
- Mean primary metric: `-0.00773` (`+` rows: `5/24`; positive mean domain-splits: `1/6`).
- Null robustness: mean `null_gap_q95_delta_auc = -0.06022`; positive null-gap rows `0/24`; positive mean null-gap domain-splits `0/6`.
- Auxiliary: mean baseline AUC `0.86547`, ID-entropy blend AUC `0.85774`.
- Interpretation: this ID-heterogeneity formulation underperforms baseline and is decisively non-robust.
- Artifacts:
  - `iterations/iter_0038/h98_id_entropy_by_domain_split_layer.csv`
  - `iterations/iter_0038/h98_id_entropy_domain_summary.csv`
  - `iterations/iter_0038/h98_id_entropy_null_summary.csv`

### H99 - Cross-Model Module Role-Graph Alignment (`cross_model_alignment`, structural reset)
- Data scope: seed42 pilot, immune/lung/external_lung, layers `{7,11}`.
- Rows tested: `6`.
- Primary metric: `module_role_graph_concordance`.
- Mean primary metric: `+0.03934` (positive rows `5/6`, range `[-0.01495, +0.07357]`).
- Null robustness: mean `null_gap_q95_concordance = -0.02497`; positive null-gap rows `0/6`; positive null-gap domains `0/3`.
- Auxiliary: mean top role-graph Jaccard `0.13039`; mean orthogonal alignment RMSE `1.22691`.
- Interpretation: the structural-reset cross-model endpoint remains null-fragile and fails the rescue gate (`0/3` domains).
- Artifacts:
  - `iterations/iter_0038/h99_cross_model_role_graph_by_domain_layer.csv`
  - `iterations/iter_0038/h99_cross_model_role_graph_domain_summary.csv`
  - `iterations/iter_0038/h99_cross_model_role_graph_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0038/iter0038_screen_summary.json`

## Iteration Decision
- `H97`: **negative** for promotion (strong directional lift but robustness remains below gate after stricter nulls).
- `H98`: **negative** (under baseline and non-robust).
- `H99`: **negative** (structural-reset cross-model formulation still fails null robustness in all domains).

## Blockers
- No data/runtime blocker.
- Branch-level blocker: cross-model alignment remains null-fragile after repeated major endpoint changes.
