# Executor Iteration Report - iter_0037

## Scope
This iteration executed a bounded 3-slot screening packet with one carry-over refinement and two materially different branches:
- `H94` (`persistent_homology`, refinement of `H93`): GO-ontology-stratified weighted filtration.
- `H95` (`graph_topology`, new method): bridge-curvature descriptor blend with degree-preserving edge-swap control.
- `H96` (`cross_model_alignment`, rescue-once major change): cross-model GO-module topology-rank concordance.

Cross-model work was limited to one rescue slot because this family has repeated negatives; the method was intentionally changed from edge-transfer endpoints to module-level topology rank concordance.

## Command Trace
All research commands were run in the required environment:

```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0037/run_iter0037_screen.py
PYTHONWARNINGS=ignore conda run --no-capture-output -n subproject40-topology python iterations/iter_0037/run_iter0037_screen.py
```

Metric extraction for report values:

```bash
conda run --no-capture-output -n subproject40-topology python - <<'PY'
import json
import pandas as pd
from pathlib import Path
b=Path('iterations/iter_0037')
h94=pd.read_csv(b/'h94_ontology_stratified_weighted_filtration_by_seed_split_layer.csv')
h94s=pd.read_csv(b/'h94_ontology_stratified_weighted_filtration_domain_summary.csv')
h95=pd.read_csv(b/'h95_graph_bridge_curvature_by_domain_split_layer.csv')
h95s=pd.read_csv(b/'h95_graph_bridge_curvature_domain_summary.csv')
h96=pd.read_csv(b/'h96_cross_model_module_topology_by_domain_layer.csv')
h96s=pd.read_csv(b/'h96_cross_model_module_topology_domain_summary.csv')
print(json.dumps({
 'h94_mean_delta': float(h94['delta_auc_ontology_weighted_minus_global_weighted'].mean()),
 'h95_mean_delta': float(h95['delta_auc_graph_bridge_curvature_minus_h70'].mean()),
 'h96_mean_spearman': float(h96['module_spearman_scgpt_geneformer'].mean())
}, indent=2))
PY
```

Paper/log maintenance command:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

No package installation was required.

## Quantitative Results

### H94 - GO-Ontology-Stratified Weighted Filtration (`persistent_homology`, refinement)
- Data scope: immune/lung/external_lung; seeds `42/43/44`; splits `source_disjoint/target_disjoint`; layers `{7,11}`.
- Rows tested: `36`.
- Primary metric: `delta_auc_ontology_weighted_minus_global_weighted`.
- Mean primary metric: `-0.00933` (`+` rows: `9/36`; positive mean domain-splits: `0/6`).
- Null robustness: positive `null_gap_q95_delta_auc` rows `0/36`; positive mean null-gap domain-splits `0/6`.
- Auxiliary: mean global weighted AUC `0.93838` vs ontology-stratified weighted AUC `0.92905`; mean between-strata weighted-gain variance `0.00261`.
- Interpretation: stratifying weighted filtration by GO-overlap tiers reduced utility relative to the simpler global weighted model and failed all null-gap checks.
- Artifacts:
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_by_seed_split_layer.csv`
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_domain_summary.csv`
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_null_summary.csv`

### H95 - Bridge-Curvature Graph Descriptor Blend (`graph_topology`, new method)
- Data scope: seed42 breadth run across immune/lung/external_lung, splits `source_disjoint/target_disjoint`, layers `{0,3,7,11}`.
- Rows tested: `24`.
- Primary metric: `delta_auc_graph_bridge_curvature_minus_h70`.
- Mean primary metric: `+0.07710` (`+` rows: `24/24`; positive mean domain-splits: `6/6`).
- Null robustness: positive `null_gap_q95_delta_auc` rows `5/24`; positive mean null-gap domain-splits `0/6`.
- Auxiliary: mean baseline AUC `0.87053`, augmented mean AUC `0.94763`.
- Interpretation: strong directional utility lift, but gains are not robust to the degree-preserving edge-swap and shuffle controls at domain-split aggregate level.
- Artifacts:
  - `iterations/iter_0037/h95_graph_bridge_curvature_by_domain_split_layer.csv`
  - `iterations/iter_0037/h95_graph_bridge_curvature_domain_summary.csv`
  - `iterations/iter_0037/h95_graph_bridge_curvature_null_summary.csv`

### H96 - Cross-Model GO-Module Topology Rank Concordance (`cross_model_alignment`, rescue-once major change)
- Data scope: seed42 pilot across immune/lung/external_lung; layers `{7,11}`; GO modules (`64` modules/domain-layer cap).
- Rows tested: `6`.
- Primary metric: `module_spearman_scgpt_geneformer`.
- Mean primary metric: `-0.00555` (positive rows `3/6` by sign, but weak and inconsistent).
- Null robustness: mean `null_gap_q95_spearman = -0.21467`; positive null-gap rows `0/6`; positive null-gap domains `0/3`.
- Auxiliary: mean top-module Jaccard `0.11146`.
- Interpretation: the rescue-once cross-model formulation failed null robustness in all domains and does not reopen the cross-model branch.
- Artifacts:
  - `iterations/iter_0037/h96_cross_model_module_topology_by_domain_layer.csv`
  - `iterations/iter_0037/h96_cross_model_module_topology_domain_summary.csv`
  - `iterations/iter_0037/h96_cross_model_module_topology_null_summary.csv`

## Machine Summary Artifact
- `iterations/iter_0037/iter0037_screen_summary.json`

## Iteration Decision
- `H94`: **negative** (carry-over refinement underperformed global weighted baseline and failed nulls).
- `H95`: **inconclusive** (large directional lift but domain-split null-gap robustness not met).
- `H96`: **negative** (major-change cross-model rescue still fails null robustness in all domains).

## Blockers
- No data/runtime blocker.
- Branch-level blocker: repeated cross-model null-gap failures persist after major objective changes; this family should remain retired unless a fundamentally different data view (e.g., matched residual tensors with shared module supervision) is introduced.
