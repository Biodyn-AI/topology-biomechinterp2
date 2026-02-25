# Executor Iteration Report — iter_0025

## Scope
Executed a 3-slot breadth packet aligned to the iter_0024 brainstormer brief:
- `H58` (`topology_stability`, refinement): biologically weighted directed/signed failure-slice rescue.
- `H59` (`cross_model_alignment`, major-change rescue): cross-model topology-signature transfer pilot.
- `H60` (`intrinsic_dimensionality`, new method broad-screen): endpoint ID-jump incremental value test.

## Environment
- Python environment: `subproject40-topology`
- No package installation required this iteration.

## Command Trace
```bash
# compile runner
conda run --no-capture-output -n subproject40-topology \
  python -m py_compile iterations/iter_0025/run_iter0025_screen.py

# execute H58/H59/H60 screening packet
conda run --no-capture-output -n subproject40-topology \
  python iterations/iter_0025/run_iter0025_screen.py

# derive compact metric digest for reporting
conda run --no-capture-output -n subproject40-topology python - <<'PY'
from pathlib import Path
import json
import pandas as pd

iter_dir = Path('iterations/iter_0025')
h58_row = pd.read_csv(iter_dir / 'h58_weighted_directed_signed_by_seed_layer_split.csv')
h58_dom = pd.read_csv(iter_dir / 'h58_weighted_directed_signed_domain_summary.csv')
h58_fail = pd.read_csv(iter_dir / 'h58_weighted_directed_signed_failure_slice_summary.csv')
h59_row = pd.read_csv(iter_dir / 'h59_cross_model_topology_signature_transfer_by_domain_layer.csv')
h59_sum = pd.read_csv(iter_dir / 'h59_cross_model_topology_signature_transfer_summary.csv')
h60_row = pd.read_csv(iter_dir / 'h60_id_jump_by_seed_layer_split.csv')
h60_dom = pd.read_csv(iter_dir / 'h60_id_jump_domain_summary.csv')

out = {
  'iteration': 'iter_0025',
  'h58': {
    'rows': int(h58_row.shape[0]),
    'mean_delta_weighted_vs_distance': float(h58_row['delta_auc_weighted_minus_distance'].mean()),
    'mean_delta_weighted_vs_unweighted': float(h58_row['delta_auc_weighted_minus_unweighted'].mean()),
    'positive_weighted_vs_distance_rows': int((h58_row['delta_auc_weighted_minus_distance'] > 0).sum()),
    'positive_weight_gain_rows': int((h58_row['delta_auc_weighted_minus_unweighted'] > 0).sum()),
    'domain_split_fisher_sig': int((h58_dom['combined_fisher_p_best'] < 0.05).sum()),
    'source_failure_slices': h58_fail.to_dict(orient='records'),
  },
  'h59': {
    'rows': int(h59_row.shape[0]),
    'mean_delta_transfer_vs_baseline': float(h59_row['delta_auc_transfer_minus_baseline'].mean()),
    'positive_rows': int((h59_row['delta_auc_transfer_minus_baseline'] > 0).sum()),
    'rows_p_best_lt_0_05': int((h59_row['p_best_upper'] < 0.05).sum()),
    'domain_fisher_sig': int((h59_sum['combined_fisher_p_best'] < 0.05).sum()),
    'domain_summary': h59_sum.to_dict(orient='records'),
  },
  'h60': {
    'rows': int(h60_row.shape[0]),
    'mean_delta_combined_vs_baseline': float(h60_row['delta_auc_combined_minus_baseline'].mean()),
    'positive_rows': int((h60_row['delta_auc_combined_minus_baseline'] > 0).sum()),
    'rows_p_best_lt_0_05': int((h60_row['p_best_upper'] < 0.05).sum()),
    'domain_split_positive_mean': int((h60_dom['mean_delta_auc_combined_minus_baseline'] > 0).sum()),
    'domain_split_fisher_sig': int((h60_dom['combined_fisher_p_best'] < 0.05).sum()),
  }
}

(iter_dir / 'iter0025_metric_digest.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
PY
```

Primary script:
- `iterations/iter_0025/run_iter0025_screen.py`

Primary machine summaries:
- `iterations/iter_0025/iter0025_screen_summary.json`
- `iterations/iter_0025/iter0025_metric_digest.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0025`.
- Compile command:
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H58 — Biologically Weighted Directed/Signed Rescue (`topology_stability`, refinement)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Comparator: weighted directed/signed vs unweighted directed/signed vs distance-only cycle baseline.
- Nulls: biological-weight shuffle within edge-degree bins, random-strata placebo, and label permutation (`32` permutations each).

Direct evidence:
- Mean `delta_AUROC(weighted - distance) = +0.01137`.
- Positive `weighted-distance` rows: `25/36`.
- Domain-split Fisher-significant groups: `4/6`.
- But rescue objective did not hold:
  - Mean `delta_AUROC(weighted - unweighted) = -0.00052`.
  - Weight-gain positive rows: `22/36` only.
  - Failure slices remained negative:
    - `lung/source_disjoint = -0.00193`
    - `external_lung/source_disjoint = -0.00433`

Interpretation:
- The directed/signed branch remains strong relative to distance baseline, but this weighting tweak did not improve the known source-disjoint failures and slightly regressed mean performance vs unweighted.

Artifacts:
- `iterations/iter_0025/h58_weighted_directed_signed_by_seed_layer_split.csv`
- `iterations/iter_0025/h58_weighted_directed_signed_domain_summary.csv`
- `iterations/iter_0025/h58_weighted_directed_signed_null_summary.csv`
- `iterations/iter_0025/h58_weighted_directed_signed_failure_slice_summary.csv`

---

### H59 — Cross-Model Topology-Signature Transfer Pilot (`cross_model_alignment`, new method)
Design:
- Coverage: seed42 pilot, held-out-domain transfer over `3 domains x 2 disjoint splits x 2 layers = 12` rows.
- Method: build per-gene topology signatures for scGPT and Geneformer, fit Procrustes alignment on source domains, and score target-domain edges via cross-model compatibility (`mapped Geneformer` vs `scGPT` signatures).
- Nulls: random-map alignment and signature-destroying permutation (`24` permutations each).

Direct evidence:
- Mean `delta_AUROC(transfer - baseline) = +0.02404`.
- Positive rows: `10/12`.
- Null-calibrated support failed:
  - Rows with `p_best < 0.05`: `0/12`.
  - Domain Fisher-significant aggregates: `0/3`.

Interpretation:
- Direction is positive but not null-robust; this pilot is inconclusive and does not satisfy the transfer gate.

Artifacts:
- `iterations/iter_0025/h59_cross_model_topology_signature_transfer_by_domain_layer.csv`
- `iterations/iter_0025/h59_cross_model_topology_signature_transfer_summary.csv`
- `iterations/iter_0025/h59_cross_model_topology_signature_transfer_null_summary.csv`

---

### H60 — Endpoint ID-Jump Broad Screen (`intrinsic_dimensionality`, new method)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 4 layers = 72` rows.
- Method: local intrinsic-dimension estimates (TWO-NN + local MLE), edge ID-jump/ID-mean features, combined score vs geodesic baseline.
- Nulls: endpoint swap within distance bins, estimator randomization placebo, and label permutation (`24` permutations each).

Direct evidence:
- Mean `delta_AUROC(combined_id_jump - geodesic_baseline) = -0.00435`.
- Positive rows: `31/72` (`43.1%`).
- Positive mean domain-splits: `3/6`.
- Fisher-significant domain-splits: `1/6` (immune/source-disjoint; direction negative).

Interpretation:
- The ID-jump endpoint is net negative in this broad-screen form and does not meet keep criteria.

Artifacts:
- `iterations/iter_0025/h60_id_jump_by_seed_layer_split.csv`
- `iterations/iter_0025/h60_id_jump_domain_summary.csv`
- `iterations/iter_0025/h60_id_jump_null_summary.csv`

## Decision Summary
- `H58`: **negative** for the rescue objective (no improvement in failure slices; slight regression vs unweighted).
- `H59`: **inconclusive** (directional gain without null-robust support).
- `H60`: **negative** (net negative incremental value in multiseed broad-screen).

## Blockers and Fallbacks
- No hard runtime/data blocker.
- Methodological caveat: `H59` gains were not calibrated by nulls in the pilot budget; expansion is not justified without stronger controls or a changed transfer objective.
