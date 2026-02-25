# Executor Iteration Report — iter_0024

## Scope
Executed a 3-slot breadth packet from the iter_0023 roadmap:
- `H55` (`topology_stability`, refinement): high-permutation directed/signed replication with failure-slice diagnostics.
- `H56` (`topology_stability`, major rescue): densified directed path-homology v2 with a utility-transfer endpoint.
- `H57` (`manifold_distance`, new method): geodesic neighborhood anisotropy-tail broad screen.

## Environment
- Python environment: `subproject40-topology`
- No new package installation this iteration.

## Command Trace
```bash
# compile runner
conda run --no-capture-output -n subproject40-topology \
  python -m py_compile iterations/iter_0024/run_iter0024_screen.py

# execute H55/H56/H57 screening packet
conda run --no-capture-output -n subproject40-topology \
  python iterations/iter_0024/run_iter0024_screen.py

# derive compact metric digest for reporting
conda run --no-capture-output -n subproject40-topology python - <<'PY'
from pathlib import Path
import json
import pandas as pd

iter_dir = Path('iterations/iter_0024')
h55_row = pd.read_csv(iter_dir / 'h55_directed_signed_highperm_by_seed_layer_split.csv')
h55_dom = pd.read_csv(iter_dir / 'h55_directed_signed_highperm_domain_summary.csv')
h55_diag = pd.read_csv(iter_dir / 'h55_directed_signed_failure_slice_diagnostics.csv')
h56_row = pd.read_csv(iter_dir / 'h56_path_homology_v2_by_domain_layer_split.csv')
h56_transfer = pd.read_csv(iter_dir / 'h56_path_homology_v2_utility_transfer_summary.csv')
h57_row = pd.read_csv(iter_dir / 'h57_geodesic_anisotropy_by_seed_layer_split.csv')
h57_dom = pd.read_csv(iter_dir / 'h57_geodesic_anisotropy_domain_summary.csv')

failure = h55_row[(h55_row['domain']=='lung') & (h55_row['split_regime']=='source_disjoint')]
others = h55_row[~((h55_row['domain']=='lung') & (h55_row['split_regime']=='source_disjoint'))]
assoc = h55_diag[h55_diag['diagnostic_name']!='row'][['diagnostic_name','delta_corr_global','failure_slice_minus_other_mean']]

out = {
  'iteration': 'iter_0024',
  'h55': {
    'rows': int(h55_row.shape[0]),
    'mean_delta': float(h55_row['delta_auc_directed_minus_distance'].mean()),
    'positive_rows': int((h55_row['delta_auc_directed_minus_distance']>0).sum()),
    'domain_split_fisher_sig': int((h55_dom['combined_fisher_p_best']<0.05).sum()),
    'lung_source_mean_delta': float(failure['delta_auc_directed_minus_distance'].mean()),
    'others_mean_delta': float(others['delta_auc_directed_minus_distance'].mean()),
    'top_assoc_by_abs_corr': assoc.reindex(assoc['delta_corr_global'].abs().sort_values(ascending=False).index).head(3).to_dict(orient='records'),
  },
  'h56': {
    'rows': int(h56_row.shape[0]),
    'mean_delta': float(h56_row['delta_auc_path_minus_distance'].mean()),
    'positive_rows': int((h56_row['delta_auc_path_minus_distance']>0).sum()),
    'mean_transfer_f1_lift': float(h56_transfer['f1_utility_lift'].mean()),
    'transfer_sig_rows': int((h56_transfer['p_transfer_upper']<0.05).sum()),
  },
  'h57': {
    'rows': int(h57_row.shape[0]),
    'mean_delta': float(h57_row['delta_auc_anisotropy_minus_baseline'].mean()),
    'positive_rows': int((h57_row['delta_auc_anisotropy_minus_baseline']>0).sum()),
    'domain_split_positive_mean': int((h57_dom['mean_delta_auc_anisotropy_minus_baseline']>0).sum()),
    'domain_split_fisher_sig': int((h57_dom['combined_fisher_p_best']<0.05).sum()),
  }
}
(iter_dir / 'iter0024_metric_digest.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
PY

# compile paper update
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

Primary script:
- `iterations/iter_0024/run_iter0024_screen.py`

Primary machine summaries:
- `iterations/iter_0024/iter0024_screen_summary.json`
- `iterations/iter_0024/iter0024_metric_digest.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0024`.
- Compiled PDF: `paper/autoloop_research_paper.pdf` via `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`.

## Results

### H55 — Directed/Signed High-Permutation Replication + Failure-Slice Diagnostics (`topology_stability`, refinement)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Comparator: directed/signed score vs distance-only cycle baseline.
- Nulls: degree-orientation relabeling, sign-shuffle, split-label placebo, metric-matched kNN randomization (`64` permutations each).

Direct evidence:
- Mean `delta_AUROC(directed_signed - distance_only) = +0.01169`.
- Positive delta in `25/36` rows (`69.4%`).
- Domain-split Fisher-significant in `6/6` groups.
- Two failure slices remain mean-negative: `lung/source_disjoint = -0.00315`, `external_lung/source_disjoint = -0.00440`.

Failure-slice diagnostics:
- Highest global association with uplift was `margin_iqr` (`corr = +0.4513`), and failure slice had lower mean `margin_iqr` (`-0.0133` vs others).
- Failure slice also showed lower directional sign balance (`-0.0364` vs others) and higher orientation entropy (`+0.0652` vs others).

Interpretation:
- The branch remains robustly positive overall and survives stronger null families.
- Residual failures are concentrated in low-directionality regimes, supporting a targeted rescue (biological weighting/stratification) rather than branch retirement.

Artifacts:
- `iterations/iter_0024/h55_directed_signed_highperm_by_seed_layer_split.csv`
- `iterations/iter_0024/h55_directed_signed_highperm_domain_summary.csv`
- `iterations/iter_0024/h55_directed_signed_highperm_null_summary.csv`
- `iterations/iter_0024/h55_directed_signed_failure_slice_diagnostics.csv`

---

### H56 — Densified Directed Path-Homology v2 Utility-First Rescue (`topology_stability`, major rescue)
Design:
- Coverage: seed42 pilot on `3 domains x 2 disjoint splits x 2 layers (7,11) = 12` rows.
- Method change vs H53: kNN sweep (`k=8,12`), denser directed complexes, path-length-2/3 directional terms, and directed flag-complex Betti-1 contribution.
- Utility endpoint: threshold-transfer F1 lift (`source->target` and `target->source`) vs distance-only baseline.
- Nulls: directed degree-rewire, sign-shuffle, random-map transfer control (`24` permutations each).

Direct evidence:
- Discrimination moved directionally positive: mean `delta_AUROC(path_v2 - distance_only) = +0.00757`, positive in `11/12` rows.
- Utility-transfer objective failed completely: mean `F1 lift = 0.0000`, positive transfer rows `0/12`, transfer-significant rows `0/12`.

Interpretation:
- This rescue improved edge discrimination but did not produce any utility-transfer gain; therefore it failed its defining objective.
- Given `H53` inconclusive + `H56` objective failure, this path-homology rescue line should be retired unless a materially different endpoint is introduced.

Artifacts:
- `iterations/iter_0024/h56_path_homology_v2_by_domain_layer_split.csv`
- `iterations/iter_0024/h56_path_homology_v2_utility_transfer_summary.csv`
- `iterations/iter_0024/h56_path_homology_v2_null_summary.csv`

---

### H57 — Geodesic Anisotropy-Tail Broad Screen (`manifold_distance`, new method)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 4 layers = 72` rows.
- Comparator: anisotropy-tail edge score vs geodesic-distance baseline.
- Nulls: distance-matched endpoint swap, neighborhood permutation, label permutation (`24` permutations each).

Direct evidence:
- Mean `delta_AUROC(anisotropy_tail - geodesic_baseline) = -0.01779`.
- Positive delta in `25/72` rows (`34.7%`).
- Positive mean delta in only `3/6` domain-splits; strongest `external_lung/target_disjoint = +0.03418`, weakest `immune/source_disjoint = -0.08534`.

Interpretation:
- The broad-screen keep gate was not met (`<4/6` domain-splits positive).
- Despite null significance in some slices, the direct incremental comparison to baseline is net negative, so this exact anisotropy-tail formulation is not promotable.

Artifacts:
- `iterations/iter_0024/h57_geodesic_anisotropy_by_seed_layer_split.csv`
- `iterations/iter_0024/h57_geodesic_anisotropy_domain_summary.csv`
- `iterations/iter_0024/h57_geodesic_anisotropy_null_summary.csv`

## Decision Summary
- `H55`: **promising** (replicates under higher null resolution; failure now localized and diagnosable).
- `H56`: **negative** for objective (utility-transfer rescue failed; retire this formulation).
- `H57`: **negative** for broad-screen gate (net negative incremental value vs baseline).

## Blockers and Fallbacks
- No hard data/runtime blockers.
- Calibration caveat: `H56/H57` use `24` permutations per null family, so row-level p-value floors are coarse.
- Fallback used this iteration: bounded layer set (`7,11`) for the expensive high-permutation `H55` packet to preserve portfolio breadth in one loop.
