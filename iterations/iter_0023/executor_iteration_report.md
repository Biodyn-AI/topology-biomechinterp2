# Executor Iteration Report — iter_0023

## Scope
Executed the planned 3-slot screening packet from the previous brainstormer brief:
- `H52` (`topology_stability`, refinement): multiseed replication of directed/signed topology under stricter null controls.
- `H53` (`topology_stability`, new method): directed path-homology surrogate pilot (high-risk/high-reward).
- `H54` (`intrinsic_dimensionality`, new method): local linearity rupture index broad-screen.

## Environment
- Python environment: `subproject40-topology`
- No new package installation this iteration.

## Command Trace
```bash
# compile runner
conda run --no-capture-output -n subproject40-topology \
  python -m py_compile iterations/iter_0023/run_iter0023_screen.py

# execute screening packet
conda run --no-capture-output -n subproject40-topology \
  python iterations/iter_0023/run_iter0023_screen.py

# derive compact metric digest for reporting
conda run --no-capture-output -n subproject40-topology python - <<'PY'
from pathlib import Path
import json
import pandas as pd

iter_dir = Path('iterations/iter_0023')
out = {"iteration": "iter_0023", "hypotheses": {}}
for hid, prefix, delta_col in [
    ('h52', 'h52_directed_signed_multiseed', 'delta_auc_directed_minus_distance'),
    ('h53', 'h53_directed_path_homology', 'delta_auc_path_minus_distance'),
    ('h54', 'h54_linearity_rupture', 'delta_auc_rupture_minus_baseline'),
]:
    row_path = iter_dir / f"{prefix}_by_seed_layer_split.csv"
    if not row_path.exists():
        row_path = iter_dir / f"{prefix}_by_domain_layer_split.csv"
    dom_path = iter_dir / f"{prefix}_domain_summary.csv"
    row = pd.read_csv(row_path)
    dom = pd.read_csv(dom_path)
    out['hypotheses'][hid] = {
        'rows': int(row.shape[0]),
        'mean_delta': float(row[delta_col].mean()),
        'positive_rows': int((row[delta_col] > 0).sum()),
        'domain_split_fisher_sig': int((dom['combined_fisher_p_best'] < 0.05).sum()),
    }
(iter_dir / 'iter0023_metric_digest.json').write_text(json.dumps(out, indent=2) + '\\n')
print(json.dumps(out, indent=2))
PY

# compile paper update
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

Primary script:
- `iterations/iter_0023/run_iter0023_screen.py`

Primary machine summaries:
- `iterations/iter_0023/iter0023_screen_summary.json`
- `iterations/iter_0023/iter0023_metric_digest.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with section marker `ITERATION UPDATE: iter_0023`.
- Compiled PDF: `paper/autoloop_research_paper.pdf` via `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`.

## Results

### H52 — Directed/Signed Topology Multiseed Replication (`topology_stability`, refinement)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 4 layers = 72` rows.
- Comparator: directed/signed score vs distance-only cycle baseline.
- Nulls: degree-orientation relabeling, sign-shuffle, split-label placebo (`16` permutations each).

Key outcomes:
- Mean `delta_AUROC(directed_signed - distance_only) = +0.01461`.
- Positive delta in `58/72` rows (`80.6%`).
- Domain-split Fisher-significant in `6/6` groups.
- One domain-split remained negative (`lung/source_disjoint`, mean delta `-0.00359`), while `5/6` domain-splits were mean-positive.

Interpretation:
- H50 replicated broadly under stricter null families and remains the strongest active branch.
- The lung/source-disjoint failure slice should be explicitly diagnosed before strong global claims.

Artifacts:
- `iterations/iter_0023/h52_directed_signed_multiseed_by_seed_layer_split.csv`
- `iterations/iter_0023/h52_directed_signed_multiseed_domain_summary.csv`
- `iterations/iter_0023/h52_directed_signed_multiseed_null_summary.csv`

---

### H53 — Directed Path-Homology Surrogate Pilot (`topology_stability`, new method)
Design:
- Coverage: seed42 pilot on `3 domains x 2 disjoint splits x 2 layers = 12` rows.
- Method: directed flag-complex path-homology surrogate (Betti-1 over distance+margin filtrations).
- Comparator: distance-only cycle baseline.
- Nulls: degree-orientation relabeling, sign-shuffle, random-map control (`6` permutations each).

Key outcomes:
- Mean `delta_AUROC(path_homology - distance_only) = +0.00276`.
- Positive delta in `8/12` rows.
- Domain-split Fisher-significant in `0/6` groups.

Interpretation:
- Directional but weak signal; the pilot did not clear continuation thresholds.
- This exact formulation should not be expanded without a materially changed endpoint/objective.

Artifacts:
- `iterations/iter_0023/h53_directed_path_homology_by_domain_layer_split.csv`
- `iterations/iter_0023/h53_directed_path_homology_domain_summary.csv`
- `iterations/iter_0023/h53_directed_path_homology_null_summary.csv`

---

### H54 — Local Linearity Rupture Index (`intrinsic_dimensionality`, new method)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 4 layers = 72` rows.
- Method: per-node depth trajectory of local reconstruction error, converted to rupture index and aggregated to edges.
- Comparator: local-linearity baseline score.
- Nulls: layer-order shuffle, endpoint swap within degree bins, label permutation (`16` permutations each).

Key outcomes:
- Mean `delta_AUROC(rupture - local_linearity_baseline) = -0.04527`.
- Positive delta in `20/72` rows (`27.8%`).
- Positive mean delta in `1/6` domain-splits.
- Domain-split Fisher-significant in `1/6` groups.

Interpretation:
- This rupture-index mechanism is decisively non-competitive against the baseline and is a negative branch.

Artifacts:
- `iterations/iter_0023/h54_linearity_rupture_by_seed_layer_split.csv`
- `iterations/iter_0023/h54_linearity_rupture_domain_summary.csv`
- `iterations/iter_0023/h54_linearity_rupture_null_summary.csv`

## Decision Summary
- `H52`: **promising** (replicates strongly across seeds/splits with one clear failure slice to debug).
- `H53`: **inconclusive** (directional pilot signal without null-robust aggregate support).
- `H54`: **negative** (large negative mean delta vs baseline; retire this formulation).

## Blockers and Fallbacks
- No hard runtime/data blockers.
- Calibration caveat: with `16` permutations in `H52/H54` and `6` in `H53`, row-level p-values have coarse floors; this limits row-level `p<0.05` resolution. Domain-level Fisher aggregation still discriminated robustly for `H52`.
