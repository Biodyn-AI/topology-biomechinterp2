# Executor Iteration Report — iter_0017

## Scope
Executed a 3-hypothesis breadth packet aligned to brainstormer priorities:
- `H34` (`graph_topology`, refinement of `H32` / `N141`): multiseed convexity+detour incremental value over geodesic+diffusion covariates.
- `H35` (`intrinsic_dimensionality`, new method / `N147`): local-linearity depth-breakpoint screen with split-shift nulls.
- `H36` (`cross_model_alignment`, rescue with major method change / `N149`): anchor-regularized spectral alignment optimized for split-held-out transfer utility.

## Environment
- Python environment: `subproject40-topology`
- New dependencies installed this iteration: none.

## Command Trace
```bash
conda run --no-capture-output -n subproject40-topology python iterations/iter_0017/run_iter0017_screen.py

# H36 control patch rerun (anchor-definition fix + label-permutation control)
conda run --no-capture-output -n subproject40-topology python - <<'PY'
import json, sys, importlib.util
from pathlib import Path
module_path = Path('iterations/iter_0017/run_iter0017_screen.py')
spec = importlib.util.spec_from_file_location('iter0017_screen', module_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
h36 = mod.run_h36_anchor_spectral_alignment()
summary_path = Path('iterations/iter_0017/iter0017_screen_summary.json')
summary = json.loads(summary_path.read_text())
summary['h36_anchor_spectral_alignment'] = h36
summary_path.write_text(json.dumps(summary, indent=2))
PY
```

Primary script:
- `iterations/iter_0017/run_iter0017_screen.py`

Primary machine summary:
- `iterations/iter_0017/iter0017_screen_summary.json`

## Paper Update + Compile Trace
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```
- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0017`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H34 — Multiseed convexity/detour incremental test (N141)
Family: `graph_topology` (carry-over refinement from `H32`).

Design:
- Domains: immune, lung, external_lung.
- Seeds: 42/43/44.
- Splits: source-disjoint + target-disjoint.
- Layers: `0,3,7,11`.
- Base model: `{source_degree, target_degree, coexpression, euclidean_distance, geodesic_distance, diffusion_t(1,2,4,8)}`.
- Incremental features: `{detour_ratio, convexity_deficit}`.
- Null: stratified feature shuffle of incremental columns within degree x coexpression x geodesic bins (`n=80` per row).

Key outcomes:
- Tested rows: `72`.
- Mean delta AUROC (incremental): `+0.00153`.
- Mean log-loss gain: `+0.00160`.
- Domain-split Fisher-significant groups: `2/6`.
- Significant groups:
  - immune/target-disjoint: mean delta `+0.00367`, Fisher `p=3.84e-05`.
  - lung/target-disjoint: mean delta `+0.00126`, Fisher `p=1.05e-04`.
- All six domain-split aggregates had positive mean delta, but effect sizes were small.

Interpretation:
- Positive directional replication is broad, but robustness gate is not met (significance concentrated in target-disjoint immune/lung only).

Artifacts:
- `iterations/iter_0017/h34_convexity_detour_multiseed_by_seed_layer_split.csv`
- `iterations/iter_0017/h34_convexity_detour_multiseed_domain_summary.csv`
- `iterations/iter_0017/h34_convexity_detour_multiseed_null_summary.csv`

---

### H35 — Local-linearity breakpoint screen (N147)
Family: `intrinsic_dimensionality` (new method).

Design:
- Domains: immune, lung, external_lung.
- Seeds: 42/43/44.
- Splits: source-disjoint + target-disjoint.
- Layers: all available (`12`), with edge AUROC from local linear reconstruction scores.
- Model: piecewise depth fit with breakpoint search; tested against layer-order permutation null (`n=300`).

Key outcomes:
- Tested rows: `18` (domain x seed x split).
- Piecewise improvement significance: source split `3/3` domains, target split `3/3` domains.
- Split-specific breakpoint shift significance: `1/3` domains.
- External-lung showed the strongest asymmetry:
  - mean breakpoint source `7.67` vs target `3.33` (shift `+4.33` layers), `p=0.0465`.
- Immune and lung shift tests were non-significant (`p=0.581`, `p=0.930`).

Interpretation:
- Strong evidence for depth-phase structure exists, but split-asymmetric breakpoint relocation is currently domain-specific rather than tri-domain robust.

Artifacts:
- `iterations/iter_0017/h35_linearity_breakpoint_by_seed_domain_split.csv`
- `iterations/iter_0017/h35_linearity_breakpoint_summary.csv`
- `iterations/iter_0017/h35_linearity_breakpoint_null_summary.csv`

---

### H36 — Anchor-regularized spectral alignment rescue (N149)
Family: `cross_model_alignment` (rescue with major objective/method change).

Design:
- Domains: immune, lung, external_lung.
- Shared symbols per domain: `240`.
- Mapping: spectral+PCA mixed spaces (`lambda` grid), weighted Procrustes with capped TRRUST-source anchors.
- Utility objective: select `lambda` by source-disjoint transfer AUROC; evaluate on held-out target-disjoint edges.
- Controls:
  - baseline unanchored lambda=0 map,
  - iter_0016 cycle-consistent baseline AUROC,
  - random-anchor null,
  - label-permutation null on target-split scores.

Key outcomes:
- Mean target AUROC (anchor model): `0.7753`.
- Mean target AUROC (baseline lambda=0): `0.5745`.
- Mean delta target AUROC: `+0.2008` (`3/3` domains positive).
- Mean target AUROC delta vs iter_0016 cycle baseline: positive in all domains (`+0.168` to `+0.208`).
- Label-permutation null: significant in all domains (`p=0.00826` each).
- Random-anchor null: non-discriminative (`p=1.0`), indicating anchor-randomization invariance in this formulation.

Interpretation:
- The utility-optimized spectral mapping is clearly stronger than prior baselines, but anchor-specific attribution is unresolved due random-anchor invariance. This remains mixed/inconclusive rather than promotable.

Artifacts:
- `iterations/iter_0017/h36_anchor_spectral_alignment_domain_summary.csv`
- `iterations/iter_0017/h36_anchor_spectral_alignment_map_quality.csv`
- `iterations/iter_0017/h36_anchor_spectral_alignment_null_summary.csv`

## Decision Summary
- `H34`: **neutral** (broad positive sign, modest effect, limited significant domain-split coverage).
- `H35`: **neutral** (novel depth-breakpoint structure supported, but split-shift robustness seen in only one domain).
- `H36`: **inconclusive** (large utility gain vs baselines and significant vs label null, but anchor-specific null unresolved).

## Blockers and Fallbacks
- No data-access/runtime blockers.
- Methodological blocker in `H36`: random-anchor null is invariant in this current mapping formulation.
- Fallback already executed this iteration: added label-permutation null to recover a non-degenerate control while keeping random-anchor diagnostics reported explicitly.
