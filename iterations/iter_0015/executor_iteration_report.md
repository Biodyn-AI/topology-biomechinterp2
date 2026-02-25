# Executor Iteration Report — iter_0015

## Scope
Executed a 3-hypothesis breadth packet with one carry-over refinement and two materially changed methods:
- `H28` (`manifold_distance`, refinement): diffusion uplift under coexpression+degree matched null.
- `H29` (`cross_model_alignment`, rescue): CCA-seeded, one-to-one projected GW.
- `H30` (`topology_stability`, new method): triangle-thinness (hyperbolicity proxy) edge screen.

## Environment
- Python environment: `subproject40-topology`
- Dependency installed this iteration (required for real coexpression controls from `.h5ad`):
  - `conda run --no-capture-output -n subproject40-topology pip install anndata`

## Command Trace
```bash
conda run --no-capture-output -n subproject40-topology pip install anndata
conda run --no-capture-output -n subproject40-topology python iterations/iter_0015/run_iter0015_screen.py
```

Primary script:
- `iterations/iter_0015/run_iter0015_screen.py`

Primary machine summary:
- `iterations/iter_0015/iter0015_screen_summary.json`

## Paper Update + Compile Trace
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper/autoloop_research_paper.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex  # run in workdir=paper
```
- Updated source: `paper/autoloop_research_paper.tex` (added `ITERATION UPDATE: iter_0015`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H28 — Diffusion uplift under coexpression-matched stronger null
Family: `manifold_distance` (refinement of H25).

Design:
- Domains: immune, lung, external_lung.
- Seeds: 42/43/44.
- Splits: source-disjoint + target-disjoint.
- Layers: `0,3,7,11`.
- Diffusion times: `t={1,2,4,8}`.
- Nulls:
  - random label permutation (`n=120` per row),
  - coexpression+degree matched label permutation (`n=120` per row), where coexpression bins are derived from absolute Pearson correlation in domain-matched processed `.h5ad` expression matrices.

Key quantitative outcomes:
- Tested rows: `72`.
- Mean best-diffusion delta vs best baseline: `+0.00774` AUROC.
- Positive delta rows: `44/72` (`61.1%`).
- Significant rows (`p<0.05`):
  - random null: `12/72`.
  - matched null: `3/72`.
- Domain mean deltas:
  - immune: `+0.00407` (matched Fisher `p=0.0749`)
  - lung: `+0.01383` (matched Fisher `p=0.1878`)
  - external_lung: `+0.00530` (matched Fisher `p=0.99999`)

Interpretation:
- Direction remains positive but attenuated under the stronger coexpression-aware control.
- Promotion gate for this iteration was **not met** (`0/3` domains with matched-null Fisher `p<0.05`).

Artifacts:
- `iterations/iter_0015/h28_diffusion_coexp_by_seed_layer_split.csv`
- `iterations/iter_0015/h28_diffusion_coexp_domain_summary.csv`
- `iterations/iter_0015/h28_diffusion_coexp_null_summary.csv`

---

### H29 — CCA-seeded one-to-one GW rescue
Family: `cross_model_alignment` (materially changed rescue from H27).

Design:
- Domains: immune, lung, external_lung.
- Matched genes/domain: `280`.
- Layer pairing: immune `L0`, lung `L0`, external_lung `L3`.
- Steps:
  1. PCA + linear CCA seed map,
  2. entropic GW with annealed epsilon schedule,
  3. one-to-one projection by Hungarian assignment on coupling.
- Null: random correspondence permutations (`n=180`/domain).
- Baselines: CCA seed mapping and H27 unseeded GW reference.

Key quantitative outcomes:
- Mean seeded-GW top-1 retrieval: `0.00833` (combined Fisher `p=0.1248`; `0/3` domains significant).
- Mean seeded-GW transfer AUROC: `0.5008` (combined Fisher `p=0.4345`; `0/3` domains significant).
- Mean delta vs H27 unseeded:
  - top-1: `+0.00714` (small absolute gain, still near chance),
  - transfer AUROC: `-0.01782`.
- CCA seed alone remained much stronger for correspondence:
  - mean CCA top-1: `0.7452`.

Interpretation:
- GW rescue still fails practical correspondence recovery and does not improve transfer utility.
- The one-to-one GW branch remains non-promotable despite stronger initialization.

Artifacts:
- `iterations/iter_0015/h29_seeded_gw_domain_summary.csv`
- `iterations/iter_0015/h29_seeded_gw_null_summary.csv`
- `iterations/iter_0015/h29_seeded_gw_map_quality.csv`

---

### H30 — Triangle-thinness (hyperbolicity proxy) edge screen
Family: `topology_stability` (new method, cheap broad-screen).

Design:
- Domains: immune, lung, external_lung.
- Seed: `seed42_main` (cheap triage mode by design).
- Splits: source-disjoint + target-disjoint.
- Layers: `0,3,7,11`.
- Score: endpoint-local geodesic triangle thinness (lower thickness -> higher score).
- Null: degree+geodesic-distance matched label permutations (`n=160` per row).
- Baseline: geodesic edge distance AUROC.

Key quantitative outcomes:
- Tested rows: `24`.
- Mean thinness AUROC: `0.4657`.
- Mean geodesic AUROC (baseline): `0.5508`.
- Mean delta (thinness - geodesic): `-0.0850`.
- Above-chance thinness rows: `1/24`.
- Significant thinness rows (`p<0.05`): `1/24`.
- Domain-split Fisher significance: `1/6` groups (immune source-disjoint `p=0.0163`), all others non-significant.

Interpretation:
- Triangle-thinness did not show broad discriminatory signal and underperformed geodesic baselines.
- This low-cost screen yields decisive negative evidence for the tested formulation.

Artifacts:
- `iterations/iter_0015/h30_hyperbolicity_by_seed_layer_split.csv`
- `iterations/iter_0015/h30_hyperbolicity_domain_summary.csv`
- `iterations/iter_0015/h30_hyperbolicity_null_summary.csv`

## Decision Summary
- `H28`: **inconclusive** (positive direction, stronger-null robustness not met).
- `H29`: **negative** (seeded one-to-one GW rescue failed; CCA seed dominates).
- `H30`: **negative** (triangle-thinness generally below chance and below geodesic baseline).

## Blockers and Fallbacks
- Blocker encountered: `anndata` missing in the environment for coexpression-aware controls.
- Resolution: installed `anndata` in `subproject40-topology` and reran full packet.
- Runtime notes:
  - NumPy emitted expected correlation warnings from zero-variance genes during coexpression computation; values were sanitized to finite zero values in-code.
  - No experiment was skipped; all three hypotheses completed with machine outputs.
