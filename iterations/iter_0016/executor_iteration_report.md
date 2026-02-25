# Executor Iteration Report — iter_0016

## Scope
Executed a 3-hypothesis breadth packet aligned to prior brainstorm guidance:
- `H31` (`manifold_distance`, refinement): diffusion incremental value after explicit covariate adjustment and stratified diffusion-feature nulls.
- `H32` (`graph_topology`, new method): geodesic convexity-deficit + detour-ratio edge screen against geodesic baseline.
- `H33` (`cross_model_alignment`, new method): tri-domain cycle-consistent non-GW alignment with random-orthogonal nulls.

## Environment
- Python environment: `subproject40-topology`
- New dependencies installed this iteration: none.

## Command Trace
```bash
conda run --no-capture-output -n subproject40-topology python iterations/iter_0016/run_iter0016_screen.py
```

Primary script:
- `iterations/iter_0016/run_iter0016_screen.py`

Primary machine summary:
- `iterations/iter_0016/iter0016_screen_summary.json`

## Paper Update + Compile Trace
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```
- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0016`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H31 — Diffusion incremental value after covariate adjustment
Family: `manifold_distance` (rescue/refinement of H28/N127).

Design:
- Domains: immune, lung, external_lung.
- Seeds: 42/43/44.
- Splits: source-disjoint + target-disjoint.
- Layers: `0,3,7,11`.
- Baseline covariates: `{source_degree, target_degree, coexpression, euclidean_distance, geodesic_distance}`.
- Added features: diffusion distances at `t={1,2,4,8}`.
- Null: stratified diffusion-feature shuffle within degree x coexpression x geodesic strata (`n=80` per row).

Key quantitative outcomes:
- Tested rows: `72`.
- Mean incremental gain: `+0.00346` AUROC.
- Mean log-loss gain: `+0.00263`.
- Domain-split Fisher-significant groups (`p<0.05`): `3/6`.
- Strongest supports:
  - immune/source-disjoint: mean delta `+0.00940`, Fisher `p=6.78e-08`.
  - immune/target-disjoint: mean delta `+0.00494`, Fisher `p=3.84e-02`.
  - external_lung/source-disjoint: mean delta `+0.00299`, Fisher `p=1.24e-02`.

Interpretation:
- Direction is consistently positive, but robust significance is concentrated in immune and not broad enough for full tri-domain promotion.

Artifacts:
- `iterations/iter_0016/h31_diffusion_incremental_by_seed_layer_split.csv`
- `iterations/iter_0016/h31_diffusion_incremental_domain_summary.csv`
- `iterations/iter_0016/h31_diffusion_incremental_null_summary.csv`

---

### H32 — Convexity-deficit and detour-ratio graph geometry screen
Family: `graph_topology` (new method; N133-like branch).

Design:
- Domains: immune, lung, external_lung.
- Seed: `seed42_main` (cheap broad screen).
- Splits: source-disjoint + target-disjoint.
- Layers: `0,3,7,11`.
- Features: edge detour ratio (`geodesic/euclidean`) and endpoint convexity-deficit (1 - geodesic-neighborhood Jaccard), combined with geodesic baseline via logistic scoring.
- Null: degree x geodesic-length x coexpression matched label permutations (`n=120` per row).

Key quantitative outcomes:
- Tested rows: `24`.
- Mean combo AUROC: `0.5682`.
- Mean delta vs geodesic baseline: `+0.01706`.
- Domain-split Fisher-significant delta groups: `4/6`.
- Strongest supports:
  - lung/source-disjoint: delta `+0.03361`, Fisher `p=4.00e-04`.
  - lung/target-disjoint: delta `+0.01808`, Fisher `p=1.40e-02`.
  - immune/source-disjoint: delta `+0.02159`, Fisher `p=3.07e-02`.
  - immune/target-disjoint: delta `+0.00899`, Fisher `p=3.49e-02`.

Interpretation:
- This new graph-geometry mechanism is the clearest positive branch in this iteration, with reproducible gains over geodesic-only scoring in immune and lung.

Artifacts:
- `iterations/iter_0016/h32_convexity_detour_by_seed_layer_split.csv`
- `iterations/iter_0016/h32_convexity_detour_domain_summary.csv`
- `iterations/iter_0016/h32_convexity_detour_null_summary.csv`

---

### H33 — Tri-domain cycle-consistent cross-model alignment (non-GW)
Family: `cross_model_alignment` (new method; N137-style objective change).

Design:
- Domains: immune, lung, external_lung.
- Shared symbols: `260`.
- Cross-model spaces: scGPT seed42 layers (`immune L0`, `lung L0`, `external_lung L3`) and Geneformer token embeddings.
- Baseline: independent domain-wise Procrustes alignment.
- Variant: tri-domain consensus-regularized cycle-consistent refinement.
- Null: random orthogonal maps (`n=160`).

Key quantitative outcomes:
- Mean edge AUROC (independent): `0.58891`.
- Mean edge AUROC (cycle-consistent): `0.58886`.
- Mean AUROC delta: `-4.55e-05` (no practical gain).
- Mean cycle-return rate improved: `0.63846 -> 0.66538` (delta `+0.02692`).
- Cycle-return significance vs random null: `p=0.00621`.
- Domains with significant cycle-consistent edge AUROC vs random null: `0/3`.

Interpretation:
- The new objective improves structural cycle consistency but does not translate into better edge-transfer utility; this is mixed/inconclusive evidence.

Artifacts:
- `iterations/iter_0016/h33_cycle_consistent_alignment_domain_summary.csv`
- `iterations/iter_0016/h33_cycle_consistent_alignment_map_quality.csv`
- `iterations/iter_0016/h33_cycle_consistent_alignment_null_summary.csv`

## Decision Summary
- `H31`: **neutral** (positive direction with partial robustness, not broad tri-domain promotion).
- `H32`: **promising** (new mechanism with consistent gain over geodesic baseline in multiple domain-split groups).
- `H33`: **inconclusive** (cycle consistency improved, downstream edge-transfer did not).

## Blockers and Fallbacks
- No hard blockers.
- Runtime notes:
  - Coexpression computation emitted expected NumPy correlation warnings for zero-variance genes; values were sanitized in-code.
  - Geneformer load emitted expected checkpoint architecture warnings (`UNEXPECTED`/`MISSING` head params) while embedding extraction completed successfully.
