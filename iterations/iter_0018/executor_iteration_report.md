# Executor Iteration Report — iter_0018

## Scope
Executed a 3-hypothesis breadth packet focused on geometric/topological screening with one carry-over refinement and two materially new tests:
- `H37` (`graph_topology`, refinement of `H34` / roadmap `N165`): biological consensus-tier concentration test for convexity/detour geometry uplift.
- `H38` (`intrinsic_dimensionality`, new method / roadmap `N160`): variance/skew moment mechanism test for layerwise local-linearity AUROC.
- `H39` (`persistent_homology`, new family branch): H1 feature-shuffle excess and coupling to geometry uplift.

## Environment
- Python environment: `subproject40-topology`
- New dependencies installed this iteration: none.

## Command Trace
```bash
# syntax check + full screen run
conda run -n subproject40-topology python -m py_compile iterations/iter_0018/run_iter0018_screen.py
conda run -n subproject40-topology python iterations/iter_0018/run_iter0018_screen.py
```

Primary script:
- `iterations/iter_0018/run_iter0018_screen.py`

Primary machine summary:
- `iterations/iter_0018/iter0018_screen_summary.json`

## Paper Update + Compile Trace
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0018`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H37 — Consensus-tier concentration of geometry uplift (`graph_topology`, refinement)
Design:
- Domain/split/layer coverage: 3 domains x 2 disjoint split regimes x layers `{0,3,7,11}` (seed42 packet; 24 rows).
- Geometry models: baseline `{degree, coexpression, euclidean, geodesic, diffusion_t(1,2,4,8)}` vs extended `+{detour_ratio, convexity_deficit}`.
- Biological tiers per edge: TRRUST support + DoRothEA support + GO co-annotation (tier range 0..3).
- Primary test: uplift gap `(delta AUROC tier>=2) - (delta AUROC tier<=1)` with stratified tier-shuffle null (`n=120`/row).

Key outcomes:
- Rows tested: `24` (`12/24` with finite tier-gap due class-balance constraints).
- Mean overall geometry uplift: `+0.00145` AUROC.
- Mean tier gap (high minus low): `-0.00801`.
- Row-level significance: `0/24` with `p_tier_gap_upper < 0.05`.
- Domain-split Fisher significance: `0/6`.

Interpretation:
- Negative evidence for the concentration claim in this formulation: higher-consensus tiers did not show stronger convexity/detour incremental value.

Artifacts:
- `iterations/iter_0018/h37_consensus_tier_geometry_by_seed_layer_split.csv`
- `iterations/iter_0018/h37_consensus_tier_geometry_domain_summary.csv`
- `iterations/iter_0018/h37_consensus_tier_geometry_null_summary.csv`

---

### H38 — ID variance/skew mechanism screen (`intrinsic_dimensionality`, new method)
Design:
- Coverage: 3 domains x 3 seeds x 2 splits x 12 layers.
- Per-layer metrics: TWO-NN local ID moments and local participation-ratio moments (mean/var/skew).
- Target: edge AUROC from local reconstruction linearity score.
- Comparison: mean-only model vs mean+variance+skew model with permutation null over layerwise AUROC (`n=300` per seed-split).

Key outcomes:
- Seed-split fits: `18`.
- Mean `ΔR²` (full minus mean-only): `+0.35673` (positive in `18/18` fits).
- Row-level significance: `1/18` fits with `p<0.05`.
- Domain-split Fisher significance: `0/6`.
- Strongest single fit: external-lung/source-disjoint seed42 with `ΔR²=+0.8234`, `p=0.00997`.

Interpretation:
- Directional evidence suggests variance/skew carries explanatory signal, but null-survival is too weak for promotion; currently neutral.

Artifacts:
- `iterations/iter_0018/h38_id_distribution_moments_by_seed_layer_split.csv`
- `iterations/iter_0018/h38_id_distribution_moments_fit_by_seed_split.csv`
- `iterations/iter_0018/h38_id_distribution_moments_domain_summary.csv`
- `iterations/iter_0018/h38_id_distribution_moments_null_summary.csv`

---

### H39 — Persistent-homology feature-shuffle excess (`persistent_homology`, new family)
Design:
- Coverage: seed42, 3 domains x 2 splits x layers `{0,3,7,11}` (24 rows).
- Topology metric: H1 persistence total lifetime/entropy from `ripser` on PCA manifold.
- Null: feature-shuffle point-cloud control (`n=20` per row).
- Coupling check: Spearman relation of `H1 z-score` with geometry uplift (`H37` overall delta AUROC).

Key outcomes:
- Rows tested: `24`.
- Mean H1 z-score vs null: `+0.34579`.
- Domain-split direction: `5/6` with positive mean H1 z-score.
- Null-survival: `0/24` rows with `p_h1_total_upper_vs_shuffle < 0.05`; domain-split Fisher `0/6`.
- Global coupling: Spearman(`H1 z`, geometry delta) `+0.3157` (directional only).

Interpretation:
- Mild positive directional signal without statistical robustness under current null budget; classify as inconclusive.

Artifacts:
- `iterations/iter_0018/h39_ph_feature_shuffle_by_seed_layer_split.csv`
- `iterations/iter_0018/h39_ph_feature_shuffle_domain_summary.csv`
- `iterations/iter_0018/h39_ph_feature_shuffle_null_summary.csv`

## Decision Summary
- `H37`: **negative** (concentration hypothesis failed; no significant support).
- `H38`: **neutral** (large directional `ΔR²`, insufficient null-robust replication).
- `H39`: **inconclusive** (positive direction but no significant null-survival).

## Blockers and Fallbacks
- No hard data/runtime blocker.
- Methodological limitation in `H37`: low-support bucket prevalence caused finite-gap coverage in only `12/24` rows even after bucket rescue (`tier<=1` vs `tier>=2`).
- Fallback executed in-loop: adjusted low-tier definition to recover finite comparisons and reran full packet.
