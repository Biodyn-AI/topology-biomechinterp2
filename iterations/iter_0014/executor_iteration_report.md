# Executor Iteration Report — iter_0014

## Scope
Screened three hypotheses in one bounded packet with explicit null controls:
- `H25` diffusion-distance manifold metric sweep (`N105` lineage).
- `H26` multi-prior biological anchoring interaction test (`N110`-style with leakage-safe priors).
- `H27` unseeded GW cross-model alignment (`N108`).

## Environment
- Required env: `subproject40-topology`.
- Package install performed this iteration:
  - `conda run -n subproject40-topology pip install POT`

## Command Trace
```bash
conda run -n subproject40-topology pip install POT
conda run -n subproject40-topology python iterations/iter_0014/run_iter0014_screen.py
```

Primary script:
- `iterations/iter_0014/run_iter0014_screen.py`

Primary machine summary:
- `iterations/iter_0014/iter0014_screen_summary.json`

## Results

### H25 — Diffusion distance vs Euclidean/geodesic baselines
Family: `manifold_distance` (new method over prior H13 baseline).

Data/design:
- Domains: immune, lung, external_lung.
- Seeds: 42/43/44.
- Splits: source-disjoint + target-disjoint.
- Layers: `0,3,7,11`.
- Diffusion times: `t={1,2,4,8}`.
- Null: label permutations (`n=180`) per seed-layer-split row.

Key quantitative outcomes:
- Best-diffusion rows: `72`.
- Mean best diffusion AUROC: `0.5676`.
- Mean best baseline AUROC (max of Euclidean/geodesic): `0.5503`.
- Mean delta (best diffusion - best baseline): `+0.0173`.
- Positive-delta rows: `86.1%`.
- Rows with `p<0.05` (upper-tail permutation): `33.3%`.
- Domain-level mean deltas:
  - immune: `+0.0161` (Fisher `p=1.30e-05`)
  - lung: `+0.0249` (Fisher `p=1.07e-11`)
  - external_lung: `+0.0109` (Fisher `p=7.07e-05`)
- Best time concentration: `t=1` in `41/72` rows.

Interpretation:
- Positive and reproducible uplift over Euclidean/geodesic across all domains.
- Promising manifold metric branch.

Artifacts:
- `iterations/iter_0014/h25_diffusion_distance_by_seed_layer_split.csv`
- `iterations/iter_0014/h25_diffusion_distance_domain_summary.csv`
- `iterations/iter_0014/h25_diffusion_distance_null_summary.csv`

---

### H26 — Biological anchoring interaction (geometry x prior support)
Family: `module_structure`/biological anchoring (materially changed from prior confidence-tier monotonicity).

Design notes:
- Leakage control: TRRUST support logged only for bookkeeping; excluded from model predictors because labels are TRRUST-defined.
- Priors used in interaction model: DoRothEA support, GO co-membership, TF-source indicator.
- Splits: source-disjoint + target-disjoint across 3 domains.
- Null/control: degree-stratified prior permutations (`n=180`) and degree-stratified bootstrap (`n=180`).

Key quantitative outcomes:
- Edge table rows: `13,563`.
- Domain-split model rows: `6`.
- Interaction coefficient > 0 in `1/6` rows.
- Interaction coefficient significant (`p<0.05`) in `0/6` rows.
- AUROC delta (full interaction model - baseline model) > 0 in `4/6` rows.
- AUROC delta significant (`p<0.05`) in `2/6` rows.
- Mean AUROC delta: `+0.00068`.
- Combined Fisher for AUROC-delta p-values across 6 rows: `p=0.01396`.

Interpretation:
- No robust positive `geometry x prior` interaction effect.
- Mild, split/domain-conditional calibration gain exists, but not interaction-supported.
- Classified as neutral/mixed.

Artifacts:
- `iterations/iter_0014/h26_bio_anchor_edge_table.csv`
- `iterations/iter_0014/h26_bio_anchor_model_summary.csv`
- `iterations/iter_0014/h26_bio_anchor_permutation_null.csv`

---

### H27 — Unseeded Gromov-Wasserstein alignment (cross-model)
Family: `cross_model_alignment` (new method vs prior CCA alignment).

Data/design:
- Domains: immune, lung, external_lung.
- Matched genes/domain: `280`.
- Layer pairing: immune `L0`, lung `L0`, external_lung `L3`.
- Null/control: random correspondence permutations (`n=180`) per domain.
- Numerical stabilization used: scaled cost matrices + entropic GW with fallback to classic GW.

Key quantitative outcomes:
- Map quality failed correspondence recovery:
  - Mean top-1 retrieval: `0.00119`.
  - Domains with significant top-1 vs null: `0/3`.
  - Combined Fisher top-1: `p=0.990`.
- Coarse geometric structure signal was positive:
  - Mean distance Spearman: `0.8556`.
  - Domains significant for distance Spearman: `3/3`.
  - Combined Fisher distance: `p=2.33e-05`.
  - kNN Jaccard significant in `3/3` domains (combined Fisher `p=2.33e-05`).
- Edge-transfer utility remained weak:
  - Mean transfer AUROC: `0.5186`.
  - Domains significant: `1/3`.
  - Combined Fisher transfer AUROC: `p=0.0511`.

Interpretation:
- Geometry-level alignment exists, but unseeded GW fails correspondence recovery and does not deliver robust edge-transfer gains.
- Branch outcome is mixed/inconclusive for practical cross-model map recovery.

Artifacts:
- `iterations/iter_0014/h27_gw_alignment_domain_summary.csv`
- `iterations/iter_0014/h27_gw_alignment_null_summary.csv`
- `iterations/iter_0014/h27_gw_alignment_map_quality.csv`

## Decision Summary
- `H25`: **promising** (promote).
- `H26`: **neutral/mixed** (do not promote as-is).
- `H27`: **inconclusive/mixed** (coarse geometry yes, correspondence recovery no).

## Blockers and Fallbacks
- No runtime blocker after installing `POT`.
- Data anchor limitation: no local, ready-to-use STRING edge prior found in the accessible paths for this packet. Fallback used: DoRothEA + GO + TF-source priors.
- Methodological safeguard applied: TRRUST feature leakage prevented by excluding TRRUST support from H26 model predictors.
