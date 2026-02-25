# Executor Iteration Report — iter_0020

## Scope
Executed a 3-hypothesis screening packet emphasizing breadth with one carry-over refinement:
- `H43` (`module_structure`, refinement with major method change): STRING + ontology support-interaction model.
- `H44` (`topology_stability`, rescue with materially changed method): true zigzag persistence with `dionysus`.
- `H45` (`intrinsic_dimensionality`, carry-over refinement): robust OOS ID/local-linearity validation.

## Environment
- Python environment: `subproject40-topology`
- New dependency installed this iteration:
  - `conda run -n subproject40-topology python -m pip install dionysus`
- Environment side effect from that install:
  - `numpy` upgraded from `1.26.4` to `2.4.2` by resolver.

## Command Trace
```bash
# install zigzag tooling
conda run -n subproject40-topology python -m pip install dionysus

# compile + run iteration screen
conda run -n subproject40-topology python -m py_compile iterations/iter_0020/run_iter0020_screen.py
conda run -n subproject40-topology python iterations/iter_0020/run_iter0020_screen.py

# compile paper update
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

Primary script:
- `iterations/iter_0020/run_iter0020_screen.py`

Primary machine summary:
- `iterations/iter_0020/iter0020_screen_summary.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0020`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H43 — STRING + Ontology Continuous Support Interaction (`module_structure`)
Design:
- Coverage: seed42 only, 3 domains x 2 disjoint splits x layers `{0,3,7,11}` = `24` rows.
- Support score: DoRothEA + STRING API edge score + GO overlap + OmniPath + cell-ontology profile similarity.
- STRING source: live API fetch, cached this iteration (`84,121` scored pairs).
- Null: support permutation within `degree x coexpression x geodesic x ontology` strata (`80` permutations/row).

Key outcomes:
- Mean interaction coefficient: `+0.23253`.
- Domain-split Fisher-significant interaction support: `4/6`.
- All domain-splits had positive mean interaction coefficient (`6/6`), including lung source-disjoint (`+0.31807`, Fisher `p=0.00104`).
- Mean AUROC delta (full minus base): `-1.84e-05` (near zero).
- Mean top-vs-bottom support-decile uplift gap positive in `3/6` domain-splits.

Interpretation:
- Biological-support interaction effect remains robust in coefficient space and extends with true STRING priors, but direct AUROC lift remains minimal. This is promising for mechanism stratification, not yet for raw predictive gain.

Artifacts:
- `iterations/iter_0020/h43_support_interaction_ontology_by_seed_layer_split.csv`
- `iterations/iter_0020/h43_support_interaction_ontology_domain_summary.csv`
- `iterations/iter_0020/h43_support_interaction_ontology_null_summary.csv`
- `iterations/iter_0020/h43_string_network_api.tsv`

---

### H44 — True Split Zigzag Persistence (`topology_stability`)
Design:
- Coverage: seed42 only, 3 domains, paired source→union←target zigzag across layers `{0,3,7,11}` = `12` rows.
- Method: true zigzag persistence via `dionysus.zigzag_homology_persistence` on kNN edge complexes.
- Primary metric: H1 total lifetime.
- Null: target-set permutation controls (`40` permutations/row).

Key outcomes:
- Mean observed H1 total lifetime: `4219.58`.
- Mean observed-minus-null H1 total lifetime delta: `+234.78`.
- Positive observed-minus-null delta in `12/12` rows.
- Row-level upper-tail p-values reached permutation floor (`0.02439`) in `12/12` rows.
- Domain-level Fisher significance: `3/3` domains (`p=2.38e-04` each after layer aggregation).

Interpretation:
- True zigzag gives consistent positive evidence for split-stable cycle structure under this control design.
- Because permutation budget is finite (`40`), p-values are resolution-limited; next step should raise null budget and tie this topology signal to downstream edge utility.

Artifacts:
- `iterations/iter_0020/h44_true_zigzag_by_seed_layer_split.csv`
- `iterations/iter_0020/h44_true_zigzag_domain_summary.csv`
- `iterations/iter_0020/h44_true_zigzag_null_summary.csv`

---

### H45 — Robust OOS ID/Local-Linearity Validation (`intrinsic_dimensionality`)
Design:
- Source features: H38 moment table from `iter_0018`.
- Evaluations: leave-layer-out and leave-seed-out by domain/split.
- Robust metrics: winsorized and trimmed `ΔR²` (`full minus mean-only`), with permutation and block sign-bootstrap nulls.

Key outcomes:
- Winsorized `ΔR²` mean: `+10.8848`, permutation-significant rows `6/12`.
- Trimmed `ΔR²` mean: `-65.4611`, permutation-significant rows `1/12`.
- Block sign-bootstrap significance: `0/12` for winsorized, `0/12` for trimmed.
- Trimmed leave-layer-out rows had insufficient stable support (`6` rows with non-finite summary p-values).

Interpretation:
- Robust OOS evidence is internally inconsistent and fails block-bootstrap confirmation; this branch remains non-promotable in current form.

Artifacts:
- `iterations/iter_0020/h45_id_oos_robust_by_seed_split.csv`
- `iterations/iter_0020/h45_id_oos_robust_domain_summary.csv`
- `iterations/iter_0020/h45_id_oos_robust_null_summary.csv`

## Decision Summary
- `H43`: **promising** for biologically anchored interaction structure; raw AUROC gain remains negligible.
- `H44`: **promising** as a true-zigzag topology-stability signal under current controls.
- `H45`: **inconclusive** with mixed robust metrics and failed block-bootstrap support.

## Blockers and Fallbacks
- No hard runtime blockers.
- Method caveats:
  - `H44` p-values are bounded by null budget (`40` permutations), so significance resolution is coarse.
  - `H45` trimmed metric is unstable for leave-layer-out (small holdout size), producing non-finite rows.
