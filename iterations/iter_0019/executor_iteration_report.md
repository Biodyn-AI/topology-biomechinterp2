# Executor Iteration Report — iter_0019

## Scope
Executed a 3-hypothesis screening packet emphasizing breadth and novelty:
- `H40` (`module_structure`, rescue/new method; roadmap `N179`): continuous biological-support interaction test for geometric utility.
- `H41` (`topology_stability`, new method fallback; roadmap `N171`): split-zigzag persistence proxy with split-swap and layer-order controls.
- `H42` (`intrinsic_dimensionality`, rescue/new method; roadmap `N174`): out-of-sample ID-moment validation (leave-layer-out and leave-seed-out).

## Environment
- Python environment: `subproject40-topology`
- New dependencies installed this iteration: none.

## Command Trace
```bash
# compile + run iteration screen
conda run -n subproject40-topology python -m py_compile iterations/iter_0019/run_iter0019_screen.py
conda run -n subproject40-topology python iterations/iter_0019/run_iter0019_screen.py

# compile paper update
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

Primary script:
- `iterations/iter_0019/run_iter0019_screen.py`

Primary machine summary:
- `iterations/iter_0019/iter0019_screen_summary.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0019`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H40 — Continuous Biological-Support Interaction (`module_structure`, rescue)
Design:
- Coverage: 3 domains x 3 seeds x 2 disjoint splits x layers `{0,3,7,11}` = `72` rows.
- Geometry terms: Euclidean/geodesic/diffusion, detour ratio, convexity deficit.
- Support score: DoRothEA confidence + GO overlap + OmniPath interaction membership.
- Null: support-score permutation within `degree x coexpression x geodesic` strata (`60` permutations/row).
- Leakage guard: TRRUST presence recorded only for bookkeeping and excluded from support model terms.

Key outcomes:
- Mean interaction coefficient: `+0.13169`.
- Domain-split Fisher-significant interaction support: `4/6` (all immune/external-lung groups).
- Mean top-vs-bottom support-decile uplift gap positive in `4/6` domain-splits.
- Lung remained non-supportive (source mean interaction `-0.02299`, target `-0.00125`).

Interpretation:
- Positive rescue signal with domain concentration (immune/external-lung), but not yet universal across domains.

Artifacts:
- `iterations/iter_0019/h40_support_interaction_by_seed_layer_split.csv`
- `iterations/iter_0019/h40_support_interaction_domain_summary.csv`
- `iterations/iter_0019/h40_support_interaction_null_summary.csv`

---

### H41 — Split-Zigzag Persistence Proxy (`topology_stability`, fallback)
Design:
- Coverage: seed42, 3 domains x 2 splits x layers `{0,3,7,11}` = `24` rows.
- Method: split-local PH neighborhood summaries and edge-local zigzag proxy scores.
- Controls: split-swap local PH control + layer-order permutation control.

Key outcomes:
- Mean delta AUROC (observed minus geodesic baseline): `+0.01153`.
- Positive mean delta in `5/6` domain-splits.
- Layer-permutation Fisher significance: `0/6`.

Interpretation:
- Directional but not robust under controls; remains inconclusive.

Artifacts:
- `iterations/iter_0019/h41_zigzag_persistence_by_seed_layer_split.csv`
- `iterations/iter_0019/h41_zigzag_persistence_domain_summary.csv`
- `iterations/iter_0019/h41_zigzag_persistence_null_summary.csv`

---

### H42 — OOS ID-Moment Validation (`intrinsic_dimensionality`, rescue)
Design:
- Reused H38 moment features from `iterations/iter_0018/h38_id_distribution_moments_by_seed_layer_split.csv`.
- Tested leave-layer-out and leave-seed-out OOS `ΔR²` (`full minus mean-only`) across all domain/split groups.
- Null: permutation over target trajectories (`320` permutations per domain/split/evaluation).

Key outcomes:
- Summary rows: `12`.
- Overall mean observed `ΔR²`: `-10.70017` (instability dominated by extreme negative holdouts).
- Significant rows (`p<0.05`): `4/12`.
- Mixed sign behavior with weak broad generalization support.

Interpretation:
- Negative evidence for robust OOS mechanism transfer of H38-style ID moments in current form.

Artifacts:
- `iterations/iter_0019/h42_id_oos_by_seed_split.csv`
- `iterations/iter_0019/h42_id_oos_domain_summary.csv`
- `iterations/iter_0019/h42_id_oos_null_summary.csv`

## Decision Summary
- `H40`: **promising** (positive interaction signal with multi-split null survival in `4/6` domain-splits).
- `H41`: **inconclusive** (directional gains but no permutation-robust domain-level significance).
- `H42`: **negative** (OOS instability and mostly non-supportive transfer metrics).

## Blockers and Fallbacks
- No hard runtime blockers.
- Data/tooling blockers:
  - Local STRING score table unavailable; fallback used OmniPath interaction membership in `H40`.
  - True zigzag persistence package unavailable; fallback executed split-local PH proxy in `H41`.
