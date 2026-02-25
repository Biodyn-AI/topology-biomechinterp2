# Executor Iteration Report: iter_0042

## Objective
Run a breadth-first 3-slot packet with one carry-over rescue and two materially new methods:
- `H109` (`cross_model_alignment`, `N546`): multi-seed cross-model perturbation Jacobian alignment rescue.
- `H110` (`topology_stability`, `N539`): perturbation persistence vineyards vs H93 backbone.
- `H111` (`topology_stability`, `N551`): biologically anchored finite-state grammar across depth.

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0042/run_iter0042_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0042/run_iter0042_screen.py
conda run -n subproject40-topology python iterations/iter_0042/run_iter0042_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H109 | cross_model_alignment | mean `module_response_rank_spearman` | `+0.79114` (9 seed-domain rows) | positive response null-gap rows: `2/9`; positive jacobian null-gap rows: `0/9` | negative |
| H110 | topology_stability | mean `delta_auc_vineyard_features_minus_h93` | `+0.00091` (12 rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H111 | topology_stability | mean `delta_auc_biofsm_minus_h70` | `+0.11202` (6 rows) | positive mean null-gap domain-splits: `1/6` | inconclusive |

### H109 details (`N546`: multi-seed cross-model perturbation Jacobian alignment)
- Strong raw concordance persisted (`mean response Spearman=+0.79114`, `mean Jacobian subspace cosine=+0.52211`), but robustness gate failed.
- Seed-level response null-gap domain counts:
  - `seed42_main`: `1/3`
  - `seed43`: `1/3`
  - `seed44`: `0/3`
- Immune fail-fast criterion triggered: immune response null-gap was negative in all seeds (`-0.07944`, `-0.12197`, `-0.02361`).
- Artifacts:
  - `iterations/iter_0042/h109_cross_model_jacobian_alignment_by_seed_domain.csv`
  - `iterations/iter_0042/h109_cross_model_jacobian_alignment_domain_summary.csv`
  - `iterations/iter_0042/h109_cross_model_jacobian_alignment_null_summary.csv`

### H110 details (`N539`: perturbation persistence vineyards)
- Directionality was near null (`mean delta=+0.00091`); only `4/6` domain-splits had positive mean delta.
- Robustness failed decisively (`positive mean null-gap=0/6` domain-splits).
- Fail-fast criterion hit for this pilot (`0/6` positive mean null-gap).
- Artifacts:
  - `iterations/iter_0042/h110_persistence_vineyard_by_domain_split_layer.csv`
  - `iterations/iter_0042/h110_persistence_vineyard_domain_summary.csv`
  - `iterations/iter_0042/h110_persistence_vineyard_null_summary.csv`

### H111 details (`N551`: biologically anchored finite-state grammar)
- Strong directional lift over H70 in every split (`mean delta=+0.11202`, positive mean deltas `6/6` domain-splits).
- Null robustness remains weak (`positive mean null-gap=1/6` domain-splits; only `external_lung/target_disjoint` was positive at `+0.00216`).
- This branch is not yet promotion-ready, but unlike H110 it did not hit the directionality fail-fast condition.
- Artifacts:
  - `iterations/iter_0042/h111_bio_anchored_fsm_by_domain_split.csv`
  - `iterations/iter_0042/h111_bio_anchored_fsm_domain_summary.csv`
  - `iterations/iter_0042/h111_bio_anchored_fsm_null_summary.csv`

## Interpretation
- `H109/N546`: negative for promotion under pre-registered robustness logic; immune remained a stable failure mode across all three seeds.
- `H110/N539`: decisive negative evidence for the tested vineyards additive formulation.
- `H111/N551`: most promising directional signal in this iteration, but still control-limited; classify as inconclusive rather than positive.

## Blockers / Runtime Notes
- No data blockers.
- Non-blocking sklearn warnings (`penalty` deprecation / `l1_ratio` consistency) appeared during CV fits; artifacts were produced successfully.

## Machine-Readable Summary
- `iterations/iter_0042/iter0042_screen_summary.json`
