# Executor Iteration Report: iter_0043

## Objective
Run a breadth-first 3-slot packet from the brainstormer brief (`N565`, `N552`, `N559`) with one rescue-once carry-over and two materially changed methods:
- `H112` (`topology_stability`, `N565`): semi-Markov biologically anchored grammar rescue over second-order FSM.
- `H113` (`persistent_homology`, `N552`): depth-transition zigzag long-bar mass and birth-depth entropy.
- `H114` (`intrinsic_dimensionality`, `N559`): intrinsic-dimension hysteresis (forward vs reverse radius sweeps).

## Environment
- Python environment: `conda run -n subproject40-topology`
- Runner: `iterations/iter_0043/run_iter0043_screen.py`

## Command Trace
```bash
conda run -n subproject40-topology python -m py_compile iterations/iter_0043/run_iter0043_screen.py
conda run -n subproject40-topology python iterations/iter_0043/run_iter0043_screen.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```

## Results (Quantitative)

| Hypothesis | Family | Primary Metric | Result | Null Robustness | Decision |
|---|---|---|---:|---:|---|
| H112 | topology_stability | mean `delta_auc_semimarkov_minus_second_order` | `-0.03805` (6 domain-split rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H113 | persistent_homology | mean `delta_long_bar_mass_positive_minus_negative` | `-155.38889` (6 domain-split rows) | positive mean null-gap domain-splits: `0/6` | negative |
| H114 | intrinsic_dimensionality | mean `delta_auc_id_hysteresis_minus_h70` | `+0.00026` (24 rows) | positive mean null-gap domain-splits: `0/6` | negative |

### H112 details (`N565`: semi-Markov bio-grammar rescue)
- Rescue objective failed directly: semi-Markov underperformed second-order FSM in all 6 domain-splits (`0/6` positive deltas).
- Mean `delta_auc_semimarkov_minus_second_order = -0.03805`; mean null-gap to q95 remained negative in every split.
- Compared with deep-layer H70, the semi-Markov score was still directionally above H70 (`mean delta_auc_semimarkov_minus_h70 = +0.05956`), but this did not rescue the carry-over objective.
- Artifacts:
  - `iterations/iter_0043/h112_semimarkov_biogrammar_by_domain_split.csv`
  - `iterations/iter_0043/h112_semimarkov_biogrammar_domain_summary.csv`
  - `iterations/iter_0043/h112_semimarkov_biogrammar_null_summary.csv`

### H113 details (`N552`: depth zigzag long-bar mass)
- Positive-vs-negative long-bar mass contrast was strongly negative on average (`mean delta = -155.38889`), with only `2/6` splits directionally positive.
- Null robustness failed decisively (`0/6` positive null-gap domain-splits); even positive-direction splits stayed below q95 null thresholds.
- Birth-depth entropy gap was near zero (`mean ~ -0.00007`), providing no useful separation.
- Artifacts:
  - `iterations/iter_0043/h113_depth_zigzag_longbar_by_domain_split.csv`
  - `iterations/iter_0043/h113_depth_zigzag_longbar_domain_summary.csv`
  - `iterations/iter_0043/h113_depth_zigzag_longbar_null_summary.csv`

### H114 details (`N559`: intrinsic-dimension hysteresis)
- Broad screen produced near-zero utility (`mean delta_auc = +0.00026`), with mixed sign (`11/24` positive rows).
- Robustness gate failed (`2/24` rows had positive null-gap, but `0/6` domain-splits had positive mean null-gap; `0/24` rows with `p_best < 0.05`).
- Strongest single row (external_lung/source_disjoint/layer3) had `delta=+0.02169` but only marginal null-gap (`+0.00221`) and non-significant `p_best=0.0769`.
- Artifacts:
  - `iterations/iter_0043/h114_id_hysteresis_by_domain_split_layer.csv`
  - `iterations/iter_0043/h114_id_hysteresis_domain_summary.csv`
  - `iterations/iter_0043/h114_id_hysteresis_null_summary.csv`

## Interpretation
- `H112/N565`: rescue-once attempt failed; this major-change semi-Markov formulation is negative under the pre-registered objective.
- `H113/N552`: high-risk zigzag branch is negative in this tested formulation (direction and null-gap both fail).
- `H114/N559`: cheap broad-screen variant is effectively null with no domain-split robustness.

## Blockers / Runtime Notes
- No data blockers.
- First draft of `H113` was runtime-heavy under larger null budgets; we bounded the final run by reducing the null budget and caching per-layer class-specific kNN graphs in the final committed script.
- Non-blocking sklearn deprecation warnings (`penalty`/`l1_ratio`) appeared during logistic CV and did not affect artifact generation.

## Machine-Readable Summary
- `iterations/iter_0043/iter0043_screen_summary.json`
