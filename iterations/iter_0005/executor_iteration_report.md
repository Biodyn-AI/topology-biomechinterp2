# Executor Iteration Report — iter_0005

## Objective
Run the brainstormer-prioritized robustness packet (`N11` + `N12`): stress-test persistent-homology signal with stronger nulls and disjoint split regimes across lung, immune, and external-lung scGPT residual embeddings.

## Command Trace (Reproducible)
1. Initial high-cost attempt (aborted for bounded-runtime policy):
   - `conda run -n subproject40-topology python iterations/iter_0005/run_iter0005_screen.py`
   - Reason for abort: distance-permutation branch with high null count was too slow for rapid screening.
2. Final experiment run used for reported results:
   - `conda run -n subproject40-topology python iterations/iter_0005/run_iter0005_screen.py`
3. Quantitative extraction from machine artifacts:
   - `conda run -n subproject40-topology python -c "import pandas as pd; from pathlib import Path; base=Path('iterations/iter_0005'); layer=pd.read_csv(base/'h1_stronger_null_split_layer_summary.csv'); fs=layer[layer.null_family=='feature_shuffle']; dp=layer[layer.null_family=='distance_permutation']; print('feature_shuffle_sig_layers',int((fs.fisher_p<0.05).sum()),'of',len(fs)); print('distance_perm_sig_layers',int((dp.fisher_p<0.05).sum()),'of',len(dp)); print('feature_shuffle_mean_delta',round(float(fs.mean_h1_sum_delta.mean()),3)); print('distance_perm_mean_delta',round(float(dp.mean_h1_sum_delta.mean()),3))"`
4. Paper compile for this iteration update:
   - `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`

## Experiment Definition
- Script: `iterations/iter_0005/run_iter0005_screen.py`
- Data: scGPT layer embeddings for lung, immune, external-lung (3 seeds/domain).
- Layer selection (from prior outputs):
  - lung: top `L0`, weak `L11`
  - immune: top `L7`, weak `L11`
  - external-lung: top `L0`, weak `L11`
- Split regimes:
  - `source_disjoint`: first-half gene index pool
  - `target_disjoint`: second-half gene index pool
- Per test unit:
  - sample `180` genes, center embeddings, PCA(`14`), compute H1 lifetime sum (`ripser`, maxdim=1)
- Null families:
  - `feature_shuffle`: 20 replicates
  - `distance_permutation`: 4 replicates (pairwise distances permuted before PH)
- Aggregation: Fisher combine p-values across 3 seeds per domain/split/layer/null.

## Hypothesis Tests

### H05 (null_sensitivity)
Hypothesis: positive H1 topology effect persists under stronger null stress-testing.

Primary metrics (from `iterations/iter_0005/h1_stronger_null_split_layer_summary.csv`):
- Feature-shuffle null:
  - Significant layer-tests (`Fisher p < 0.05`): `8/12`
  - Mean layer delta: `+3.998`
  - Split-level means:
    - source-disjoint: `4/6` significant, mean delta `+5.219`
    - target-disjoint: `4/6` significant, mean delta `+2.777`
- Distance-permutation null:
  - Significant layer-tests (`Fisher p < 0.05`): `0/12`
  - Mean layer delta: `-850.942`
  - Negative-delta fraction across layer-tests: `10/12` (`0.833`)

Directional interpretation:
- Mixed. Topology remains clearly above feature-shuffle null in most tests, but fails under distance-permutation null, which appears to induce unrealistically high null persistence.

### H06 (split_robustness)
Hypothesis: H1 topology effect is robust across disjoint gene split regimes.

Primary metrics (feature-shuffle branch):
- Both-splits significant (`source_disjoint` and `target_disjoint`) for `2/6` domain-layer combinations:
  - lung top layer `L0`: source `p=0.0386`, target `p=0.0056`
  - external-lung weak layer `L11`: source `p=0.0455`, target `p=0.0311`
- Partial robustness elsewhere:
  - external-lung top `L0`: source significant (`p=0.0097`), target not significant (`p=0.1158`)
  - immune top `L7`: source significant (`p=0.0056`), target not significant (`p=0.2367`)

Directional interpretation:
- Mixed/partial. Disjoint robustness is present but not yet broad across all domains and tested layers.

## Limitations / Blockers
- `distance_permutation` likely violates metric/manifold structure assumptions enough to overinflate null H1 (very large negative deltas), so this null family is informative as a stress test but may be too adversarial for biological interpretation.
- One high-null run configuration was interrupted for runtime control; final reported results come from the bounded rerun with explicit final config in `iter0005_screen_summary.json`.

## Decision Summary
- H05: **Neutral** (feature-shuffle branch positive, stronger distance-permutation branch negative).
- H06: **Neutral** (partial split robustness; not yet broad enough for promotion).

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0005/h1_stronger_null_split_by_seed_layer.csv`
- `iterations/iter_0005/h1_stronger_null_split_layer_summary.csv`
- `iterations/iter_0005/h1_stronger_null_split_domain_summary.csv`
- `iterations/iter_0005/iter0005_screen_summary.json`
