# Executor Iteration Report — iter_0006

## Objective
Run the brainstormer-prioritized immune full-layer robustness packet (`N21` + `N22`): replace distance-permutation with a degree-preserving geodesic rewiring null and map split robustness across all 12 immune layers.

## Command Trace (Reproducible)
1. Initial run attempt (failed due disconnected kNN at fixed `k=12`):
   - `conda run -n subproject40-topology python iterations/iter_0006/run_iter0006_screen.py`
2. Adaptive-k rerun attempt (failed for rare disconnected cases up to `k=30`):
   - `conda run -n subproject40-topology python iterations/iter_0006/run_iter0006_screen.py`
3. Final experiment run after component-bridge fallback patch:
   - `conda run -n subproject40-topology python iterations/iter_0006/run_iter0006_screen.py`
4. Quantitative extraction checks from generated machine artifacts:
   - `conda run -n subproject40-topology python -c "import pandas as pd; from pathlib import Path; base=Path('iterations/iter_0006'); layer=pd.read_csv(base/'h1_immune_rewire_split_layer_summary.csv'); passm=pd.read_csv(base/'h1_immune_rewire_split_pass_matrix.csv'); seed=pd.read_csv(base/'h1_immune_rewire_split_by_seed_layer.csv'); print('TOTAL_TESTS',len(layer)); print('FS_SIG_TOTAL',int(((layer.null_family=='feature_shuffle') & (layer.fisher_p<0.05)).sum())); print('RW_SIG_TOTAL',int(((layer.null_family=='degree_preserving_geodesic_rewire') & (layer.fisher_p<0.05)).sum())); print('FS_BOTH_SPLIT_SIG',int(passm[passm.null_family=='feature_shuffle'].both_splits_sig.sum())); print('RW_BOTH_SPLIT_SIG',int(passm[passm.null_family=='degree_preserving_geodesic_rewire'].both_splits_sig.sum())); print('KNN_K_MIN',int(seed.knn_k.min()),'KNN_K_MAX',int(seed.knn_k.max()),'KNN_K_MEAN',round(float(seed.knn_k.mean()),3)); print('BRIDGED_ROWS',int(seed.used_component_bridging.sum()),'OF',len(seed));"`
   - `conda run -n subproject40-topology python -c "import pandas as pd; df=pd.read_csv('iterations/iter_0006/h1_immune_rewire_split_pass_matrix.csv'); fs=df[df.null_family=='feature_shuffle']; rw=df[df.null_family=='degree_preserving_geodesic_rewire']; print('FS_BOTH_SIG_LAYERS',fs.loc[fs.both_splits_sig,'layer'].tolist()); print('FS_SOURCE_ONLY_SIG_LAYERS',fs.loc[(fs.source_sig)&(~fs.target_sig),'layer'].tolist()); print('RW_ANY_SIG_LAYERS',rw.loc[rw.source_sig|rw.target_sig,'layer'].tolist());"`
5. Paper compile for this iteration update:
   - `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`

## Experiment Definition
- Script: `iterations/iter_0006/run_iter0006_screen.py`
- Domain: `immune` (three seeds: `seed42_main`, `seed43`, `seed44`)
- Layers: all 12 (`0..11`)
- Split regimes:
  - `source_disjoint`: first-half gene index pool
  - `target_disjoint`: second-half gene index pool
- Per test unit:
  - sample `180` genes
  - center embeddings and PCA(`14`)
  - compute observed H1 lifetime sum (`ripser`, `maxdim=1`)
- Null families:
  - `feature_shuffle`: 20 replicates
  - `degree_preserving_geodesic_rewire`: 8 replicates
    - build symmetric kNN graph (adaptive `k` in `[12, 30]`)
    - if disconnected at `k=30`, connect components by nearest Euclidean bridge edges
    - apply degree-preserving double-edge swaps (`swap_multiplier=1.5`), recompute weighted geodesics, evaluate H1 on shortest-path distance matrix
- Aggregation:
  - empirical permutation p-values per seed-layer-split-null
  - Fisher-combined p-values across 3 seeds
  - dual-split pass matrix per layer (`both_splits_sig`)

## Quantitative Results

### H07 — Null sensitivity with degree-preserving geodesic rewiring (family: `null_sensitivity`)
Primary artifact: `iterations/iter_0006/h1_immune_rewire_split_domain_summary.csv`

- Rewiring null branch (`degree_preserving_geodesic_rewire`):
  - Significant layer-tests (`Fisher p < 0.05`): `0/24`
  - Source split: `0/12` significant, mean layer delta `-140.519`, mean z `-9.875`
  - Target split: `0/12` significant, mean layer delta `-129.702`, mean z `-5.412`
  - Best-case (least negative) deltas were still negative:
    - source: layer `11`, delta `-70.527`, `p=1.0`
    - target: layer `0`, delta `-86.886`, `p=1.0`

Directional interpretation:
- Negative for the hypothesis "immune H1 signal survives this stronger rewiring null." This null family is currently too strict in this implementation/regime to preserve prior positive signal.

### H08 — Immune full-layer split robustness map (family: `split_robustness`)
Primary artifacts:
- `iterations/iter_0006/h1_immune_rewire_split_pass_matrix.csv`
- `iterations/iter_0006/h1_immune_rewire_dual_split_summary.csv`

Feature-shuffle branch (`feature_shuffle`) across all immune layers:
- Total significant layer-tests (`Fisher p < 0.05`): `16/24`
- Source split: `12/12` significant, mean layer delta `+6.646`
- Target split: `4/12` significant, mean layer delta `+0.875`
- Dual-split pass rate (`both_splits_sig`): `4/12` layers
  - passing layers: `[7, 9, 10, 11]`
- Positive-delta dual-split rate: `9/12` layers

Directional interpretation:
- Mixed/partial. The full-layer map reveals clear depth-structured asymmetry in immune: source-disjoint is uniformly positive, while target-disjoint support is concentrated in later layers.

## Diagnostics / Runtime Notes
- kNN connectivity diagnostics from `h1_immune_rewire_split_by_seed_layer.csv`:
  - effective `k`: min `23`, max `30`, mean `29.903`
  - component-bridge fallback used in `142/144` rows (`98.61%`)
- Rewiring null generation was stable once graph connectivity fallback was added:
  - rewiring null draws: exactly `8` for every rewiring row
  - mean rewiring attempts per row: `8.0`

## Limitations
- The rewiring-geodesic null produced very large null persistence relative to observed Euclidean PH, yielding uniformly negative deltas. This could reflect a genuinely strong falsification under this metric pairing, or an over-adversarial mismatch between observed and null geometry scales.
- Because bridging fallback was frequently required, the current kNN graph construction regime for disjoint immune subsets may be near a connectivity boundary, which can influence geodesic null behavior.

## Decision Summary
- H07 (`null_sensitivity` with rewiring null): **Negative**.
- H08 (`split_robustness` full immune layer map): **Neutral** (informative partial robustness, not broad pass).

## Machine-Readable Artifacts Generated This Iteration
- `iterations/iter_0006/h1_immune_rewire_split_by_seed_layer.csv`
- `iterations/iter_0006/h1_immune_rewire_split_layer_summary.csv`
- `iterations/iter_0006/h1_immune_rewire_split_pass_matrix.csv`
- `iterations/iter_0006/h1_immune_rewire_split_domain_summary.csv`
- `iterations/iter_0006/h1_immune_rewire_dual_split_summary.csv`
- `iterations/iter_0006/iter0006_screen_summary.json`
