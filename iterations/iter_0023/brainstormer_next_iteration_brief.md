# Brainstormer Next Iteration Brief - iter_0023

## Objective
Convert the `H52` lead into decision-grade evidence with failure-slice diagnosis, give path-homology one objective-changed rescue attempt, and add one cheap manifold broad-screen to keep discovery breadth.

## Required Execution Packet (3 slots)

### Slot A - High-probability discovery (`N241` -> proposed `H55`)
- Hypothesis: directed/signed topology remains robust under higher permutation resolution, and `lung/source_disjoint` negativity is explainable by identifiable graph-regime diagnostics.
- Scope: domains `immune/lung/external_lung`, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `0,3,7,11`, with `>=64` permutations per null family.
- Required outputs:
  - `iterations/iter_0024/h55_directed_signed_highperm_by_seed_layer_split.csv`
  - `iterations/iter_0024/h55_directed_signed_highperm_domain_summary.csv`
  - `iterations/iter_0024/h55_directed_signed_highperm_null_summary.csv`
  - `iterations/iter_0024/h55_directed_signed_failure_slice_diagnostics.csv`
- Null/control: orientation-preserving degree rewires, sign-shuffle, split-label placebo, and metric-matched kNN randomization.

### Slot B - High-risk/high-reward (`N244` -> proposed `H56`)
- Hypothesis: densified directed path-homology with a utility-first endpoint yields signal missed by the original `H53` pilot.
- Scope: seed42 pilot on all domains/splits/layers `7,11`; expand to seeds `43/44` only if pilot clears continuation gate.
- Required outputs:
  - `iterations/iter_0024/h56_path_homology_v2_by_domain_layer_split.csv`
  - `iterations/iter_0024/h56_path_homology_v2_utility_transfer_summary.csv`
  - `iterations/iter_0024/h56_path_homology_v2_null_summary.csv`
- Null/control: directed degree-preserving rewiring, sign-shuffle/inversion, and random-map transfer control.

### Slot C - Cheap broad-screen (`N247` -> proposed `H57`)
- Hypothesis: local geodesic anisotropy-tail features provide low-cost incremental value over current geodesic baseline.
- Scope: `3 domains x 3 seeds x 2 splits x layers {0,3,7,11}` using existing residual embeddings.
- Required outputs:
  - `iterations/iter_0024/h57_geodesic_anisotropy_by_seed_layer_split.csv`
  - `iterations/iter_0024/h57_geodesic_anisotropy_domain_summary.csv`
  - `iterations/iter_0024/h57_geodesic_anisotropy_null_summary.csv`
- Null/control: distance-matched endpoint swap, neighborhood permutation, and label permutation.

## Promotion / Continuation Gates
1. `H55` promote gate: positive `delta_AUROC` in `>=75%` rows, Fisher-significant in `>=4/6` domain-splits, and a quantitative explanation for the `lung/source_disjoint` failure slice.
2. `H56` continue gate: significant utility or transfer lift in `>=2` domains under directed nulls.
3. `H57` keep gate: positive incremental AUROC or calibration gain in `>=4/6` domain-splits with `>=1` domain-level Fisher-significant aggregate.

## Stop / Retire Rules
- Do not spend slots on `H54` rupture-index reruns, rewiring-survival lineage, GW-first correspondence recovery, or motif-overlap-as-endpoint reruns.
- If `H56` fails continuation gate, retire path-homology as an active branch for at least the next two iterations.
- If `H55` fails despite high permutation budget, pause directed/signed expansion and pivot to filtration redesign (`N242`) instead of repeating the same formulation.

## Minimal Recovery Plan (only if next executor gate unexpectedly fails)
1. Run a reduced `H55` (`seed42`, all domains, layers `7,11`, `24` permutations/null) to produce one valid topology artifact quickly.
2. Run a reduced `H57` (`seed42`, all domains, layer `11`) to guarantee a second independent machine result.
3. Emit mandatory report triad (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, `executor_research_validation.json`) before any expansion.
