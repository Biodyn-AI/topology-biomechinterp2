# Brainstormer Structured Feedback — iter_0005

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0005/executor_research_validation.json`).

## Hypothesis Triage

### Promising
- `H01` + `H03` (`persistent_homology`) remain the strongest cumulative branch.
- Cumulative evidence: lung (iter_0003) plus immune and external-lung (iter_0004) were strongly positive under feature-shuffle null; iter_0005 still shows support under stricter split design.
- `H05` feature-shuffle branch is still supportive in iter_0005: `8/12` layer-tests significant (`Fisher p < 0.05`), mean layer delta `+3.998`.
- Split-robust positives already exist and are concrete: lung `L0` and external-lung `L11` pass both source/target splits.

### Neutral
- `H06` (`split_robustness`) is partial, not broad: only `2/6` tested domain-layer combinations pass both splits.
- `H04` (`intrinsic_dimensionality`) remains mixed/domain-specific from iter_0004.
- `H02` (`cross_model_alignment`) remains inconclusive because null-calibrated evidence was weak and residual-level alignment is still missing.

### Negative / Cautionary
- `H05` distance-permutation branch is strongly non-supportive: `0/12` significant, mean layer delta `-850.942`, `10/12` negative deltas.
- This falsifies the current distance-permutation null choice as a biologically interpretable stress test, not necessarily the topology branch itself.
- Additional caution: some distance-permutation rows show pathological behavior (e.g., immune target split null means at/near zero with only `n=4` null replicates), indicating unstable calibration under this null family.

## Machine Artifacts Inspected
- `iterations/iter_0005/executor_iteration_report.md`
- `iterations/iter_0005/executor_hypothesis_screen.json`
- `iterations/iter_0005/executor_research_validation.json`
- `iterations/iter_0005/h1_stronger_null_split_by_seed_layer.csv`
- `iterations/iter_0005/h1_stronger_null_split_layer_summary.csv`
- `iterations/iter_0005/h1_stronger_null_split_domain_summary.csv`
- `iterations/iter_0005/iter0005_screen_summary.json`
- `reports/autoloop_master_log.md`
- `paper/autoloop_research_paper.tex`

## Recommendation
- Keep topology branch active, but shift from “more replication” to “robustness map + biologically interpretable stronger null.”
- Replace distance-permutation with degree-preserving/geodesic rewiring controls before drawing negative conclusions.
- Prioritize a full-layer split robustness map in one domain (immune) to localize where brittleness emerges by depth.
