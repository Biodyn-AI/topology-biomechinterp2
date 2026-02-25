# Brainstormer Structured Feedback - iter_0050

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0050/executor_research_validation.json`).
- Full 3-slot planning is valid.

## Iteration Evidence Snapshot
- `H133` (`persistent_homology`) is decisively negative.
- Evidence: mean `delta_vs_h70=-0.04611`, positive mean null-gap domain-splits `0/6`, row-level positive null-gap `0/6`.
- `H134` (`intrinsic_dimensionality`) is directional-only and non-robust.
- Evidence: mean `delta_vs_h70=+0.01132`, positive mean null-gap domain-splits `0/6`, row-level positive null-gap `0/12`.
- `H135` (`module_structure`, H130 lineage refinement) remains directionally strong but null-fragile.
- Evidence: mean `delta_vs_h70=+0.13870`, positive mean null-gap domain-splits `0/4`; hard slices still negative (`lung/dual_axis_disjoint=-0.00502`, `immune/source_disjoint=-0.01473`).
- Trend check from cumulative log: H123 had strong strict-null support, but successive add-on variants (`H124/H127/H130/H135`) degraded to persistent hard-slice null failure.

## New Artifacts Reviewed (iter_0050)
- `iterations/iter_0050/executor_iteration_report.md`
- `iterations/iter_0050/executor_hypothesis_screen.json`
- `iterations/iter_0050/executor_research_validation.json`
- `iterations/iter_0050/iter0050_screen_summary.json`
- `iterations/iter_0050/h133_rank_surface_persistence_by_domain_split.csv`
- `iterations/iter_0050/h133_rank_surface_persistence_domain_summary.csv`
- `iterations/iter_0050/h133_rank_surface_persistence_null_summary.csv`
- `iterations/iter_0050/h134_id_phase_descriptor_by_domain_split_layer.csv`
- `iterations/iter_0050/h134_id_phase_descriptor_domain_summary.csv`
- `iterations/iter_0050/h134_id_phase_descriptor_null_summary.csv`
- `iterations/iter_0050/h135_hard_slice_semantic_refinement_by_seed_domain_split.csv`
- `iterations/iter_0050/h135_hard_slice_semantic_refinement_domain_summary.csv`
- `iterations/iter_0050/h135_hard_slice_semantic_refinement_null_summary.csv`
- `reports/autoloop_master_log.md`
- `paper/autoloop_research_paper.tex`

## Stale Direction Triage
1. Rank-surface persistent surrogate line (`H67`, `H133`) -> `retire_now`.
Reason: repeated decisive negatives with no null-surviving support.

2. Standalone intrinsic-dimension additive screens (`H89`, `H98`, `H114`, `H134`) -> `retire_now`.
Reason: repeated directional-near-zero behavior and persistent `0/6` null-gap support.

3. H130-style semantic hardening endpoint (`H130`, `H135`) -> `retire_now` for this exact feature/null form.
Reason: repeated hard-slice null failure after targeted reruns.

4. Cross-model map-learning transfer utilities (`H119`, `H122`, `H125`, `H131`) -> `retire_now`.
Reason: repeated failure despite objective resets; low rescue yield.

5. Manifold path additive variants (`H129`, `H132`) -> `rescue_once_with_major_change`.
Constraint: allow only interaction or topology-coupled formulations, not more additive path descriptors.

6. Module-structure branch overall (`H123` lineage) -> `rescue_once_with_major_change`.
Constraint: reset to H123 backbone and introduce a materially different hard-slice mechanism (ontology/sheaf obstruction or signed motif closure), with adversarial hard-slice nulls.

## Strategic Pivot For Next Loop
- Allocate one exploitation slot to a hard-slice rescue built on H123-level features (not H130 additive semantics).
- Allocate one high-risk slot to correspondence-free cross-model topology alignment (no learned gene mapping).
- Allocate one cheap geometric broad-screen that is structurally different from prior path-additive ID/torsion screens.
- Enforce strict keep gates on null-gap survival; directional lift alone is insufficient.

## Minimal Recovery Plan (only if a future gate flips to `false`)
1. Run the cheap broad-screen slot first with reduced null draws (`>=12`) to restore valid machine outputs quickly.
2. Run the hard-slice rescue only on `immune/source_disjoint` and `lung/dual_axis_disjoint` with full strict nulls.
3. Defer the high-cost cross-model slot until the gate is back to `true`.
