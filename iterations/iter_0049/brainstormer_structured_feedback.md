# Brainstormer Structured Feedback - iter_0049

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0049/executor_research_validation.json`).
- Full 3-slot execution planning is valid.

## Iteration Evidence Snapshot
- `H130` (`module_structure`) is still the only active signal branch but remains null-fragile.
- Evidence: mean `delta_vs_h70=+0.13096` (`27/27` directional positives), but positive mean null-gap domain-splits `0/9`; hardest failures remain `lung/dual_axis_disjoint=-0.00541`, `immune/source_disjoint=-0.01486`.
- Null diagnostics indicate the strongest competing nulls are semantic/STRING-aware controls (global mean null around observed lift: GO-semantic rewiring mean null `0.130182`, STRING-bin permutation mean null `0.127548`).
- `H131` (`cross_model_alignment`) is decisively negative.
- Evidence: mean `alignment_delta_vs_random=-0.00293`, positive mean null-gap domains `0/3`, immune mean null-gap `-0.14339`.
- `H132` (`manifold_distance`) is negative in this endpoint form.
- Evidence: mean `delta_vs_h70=+0.01637`, positive mean null-gap domain-splits `0/6`.

## New Artifacts Reviewed (iter_0049)
- `iterations/iter_0049/executor_iteration_report.md`
- `iterations/iter_0049/executor_hypothesis_screen.json`
- `iterations/iter_0049/executor_research_validation.json`
- `iterations/iter_0049/iter0049_screen_summary.json`
- `iterations/iter_0049/h130_semantic_go_string_hardening_by_seed_domain_split.csv`
- `iterations/iter_0049/h130_semantic_go_string_hardening_domain_summary.csv`
- `iterations/iter_0049/h130_semantic_go_string_hardening_null_summary.csv`
- `iterations/iter_0049/h131_chart_sheaf_alignment_by_domain_split_layer.csv`
- `iterations/iter_0049/h131_chart_sheaf_alignment_domain_split_summary.csv`
- `iterations/iter_0049/h131_chart_sheaf_alignment_domain_summary.csv`
- `iterations/iter_0049/h131_chart_sheaf_alignment_null_summary.csv`
- `iterations/iter_0049/h132_chart_fracture_by_domain_split_layer.csv`
- `iterations/iter_0049/h132_chart_fracture_domain_summary.csv`
- `iterations/iter_0049/h132_chart_fracture_null_summary.csv`
- `reports/autoloop_master_log.md`
- `paper/autoloop_research_paper.tex`

## Stale Direction Triage
1. Direct cross-model mapping/transfer utility endpoints (`H119/H122/H125/H131`) -> `retire_now`.
Reason: repeated failures after multiple objective resets; current loop again gives domain-level null-gap `0/3`.

2. Cross-model perturbation-concordance lineage (`H108/H109`) -> `rescue_once_with_major_change`.
Reason: one prior partial result existed, but multi-seed robustness collapsed; only correspondence-free topology/response invariants should be allowed next.

3. Local path-geometry additive rescues (`H120/H126/H129/H132`) -> `retire_now` for current endpoint family.
Reason: repeated directional-but-non-robust pattern and now two consecutive strict-null collapses (`H129`, `H132`: `0/6`).

4. Additive graph-topology surrogate endpoints (`H95/H97/H128`) -> `retire_now`.
Reason: repeated strict-null failures with low rescue yield.

5. Signed motif-community + biological hardening lineage (`H116/H118/H123/H124/H127/H130`) -> `keep_active_high_priority`.
Reason: strongest and most consistent directional signal; bottleneck is localized to hard slices, which is tractable.

## Strategic Pivot
- Keep one exploitation slot on the active module-structure branch, but change null design to slice-conditional randomization that specifically attacks `lung/dual_axis_disjoint` and `immune/source_disjoint` leakage.
- Re-open cross-model only with correspondence-free invariant objectives (distributional topology agreement), not map-learning transfer AUROC.
- Use one cheap manifold screen focused on intrinsic-dimension phase behavior to quickly test whether geometry adds non-redundant signal before investing in expensive curvature pipelines.

## Minimal Recovery Plan (if a future gate flips to `false`)
1. Run the cheap manifold broad-screen first with reduced null budget (`>=12`) to restore valid artifacts rapidly.
2. Run a narrowed hard-slice module packet (`lung/dual_axis_disjoint`, `immune/source_disjoint`) with full strict nulls.
3. Defer high-cost cross-model slot until the gate returns to `true`.
