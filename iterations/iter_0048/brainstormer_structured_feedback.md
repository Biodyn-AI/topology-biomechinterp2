# Brainstormer Structured Feedback - iter_0048

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0048/executor_research_validation.json`).
- Full 3-slot execution planning is valid.

## Iteration Evidence Snapshot
- `H127` (`module_structure`) remains the strongest active branch but still misses strict-null robustness on hard slices.
- Evidence: mean `delta_vs_h70=+0.13222`, positive deltas `9/9`, positive mean null-gap `2/9`, `lung/dual_axis_disjoint` null-gap `-0.00596`.
- `H128` (`graph_topology` surrogate) is negative in this formulation.
- Evidence: mean `delta_vs_h70=+0.00753`, positive mean null-gap `0/6`.
- `H129` (`manifold_distance` multi-scale torsion) is negative in this formulation.
- Evidence: mean `delta_vs_h70=+0.02100`, positive mean null-gap `0/6`.

## New Artifacts Reviewed (iter_0048)
- `iterations/iter_0048/executor_iteration_report.md`
- `iterations/iter_0048/executor_hypothesis_screen.json`
- `iterations/iter_0048/executor_research_validation.json`
- `iterations/iter_0048/iter0048_screen_summary.json`
- `iterations/iter_0048/h127_signed_string_go_hardening_by_seed_domain_split.csv`
- `iterations/iter_0048/h127_signed_string_go_hardening_domain_summary.csv`
- `iterations/iter_0048/h127_signed_string_go_hardening_null_summary.csv`
- `iterations/iter_0048/h128_graph_topology_surrogate_by_domain_split_layer.csv`
- `iterations/iter_0048/h128_graph_topology_surrogate_domain_summary.csv`
- `iterations/iter_0048/h128_graph_topology_surrogate_null_summary.csv`
- `iterations/iter_0048/h129_multiscale_torsion_by_domain_split_layer.csv`
- `iterations/iter_0048/h129_multiscale_torsion_domain_summary.csv`
- `iterations/iter_0048/h129_multiscale_torsion_null_summary.csv`
- `reports/autoloop_master_log.md`
- `paper/autoloop_research_paper.tex`

## Stale Direction Triage
1. Direct cross-model utility transfer chain (`H109/H119/H122/H125`) -> `retire_now`.
Reason: repeated domain-level null-gap failure after multiple objective resets.

2. Additive graph-topology surrogates (`H95/H97/H128`) -> `retire_now`.
Reason: weak directional lift and repeated strict-null collapse.

3. Coarse sequence-grammar lineage (`H104/H107/H111/H112`) -> `retire_now`.
Reason: directional effects without robust null survival; rescue attempts already consumed.

4. Standalone additive intrinsic-dimension branch (`H98/H114`) -> `retire_now`.
Reason: repeated near-zero utility with no credible rescue signal.

5. Scalar PH rescue variants (`H103/H106/H113`) -> `retire_now`.
Reason: repeated negative outcomes under changed filtrations; low rescue potential in additive form.

6. Torsion-only manifold rescue chain (`H126/H129`) -> `rescue_once_with_major_change`.
Constraint: only reopen with representation/objective shift (not more scale tuning).

7. Signed motif-community branch (`H116/H118/H123/H124/H127`) -> `keep_active_high_priority`.
Reason: strongest directional signal, clear bottleneck localized to specific split-domain slices.

## Strategic Pivot
- Keep one exploitation slot on `H127` hard slices, but change feature geometry from binary GO membership to continuous semantic-distance interactions.
- Re-open cross-model work only with topology-invariant/chart-consistency objectives, not direct transfer AUROC deltas.
- Use one low-cost manifold diagnostic to quickly map where local chart breaks drive false positives/false negatives.

## Minimal Recovery Plan (only if a future gate flips to `false`)
1. Run the cheap broad-screen slot first with reduced null budget (`>=12`) to restore valid machine outputs fast.
2. Run a narrowed `H127` hard-slice packet (`lung dual-axis`, `immune source`) with full strict nulls.
3. Defer high-cost cross-model alignment slot until gate returns to `true`.
