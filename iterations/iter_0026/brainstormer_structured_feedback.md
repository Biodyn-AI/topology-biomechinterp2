# Brainstormer Structured Feedback - iter_0026

## Gate Status
- `passed_min_research_gate`: `true` (`iterations/iter_0026/executor_research_validation.json`).
- No recovery-only mode required for this pass.

## Iteration Readout (What Matters)
- `H61` is net negative and split-fragile: mean `delta_AUROC=-0.00719`; only `lung/target_disjoint` is clearly positive.
- `H62` has directional gain but fails robustness: mean transfer delta `+0.04757`, mean `null_gap_q95=-0.12923`; immune carries signal, lung/external-lung are null-dominated.
- `H63` is decisively negative: mean `delta_AUROC=-0.02061`; all transition aggregates are negative.

## Stale Direction Triage
| Direction | Status | Evidence anchor | Next action |
|---|---|---|---|
| Undirected curvature-assortativity direct score (`H61` + earlier raw curvature variants) | `retire_now` | Repeated net-negative/fragile behavior; no robust cross-domain lift. | Stop direct curvature scoring; only re-open as interaction or dynamic feature. |
| Standalone ID-as-endpoint discrimination (`H60`, `H63`) | `retire_now` | Endpoint and transition ID forms both net negative under multiseed controls. | Use ID only as interaction/modulator, not primary score. |
| Directed/signed weighting tweaks (`H58` form) | `retire_now` | Failed source-disjoint rescue objective. | Do not spend more on reweighting variants. |
| Path-homology utility-transfer endpoint (`H53/H56` form) | `retire_now` | Repeated transfer-gate failure despite directional discrimination. | Re-open only with materially different endpoint. |
| Rewiring-survival redesign lineage (`H05-H12`) | `retire_now` | Many calibrated reruns stayed non-supportive. | Keep closed for this loop. |
| Cross-model alignment (raw/anchor-Procrustes forms: `H59`, `H62`) | `rescue_once_with_major_change` | Directional deltas exist but null-gap fails outside immune. | One final redesign with cycle-consistency/null-gap-first objective, otherwise retire. |
| Bifiltration utility-coupling endpoint (`H49` claim) | `rescue_once_with_major_change` | Discrimination strong; utility coupling weak/placebo-sensitive. | One redesign with calibration/ranking endpoint. |

## Strategic Pivot
- Keep discovery pressure on topological formulations that are not scalar score tweaks: multiparameter filtrations, stability trajectories, and motif-level topology.
- In geometry, shift from direct endpoint scores to interaction terms and transition dynamics that condition on the already-strong directed/signed branch.
- In cross-model work, optimize directly for null-gap robustness rather than raw transfer delta.

## Recommended Next 3
1. High-probability discovery: `N315` (support-margin multiparameter persistence).
2. High-risk/high-reward: `N326` (cross-model topological codebook transport with null-gap objective).
3. Cheap broad-screen: `N324` (ID heterogeneity as interaction-only feature block).
