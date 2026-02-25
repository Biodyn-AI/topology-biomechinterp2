# Brainstormer Structured Feedback - iter_0025

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0025/executor_research_validation.json`).
- No recovery-only mode required; proceed with normal hypothesis expansion.

## What Changed This Iteration
- `H58` failed as a rescue despite preserving baseline strength: weighted-vs-distance stayed positive (`+0.01137` mean), but weighted-vs-unweighted was slightly negative (`-0.00052` mean), and source-disjoint failures remained negative in both `lung` and `external_lung`.
- `H59` was directionally positive (`+0.0240` mean delta) but null-dominated: random-map and signature-destroy null deltas were typically larger than observed, so `p_best < 0.05` in `0/12` rows.
- `H60` was net negative overall (`-0.00435` mean delta), with signal concentrated only in late layer (`11`), indicating endpoint ID-jump is the wrong formulation.

## Stale Direction Triage
| Direction | Status | Why now | Action for next loop |
|---|---|---|---|
| Directed/signed biological weight tweaks (`H58`-style) | `retire_now` | Second pass did not rescue failure slices; effect vs unweighted is effectively zero/negative. | Stop parameter/weight tweaks; only re-open with a changed topology representation class. |
| Standalone endpoint ID-jump (`H60` + prior ID endpoint lines) | `retire_now` | Repeated neutral/negative outcomes; broad screen failed with multiseed controls. | Pivot to transition-based geometry (layer-delta ID/tangent transport), not endpoint jump. |
| Path-homology utility-transfer endpoint (`H53/H56` line) | `retire_now` | Directional discrimination with repeated failure on utility-transfer gate. | Do not spend more on this endpoint form. |
| Standalone anisotropy-tail/geodesic-tail endpoint (`H57`) | `retire_now` | Broad-screen keep gate failed and domain instability persists. | Reuse only as interaction feature with stronger topological signals. |
| Cross-model raw-Procrustes transfer objective (`H59` current objective) | `rescue_once_with_major_change` | Observed deltas are smaller than random-map nulls; objective is mis-specified. | One redesign using null-gap objective and/or biologically anchored correspondence supervision. |
| Bifiltration-to-utility coupling claim (`H49`-style endpoint) | `rescue_once_with_major_change` | Discrimination can be strong, utility linkage weak/placebo-sensitive. | One redesign on ranking-calibration transfer instead of hard threshold transfer. |

## Strategic Pivot
- Keep the global direction anchored in topology, but move from scalar score tweaks to **structure-aware formulations** (multiparameter persistence, stability profiles, motif persistence).
- In cross-model work, optimize against **null-gap** rather than raw delta.
- In geometry work, prioritize **layer-transition dynamics** over static endpoint descriptors.

## Execution Priorities for iter_0026
1. High-probability: multiparameter support-margin persistence rescue focused on source-disjoint lung/external-lung slices.
2. High-risk/high-reward: biologically anchored cross-model topology alignment with explicit random-map null-gap objective.
3. Cheap broad-screen: layer-transition ID-gradient/tangent-consistency screen across full multiseed matrix.
