# Brainstormer Structured Feedback: iter_0052

## Gate and state
- `passed_min_research_gate`: `true` (from `iterations/iter_0052/executor_research_validation.json`).
- Iteration outcome: one active promotion (`H139`), one neutral branch (`H140`), one diagnostic/inconclusive branch (`H141`).

## Iteration signal readout
- `H139` remains the strongest active branch: mean `delta_vs_h70=+0.03135`, positive mean null-gap in `6/9` domain-splits, strict-positive rows `16/25`.
- Hard-slice status improved but is incomplete: `immune/source_disjoint` and `lung/dual_axis_disjoint` are now mean null-gap positive, while `external_lung/dual_axis_disjoint` is negative (`-0.06927`) with only one evaluable seed row.
- `H140` shows directional neighborhood stability (`mean delta_gain_vs_swap=+0.03374`, `19/24` gain-positive rows) but is single-seed and still split-fragile (`immune/source_disjoint`, `lung/dual_axis_disjoint`).
- `H141` confirms residual fragility map: mean strict margin is negative (`-0.00523`), with concentrated failures in `external_lung/*`, `lung/source_disjoint`, `lung/target_disjoint`, and `immune/dual_axis_disjoint`.

## Stale direction triage
| Direction | Evidence pattern | Action | Why now |
|---|---|---|---|
| Cross-model alignment via descriptor-only or correspondence-free mapping (`H119/H122/H125/H131/H137`) | Repeated negatives; latest runs keep domain null-gap at or near `0/3` | `retire_now` | Multiple method resets did not change outcome distribution |
| Additive ontology/module hardening lineage (`H124/H127/H130/H135/H138`) | Strong directional deltas with recurring strict-null collapse, including repeated hard-slice negatives | `rescue_once_with_major_change` | One rescue is still justified only if mechanism is changed (causal/anchor-driven), not additive feature tweaks |
| Standalone additive intrinsic-dimension uplift (`H114/H134` line) | Directional-only or negative; repeated `0/6` null-positive split patterns | `retire_now` | Low rescue potential in current endpoint form |
| Rank-surface / scalar PH surrogates (`H133` and predecessors) | Decisive negatives with no null survival | `retire_now` | Branch is exhausted without changing filtration object |
| Lightweight manifold descriptor tweaks (`H129/H132` style) | Small directional lifts but strict-null failure | `rescue_once_with_major_change` | Allow at most one reset tied to fail-slice mechanism |

## Navigation guidance
- Keep the next packet centered on fail-slice closure for `H139`, not global retuning.
- Treat `H140` as a replication task (seeds `43/44`) before using it as a claim.
- Use `H141` as a routing map for experiments: prioritize `external_lung/*` first, then lung source/target, then immune dual-axis.
- If the next executor gate fails unexpectedly, run a minimal recovery packet with one continuity test (`H139` fail-slice-only rerun), one cheap diagnostic (`null-dominance decomposition`), and one low-cost broad screen (`depth motif tokens`) to restore valid machine artifacts quickly.
