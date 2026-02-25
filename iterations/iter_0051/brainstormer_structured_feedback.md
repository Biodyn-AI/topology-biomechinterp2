# Brainstormer Structured Feedback: iter_0051

## Gate and state
- `passed_min_research_gate`: `true` (from `executor_research_validation.json`).
- Iteration outcome: 1 promotable continuation (`H136`), 2 negatives (`H137`, `H138`).

## Iteration signal readout
- `H136` (`manifold_distance`): directional lift is real but narrow (`mean delta +0.0257`, positive mean null-gap in `2/6` domain-splits).
- Null-surviving slices are concentrated in `lung/source_disjoint` (`+0.01017`) and `immune/target_disjoint` (`+0.00057`); other slices remain null-negative.
- `H137` (`cross_model_alignment`): near-zero effect (`+0.00194`) and no domain-level null survival (`0/3`), including strong immune failure (`-0.09748`).
- `H138` (`module_structure`): repeated directional-only pattern (`+0.1338`, `9/9` positive mean deltas) with strict-null failure (`0/9` positive mean null-gap).

## Stale direction triage
| Direction | Evidence pattern | Action | Rationale |
|---|---|---|---|
| Cross-model alignment using edge/signature descriptors without new biological anchors | Family-level history is dominated by negatives; latest major reset `H137` still fails `0/3` domain null-gap gate | `retire_now` | Repeated method resets are not changing the outcome distribution; continue only if inputs/objective change materially |
| Additive ontology/module hardening on H130 lineage (`H124 -> H127 -> H130 -> H135 -> H138`) | Strong directional uplift but persistent strict-null failure, including repeated hard-slice negatives | `rescue_once_with_major_change` | Keep only one attempt with causal chart construction and hard-slice-specific adversarial nulls; stop local feature tweaks |
| Standalone/additive intrinsic-dimension descriptor uplift variants | Multiple rounds of directional-but-null-fragile or negative outcomes (`H98`, `H134`, earlier ID add-ons) | `retire_now` | Current endpoint form is exhausted; ID can return only as a constrained interaction term with explicit mechanism |
| Rank-surface / surrogate PH add-ons over H70 | Recent surrogates remain negative (`H133` and predecessor variants) | `retire_now` | Low rescue potential without changing filtration objective itself |
| Lightweight manifold descriptor add-ons that only chase small directional deltas | Many variants collapse under strict nulls (`H132`, `H134`) | `rescue_once_with_major_change` | Allow one reset that explicitly targets null survival, not marginal AUROC lift |

## Navigation guidance for next loop
- Keep one continuity slot on `H136` with multiseed + dual-axis validation and hard-slice gate.
- Spend one slot on a true high-risk reset that changes representational object (not another descriptor reshuffle).
- Spend one cheap breadth slot that can be screened in one seed and all domains/splits to preserve discovery velocity.
