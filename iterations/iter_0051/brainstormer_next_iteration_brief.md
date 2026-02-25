# Brainstormer Next Iteration Brief (for iter_0052 executor)

## Context checkpoint
- Research gate is currently valid (`passed_min_research_gate=true`), so proceed with normal 3-slot execution.
- Portfolio priority: one continuity validation (`H136` lineage), one major-reset high-risk test (cross-model), one cheap orthogonal broad screen.

## Required 3-slot packet
1. **Slot A (high-prob): `P01` H136 robustness expansion**
- Scope: seeds `{42,43,44}`; domains `{immune, lung, external_lung}`; splits `{source_disjoint, target_disjoint, dual_axis_disjoint}`; layers `{7,11}`.
- Deliverables: by-row CSV, domain-split summary CSV, null summary CSV, machine-readable JSON summary.
- Decision gate: keep only if positive mean null-gap `>=3/9` domain-splits and one hard slice non-negative.

2. **Slot B (high-risk): `P09` TF-anchored cross-model persistence-image OT pilot**
- Scope: seed42 pilot across all domains and both disjoint splits at layers `{7,11}`.
- Deliverables: per-domain transfer/alignment table + domain null summary.
- Fast-fail gate: if positive mean null-gap domains `0/3`, retire immediately.

3. **Slot C (cheap broad screen): `P14` motif automaton recurrence**
- Scope: seed42; all domains; source/target disjoint; layers `{0,3,7,11}`.
- Null budget: low-cost (`16` permutations per null family) to preserve breadth.
- Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=1/6`.

## Execution notes
- Reuse existing loader/utilities from `iterations/iter_0051/run_iter0051_screen.py` and prior baseline modules.
- Preserve strict-null reporting (`mean_null_gap_q95`, row-level and domain-level).
- Keep artifacts machine-readable and aligned with prior naming conventions.

## Minimal recovery fallback (only if run is blocked)
- If Slot B cannot run due missing cross-model inputs, do not replace with another cross-model tweak.
- Replace Slot B with `P04` (persistent entropy slope broad screen, seed42/all domains/splits/layers `{0,3,7,11}`) so iteration still yields 3 valid experiments.
