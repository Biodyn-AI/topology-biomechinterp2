# Brainstormer Next Iteration Brief (for iter_0053 executor)

## Context checkpoint
- Current gate status is valid (`passed_min_research_gate=true`).
- Next packet should prioritize strict-margin closure on the active manifold branch while keeping one high-risk reset and one cheap breadth slot.

## Required 3-slot packet
1. **Slot A (high-prob): `Q01` adaptive-neighborhood fail-slice rescue for `H139` lineage**
- Scope:
  - seeds `{42,43,44}`
  - layer `11`
  - target slices: `external_lung/{source,target,dual_axis}`, `lung/{source,target}`, `immune/dual_axis`
  - adaptive neighborhood schedule (density-aware `k` in bounded range, e.g., 8-20)
- Deliverables:
  - `h142_adaptive_fail_slice_by_seed_domain_split.csv`
  - `h142_adaptive_fail_slice_domain_split_summary.csv`
  - `h142_adaptive_fail_slice_null_summary.csv`
- Decision gate:
  - external-lung dual-axis has `>=2` evaluable seed rows, and
  - at least `3/6` fail slices show mean `strict_margin >= 0`.

2. **Slot B (high-risk/high-reward): `Q07` TF-module persistence-image cross-model pilot**
- Scope:
  - seed42 pilot
  - domains `{immune, lung, external_lung}`
  - splits `{source_disjoint, target_disjoint}`
  - layer `11` (pilot-first)
- Deliverables:
  - `h143_tf_module_cross_model_by_domain_split.csv`
  - `h143_tf_module_cross_model_domain_summary.csv`
  - `h143_tf_module_cross_model_null_summary.csv`
- Fast-fail gate:
  - retire immediately if positive domain mean null-gap is `0/3`.

3. **Slot C (cheap broad screen): `Q11` depth motif token screen**
- Scope:
  - seed42
  - domains `{immune, lung, external_lung}`
  - splits `{source_disjoint, target_disjoint, dual_axis_disjoint}` where evaluable
  - layers `{0,3,7,11}`
- Deliverables:
  - `h144_depth_motif_tokens_by_domain_split.csv`
  - `h144_depth_motif_tokens_domain_summary.csv`
  - `h144_depth_motif_tokens_null_summary.csv`
- Keep gate:
  - positive mean delta in at least half of evaluable domain-splits,
  - and at least one strict-null-positive domain-split.

## Execution constraints
- Do not reopen retired cross-model descriptor-only formulations.
- Reuse current `H139/H141` strict-margin machinery so slot outputs remain directly comparable.
- Keep artifact naming and schema consistent with prior iterations (`by_*`, `domain_split_summary`, `domain_summary`, `null_summary`).

## Minimal recovery plan (only if next executor gate fails)
- Run a compact 3-experiment salvage packet that restores valid machine artifacts fast:
1. `Q01-lite`: seed42-only fail-slice rerun on `external_lung/dual_axis` + `lung/source`.
2. `Q12`: null-dominance decomposition on existing `H139` rows (no new heavy training).
3. `Q11-lite`: motif screen on `{immune,lung}` and layers `{7,11}` only.
- Require each salvage slot to emit one machine-readable CSV plus one summary CSV to pass validation quickly.
