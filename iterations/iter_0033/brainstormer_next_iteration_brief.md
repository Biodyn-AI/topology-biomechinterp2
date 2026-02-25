# Next Iteration Brief - iter_0033 -> iter_0034

## Research Gate
- `passed_min_research_gate = true`, so run a full discovery packet (not recovery-only).

## Packet Goal
- Confirm and sharpen the `H82` signal.
- Spend exactly one slot on a single major-change cross-model test.
- Keep one cheap orthogonal scan for rapid branching.

## Required 3-Slot Execution Packet

1. Slot A (high-probability): `N420` dual-filtration local witness persistence.
Scope: `immune/lung/external_lung`, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
Primary metric: `delta_auc_local_dual_filtration_plus_h70_minus_h70`.
Secondary metric: `null_gap_q95_local_dual_filtration_hotspot_gap`.
Keep gate: positive mean delta in `>=5/6` domain-splits and positive mean null-gap in `>=4/6`.
Fail-fast: if first two processed domains both have non-positive mean null-gap in both splits, stop multiseed expansion.

2. Slot B (high-risk/high-reward): `N429` cross-model barcode OT depth alignment.
Scope: seed42 pilot first on all 3 domains; expand to more seeds only if pilot passes.
Primary metric: `barcode_ot_depth_alignment_score`.
Secondary metrics: OT-regularized trajectory Spearman, profile retrieval.
Keep gate: positive null-gap in `>=2/3` domains on the primary metric.
Fail-fast: if `0/3` pilot domains clear null-gap, retire immediately.

3. Slot C (cheap broad-screen): `N433` sparse descriptor blend screen.
Scope: seed42 breadth run across domains, splits, layers `{0,3,7,11}`.
Primary metric: `delta_auc_descriptor_blend_minus_h70`.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fail-fast: if positive mean deltas are `<3/6`, retire without reseeding.

## Pre-Registered Nulls
- Slot A: support-margin shuffle within geodesic bins, matched-random hotspot sets, label permutation.
- Slot B: random transport plan with fixed marginals, depth-order permutation, module-label permutation.
- Slot C: descriptor-column shuffle within geodesic bins, endpoint swap, label permutation.

## Hard Retire Rules for iter_0034
- Do not run another `cross_model_alignment` variant unless Slot B hits its pilot keep gate.
- Do not run standalone additive graph-topology scores (SBC/curvature-only) in this packet.
- Do not allocate slots to standalone intrinsic-dimension AUROC-lift objectives.

## Deliverables
- `iterations/iter_0034/executor_iteration_report.md`
- `iterations/iter_0034/executor_hypothesis_screen.json`
- `iterations/iter_0034/executor_next_steps.md`
- Per-slot by-row/domain/null CSV artifacts.
- `iterations/iter_0034/iter0034_screen_summary.json`.
