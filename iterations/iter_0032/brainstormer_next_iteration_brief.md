# Next Iteration Brief - iter_0032 -> iter_0033

## Objective
- Keep momentum on the strongest positive branch (`H70` lineage).
- Stop spending on stale utility endpoints.
- Re-open cross-model work only via invariance/stability objectives.

## Required 3-Slot Packet

1. Slot A (high-probability): `N399` local witness-cycle persistence on H70 hotspots.
Scope: immune/lung/external_lung, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `{7,11}`.
Primary metric: `delta_auc_local_cycle_plus_h70_minus_h70`.
Secondary metric: local-cycle `null_gap_q95`.
Keep gate: `>=4/6` domain-splits with positive mean delta and positive mean null-gap.
Fast-fail rule: if first two domains both show negative mean null-gap in both splits, stop expansion and preserve compute.

2. Slot B (high-risk/high-reward): `N407` cross-model pathway persistence-trajectory invariance.
Scope: seed42 pilot first; domains immune/lung/external_lung; layers `{0,3,7,11}`.
Primary metric: pathway trajectory concordance (distance Spearman across depth).
Secondary metrics: trajectory CKA and top-k pathway retrieval.
Keep gate: `>=2/3` domains with positive mean null-gap on primary metric.
Fast-fail rule: if all domains have negative null-gap at pilot stage, retire this variant immediately.

3. Slot C (cheap broad-screen): `N412` shortcut-bridge competition index.
Scope: seed42 breadth screen; domains immune/lung/external_lung; splits `source_disjoint/target_disjoint`; layers `{0,3,7,11}`.
Primary metric: `delta_auc_sbc_index_minus_baseline`.
Keep gate: positive mean delta in `>=4/6` domain-splits.
Fast-fail rule: if positive mean domain-splits `<3/6`, retire without multiseed expansion.

## Pre-Registered Nulls
- Slot A: endpoint swap, matched-random edge sets, label shuffle.
- Slot B: pathway-label permutation, depth-order permutation, signature-destroy.
- Slot C: endpoint swap, feature shuffle, label shuffle.

## Deliverables Required
- `executor_iteration_report.md`
- `executor_hypothesis_screen.json`
- domain summary + null summary CSVs for all three slots
- one compact machine summary JSON (`iter0033_screen_summary.json`)

## Branching Policy After iter_0033
- If Slot A passes and Slot C is non-negative: continue mechanism localization around H70.
- If Slot B passes: create a dedicated cross-model invariance track, separate from utility transfer claims.
- If only Slot C passes: use it as a low-cost triage feature in future breadth screens.
