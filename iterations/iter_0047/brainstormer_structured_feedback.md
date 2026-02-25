# Brainstormer Structured Feedback - iter_0047

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0047/executor_research_validation.json`).
- Full 3-slot execution is valid; no recovery-only restriction is required.

## Iteration Evidence Snapshot
- `H124` (`module_structure`) is still the strongest active branch but not promotion-ready under strict nulls.
  - mean `delta_vs_h70 = +0.13098` across `27` rows
  - positive direction in `27/27` rows and `9/9` domain-splits
  - positive mean null-gap in only `4/9` domain-splits
  - restored `lung/dual_axis_disjoint` coverage, but mean null-gap remains slightly negative (`-0.00326`)
- `H125` (`cross_model_alignment`) is decisively negative for this endpoint.
  - mean `transfer_delta_auc_vs_h70 = +0.09855`
  - positive domain null-gap count `0/3`
  - immune mean null-gap `-0.01894`
- `H126` (`manifold_distance`) is a viable continuation branch.
  - mean `delta_vs_h70 = +0.04421` across `12` rows
  - positive mean null-gap in `2/6` domain-splits
  - one source-disjoint split is null-positive (`immune/source_disjoint`)

## New Artifacts Reviewed (iter_0047)
- `iterations/iter_0047/h124_signed_string_hardening_by_seed_domain_split.csv`
- `iterations/iter_0047/h124_signed_string_hardening_domain_summary.csv`
- `iterations/iter_0047/h124_signed_string_hardening_null_summary.csv`
- `iterations/iter_0047/h125_anchor_cycle_alignment_by_domain_split_layer.csv`
- `iterations/iter_0047/h125_anchor_cycle_alignment_domain_split_summary.csv`
- `iterations/iter_0047/h125_anchor_cycle_alignment_domain_summary.csv`
- `iterations/iter_0047/h125_anchor_cycle_alignment_null_summary.csv`
- `iterations/iter_0047/h126_geodesic_torsion_by_domain_split_layer.csv`
- `iterations/iter_0047/h126_geodesic_torsion_domain_summary.csv`
- `iterations/iter_0047/h126_geodesic_torsion_null_summary.csv`
- `iterations/iter_0047/iter0047_screen_summary.json`
- Cumulative sources checked: `reports/autoloop_master_log.md`, `paper/autoloop_research_paper.tex`

## Stale Direction Triage
1. Cross-model direct transfer utility endpoints (`H96/H99/H102/H109/H119/H122/H125`) -> `retire_now`.
Reason: repeated null-gap collapse after multiple objective resets.

2. Additive/standalone topology-stability utility variants (`H90/H92/H107/H110/H111/H112/H113`) -> `retire_now`.
Reason: repeated directional-but-non-robust behavior (`0/6` or near-`0/6` null-gap support).

3. Standalone additive intrinsic-dimension utilities (`H60/H63/H66/H98/H114`) -> `retire_now`.
Reason: long negative run; no credible robustness recovery signal.

4. Scalar additive PH rescue chain (`H100/H101/H103/H106`) -> `retire_now`.
Reason: repeated failure to clear strict nulls despite filtration changes.

5. Cross-model alignment as a family -> `rescue_once_with_major_change` only.
Constraint: allow one slot only if objective shifts to structure-invariant alignment stability (not direct edge-transfer AUROC).

6. `H124` lineage (`H116/H118/H123/H124`) -> `keep_active_high_priority`.
Reason: strongest directional and partial null-robust branch with an identifiable bottleneck (split-specific null calibration).

7. `H126` lineage (`H121/H126`) -> `rescue_once_with_major_change`.
Constraint: target external_lung/source null fragility with path-ensemble and scale-aware geometry features.

## Strategic Pivot
- Keep one exploitation slot on `H124` hardening with stricter adversarial controls focused on null-gap calibration, not raw delta.
- Keep one manifold slot on `H126` expansion, but require source-disjoint null survival as the primary gate.
- Keep at most one cross-model slot and require a non-utility primary objective (alignment stability under perturbation/topology constraints).

## Minimal Recovery Plan (only if a future gate flips to `false`)
1. Run a low-cost `H126`-line sanity packet first (seed42, layers `{7,11}`, null budget `>=12`) to keep machine artifact continuity.
2. Run a narrowed `H124` packet on `lung` + `immune` dual-axis splits with full strict nulls to re-establish the hardest calibration case.
3. Defer any cross-model slot until the gate returns to `true`.
