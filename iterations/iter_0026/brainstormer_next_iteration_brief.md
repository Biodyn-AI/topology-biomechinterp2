# Brainstormer Next Iteration Brief - iter_0027

## Operating Goal
Execute one 3-slot packet that (1) attempts a direct rescue of known source-disjoint failures via a new topology formulation, (2) runs one true high-risk cross-model redesign with strict null-gap gating, and (3) keeps low-cost geometric breadth.

## Slot Plan

### Slot A (Primary, high-probability): N315 - Support-Margin Multiparameter Persistence
- Hypothesis: 2-parameter persistence over distance and directed-support margin recovers `lung/external_lung` source-disjoint slices.
- Design:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: start `{7,11}`, expand to `{0,3,7,11}` only if runtime permits.
  - Model comparison: `H55` directed/signed baseline vs baseline + N315 topological features.
- Controls:
  - Margin shuffle within degree bins.
  - Support-score shuffle within degree bins.
  - One-parameter filtration ablation.
- Keep gate:
  - `lung/source_disjoint` mean incremental AUROC `>= 0`.
  - `external_lung/source_disjoint` mean incremental AUROC `>= 0`.
  - Positive mean incremental AUROC in `>=4/6` domain-splits.

### Slot B (High-risk/high-reward): N326 - Cross-Model Topology Codebook Transport
- Hypothesis: OT transport across discrete topological tokens beats Procrustes-style alignment on null-gap robustness.
- Design:
  - Start with `seed42` pilot at layers `{7,11}`; expand only if pilot passes immune gate.
  - Build per-model topology codebooks from local signatures.
  - Learn OT mapping on shared genes and score held-out edges by token compatibility.
  - Optimize/report `null_gap_q95` as primary endpoint, not raw transfer delta.
- Controls:
  - Random codebook mapping.
  - Token-frequency matched shuffle.
  - Signature-destroy permutation.
- Keep gate:
  - Immune mean `null_gap_q95 > 0`.
  - At least one non-immune domain with positive mean null-gap.

### Slot C (Cheap broad-screen): N324 - ID Interaction-Only Broad Screen
- Hypothesis: ID features provide signal only through interaction with directed support/margin, not as direct predictors.
- Design:
  - Domains/seeds/splits matched to Slot A.
  - Layers/transitions: `0->3`, `3->7`, `7->11`.
  - Features: ID variance/gradient + interaction terms with directed support/margin from strong topology branch.
  - Compare against directed baseline without ID interactions.
- Controls:
  - Interaction partner shuffle.
  - Layer-order permutation per gene.
  - Label permutation.
- Keep gate:
  - Global mean incremental AUROC `>= 0`.
  - At least one Fisher-significant domain-split aggregate.

## Output Requirements
- Produce, per slot:
  - by-row CSV,
  - domain summary CSV,
  - null summary CSV,
  - compact JSON rollup (`iter0027_screen_summary.json`).
- Update mandatory iteration artifacts:
  - `iterations/iter_0027/executor_iteration_report.md`
  - `iterations/iter_0027/executor_hypothesis_screen.json`
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (+ compiled PDF)

## Fast-Fail Rules
1. If Slot A remains negative in both source-disjoint failure slices after `{7,11}` pilot (`mean delta < -0.003` for both), stop expansion.
2. If Slot B immune null-gap is non-positive at pilot stage, stop cross-model expansion immediately.
3. Always finish Slot C; it is the low-cost breadth-preserving screen.

## Gate-Failure Contingency (If Next Executor Gate Fails)
- Minimal executable recovery packet:
  1. Run Slot C only on `seed42`, both disjoint splits, transitions `0->3` and `7->11`.
  2. Write one valid machine summary and hypothesis screen JSON.
  3. Defer Slot A/B until gate returns green.
