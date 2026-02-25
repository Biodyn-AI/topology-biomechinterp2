# Brainstormer Next Iteration Brief - iter_0026

## Operating Goal
Run one 3-slot packet that (a) targets the known source-disjoint failure slices with a major topology change, (b) gives one true high-upside cross-model rescue, and (c) cheaply screens a broad geometric alternative.

## Slot Plan

### Slot A (Primary, high-probability): N301 - Support-Margin Multiparameter Persistence
- Hypothesis: 2-parameter filtration over distance and directed-support margin rescues `lung/external_lung` source-disjoint failure slices.
- Design:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44` (same as H55/H58 coverage where feasible).
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: start with `{7,11}`; expand to `{0,3,7,11}` only if runtime permits.
  - Compare model: `H55` directed/signed baseline + N301 features (nested model).
- Required controls:
  - Margin permutation within degree bins.
  - Support permutation within degree bins.
  - One-parameter ablation (distance-only or margin-only filtration).
- Keep gate:
  - `lung/source_disjoint` mean incremental AUROC >= `0`.
  - `external_lung/source_disjoint` mean incremental AUROC >= `0`.
  - Global positive mean incremental AUROC in `>=4/6` domain-splits.

### Slot B (High-risk/high-reward): N310 - Biologically Anchored Contrastive Cross-Model Alignment
- Hypothesis: anchor-supervised alignment (TRRUST/DoRothEA/STRING) yields real cross-model transfer that survives nulls.
- Design:
  - Seed: start with `seed42` pilot (matching H59 budget), then expand only if signal appears.
  - Layers: `{7,11}`.
  - Objective: optimize transfer **null-gap** (`observed delta - random-map/signature-destroy quantile`) rather than raw delta.
  - Anchors: high-confidence TF-target and hub-neighborhood pairs only.
- Required controls:
  - Random-map alignment.
  - Signature-destroy permutation.
  - Anchor-label shuffle / degree-matched fake anchors.
- Keep gate:
  - Domain-level `p<0.05` in at least `2/3` domains.
  - Positive mean transfer delta and positive mean null-gap.

### Slot C (Cheap broad screen): N307 - Layer-Transition ID Gradient
- Hypothesis: ID information is in layer transitions, not same-layer endpoint jumps.
- Design:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: both disjoint regimes.
  - Layer transitions: `0->3`, `3->7`, `7->11`.
  - Features: edge-level `ΔID_two_nn`, `ΔID_mle`, transition-consistency score; test incremental over geodesic baseline.
- Required controls:
  - Layer-order permutation per gene.
  - Endpoint swap within degree bins.
  - Label permutation.
- Keep gate:
  - Positive mean incremental AUROC in `>=4/6` domain-splits.
  - At least one Fisher-significant aggregate.

## Output Requirements
- Produce per-slot:
  - by-row CSV (`..._by_seed_layer_split.csv` or `..._by_domain_layer.csv`),
  - domain summary CSV,
  - null summary CSV,
  - compact JSON summary (`iter0026_screen_summary.json`).
- Update:
  - `iterations/iter_0026/executor_iteration_report.md`
  - `iterations/iter_0026/executor_hypothesis_screen.json`
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (+ compile PDF)

## Fast-Fail / Budget Rules
1. If Slot A fails both failure slices decisively (mean delta < `-0.005` in both) at layer `{7,11}`, stop A expansion.
2. If Slot B null-gap remains <= `0` in all domains after pilot, do not seed-expand.
3. Preserve Slot C even if A/B fail; it is the low-cost breadth check.

## Contingency (only if next iteration gate fails)
- Minimal executable recovery packet:
  1. Run Slot C only on `seed42`, both splits, layers `{0,7,11}`.
  2. Produce one machine-readable summary + valid hypothesis screen JSON.
  3. Defer cross-model and multiparameter runs until gate returns green.
