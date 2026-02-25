# Brainstormer Next Iteration Brief - iter_0031

## Status and Goal
- Gate is green (`passed_min_research_gate=true`), so run a full 3-slot packet.
- Goal: recover biological specificity on the active `H70` lineage, take one high-upside cross-model shot with a new objective, and keep one cheap geometric breadth screen.

## Suggested Slot Mapping (next executor loop)
- `H79` <- `N395` (high-probability discovery).
- `H80` <- `N392` (high-risk/high-reward).
- `H81` <- `N389` (cheap broad-screen).

## Slot A (Primary): N395 - TF-Module Conditioned Immune/Source Rescue
- Hypothesis: immune/source failures come from TF-module heterogeneity and are recoverable with module-conditional support calibration.
- Scope:
  - Domains: `immune`, `lung`, `external_lung`.
  - Seeds: `42,43,44`.
  - Splits: `source_disjoint`, `target_disjoint`.
  - Layers: `{7,11}`.
- Design:
  - Keep `H70` triangle-defect backbone fixed.
  - Build module labels from TRRUST TF neighborhoods and STRING local-density tiers.
  - Estimate module-conditional support interaction instead of one global interaction.
- Controls:
  - Module-label permutation preserving module sizes.
  - Matched random module weights.
  - Label shuffle within geodesic bins.
- Keep gate:
  - Immune/source `delta_AUROC > 0`.
  - Global interaction mean `> 0`.
  - Interaction `null_gap_q95 > 0` in `>=2/6` domain-splits.

## Slot B (High-Risk): N392 - Cross-Model Pathway-Centroid Alignment
- Hypothesis: cross-model geometry aligns at pathway-centroid level even when gene-level mapping fails.
- Scope:
  - Start with `seed42` pilot.
  - Domains: held-out-domain protocol over `immune`, `lung`, `external_lung`.
  - Layers: `{7,11}`.
- Design:
  - Build pathway centroids (TRRUST/GO modules) in scGPT and Geneformer signature spaces.
  - Evaluate alignment with Procrustes, CKA, and top-k centroid retrieval.
  - Use these as primary endpoints (no edge-transfer AUROC in this slot).
- Controls:
  - Pathway-membership permutation preserving module sizes.
  - Random orthogonal basis baseline.
  - Signature-destroy permutation.
- Keep gate:
  - Immune centroid concordance metric above null q95.
  - At least one non-immune domain above null q95 on the same metric family.

## Slot C (Cheap Breadth): N389 - Neighbor-Dropout Detour Elasticity v2
- Hypothesis: true edges are geodesically robust to targeted neighborhood dropout.
- Scope:
  - `seed42`.
  - All domains and both disjoint splits.
  - Layers `{0,3,7,11}`.
- Design:
  - Redesign `H78` perturbation: remove top local-betweenness neighbors at `{10%,20%,30%}`.
  - Recompute detour inflation and elasticity descriptors and test incremental lift over baseline.
- Controls:
  - Matched random-node dropout with identical dropout budgets.
  - Endpoint swap within geodesic bins.
  - Label shuffle within bins.
- Keep gate:
  - Non-negative global mean `delta_AUROC`.
  - Positive mean in `>=4/6` domain-splits.
  - At least one domain-split with positive `null_gap_q95`.

## Required Outputs
- Per slot:
  - by-row CSV,
  - domain-summary CSV,
  - null-summary CSV.
- Packet summary:
  - `iterations/<next_iter>/iterXXXX_screen_summary.json`.
- Mandatory executor artifacts:
  - `iterations/<next_iter>/executor_iteration_report.md`
  - `iterations/<next_iter>/executor_hypothesis_screen.json`
  - `iterations/<next_iter>/executor_next_steps.md`

## Fast-Fail Rules
1. Stop Slot B expansion after pilot if immune fails null-q95 gate.
2. Stop Slot A expansion after half permutation budget if immune/source remains negative and interaction mean is non-positive.
3. Always finish Slot C for breadth, unless data/runtime failure blocks all slots.

## Minimal Recovery Plan (if a future gate fails)
1. Run Slot C only (`N389`) with seed42 and 24 permutations per null family.
2. Run reduced Slot A (`N395`) on immune/lung at layers `{7,11}` with module-label permutation only.
3. Emit valid `executor_hypothesis_screen.json` and `iterXXXX_screen_summary.json` before any additional expansion.
