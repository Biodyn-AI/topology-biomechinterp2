# Brainstormer Next Iteration Brief — iter_0005

## Gate Context
- Current gate passed (`passed_min_research_gate=true`), so next iteration should maximize information gain on robustness, not do recovery triage.

## Prioritized Plan

### 1) Primary Experiment
- Experiment: `N21 + N22` combined packet (replace null + full-layer split map in immune).
- Why first: iter_0005 uncertainty is concentrated in two points only: null validity and split brittleness localization.
- Minimal implementation:
  1. In the iter_0005 pipeline, replace `distance_permutation` with degree-preserving kNN rewiring/geodesic null.
  2. Run immune across all 12 layers, both split regimes, 3 seeds, same sample/PCA defaults as iter_0005.
  3. Keep feature-shuffle branch in parallel for calibration and comparability.
  4. Emit per-seed CSV + per-layer summary + pass matrix (`both_splits_sig` per layer).
- Success criterion:
  - At least 6/12 immune layers pass both splits under at least one null family, and top immune layer remains significant.

### 2) Backup Experiment
- Experiment: `N28` biological anchoring on currently split-robust positives.
- Trigger: use if rewiring null implementation/runtime blocks primary progress.
- Minimal implementation:
  1. Start with lung `L0` and external-lung `L11` (already split-robust in iter_0005).
  2. Rank genes by H1 contribution (ablation-drop or leave-one-out approximation).
  3. Run TRRUST/DoRothEA/GO enrichment with FDR correction and report direct evidence + interpretation.
- Success criterion:
  - At least one regulator/pathway family enriched in at least two domains/layers tested.

### 3) Stretch Experiment
- Experiment: `N26` residual-level cross-model topology alignment.
- Minimal implementation:
  1. Surface matched-gene scGPT and Geneformer residual embeddings for shared domains.
  2. Compute persistence-image/Betti-curve similarity and calibrate with permutation null.
  3. Compare alignment in split-robust vs split-brittle layer sets.
- Success criterion:
  - Above-null cross-model topology alignment in at least two domains.

## Contingency (If Next Gate Fails)
- Run a narrow recovery packet in one fast cycle:
  1. Single domain (`immune`), single family (`topology`), single stronger null (`rewiring`).
  2. Restrict to top 4 candidate layers by prior effect size.
  3. Require one machine CSV and one machine JSON with explicit pass/fail thresholds.
- This preserves screening velocity while still improving scientific validity.
