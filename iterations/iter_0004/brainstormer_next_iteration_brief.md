# Brainstormer Next Iteration Brief — iter_0004

## Gate Context
- Current gate passed (`passed_min_research_gate=true`), so prioritize high-information stress tests over recovery triage.

## Prioritized Plan

### 1) Primary Experiment
- Experiment: stronger-null + split-robustness packet for topology (`N11` + `N12`).
- Why first: this is the fastest way to convert repeated positive topology evidence into a defensible claim.
- Minimal implementation:
  1. Reuse `iter_0004` pipeline and add two stronger null families (distance permutation and kNN rewiring).
  2. Run on top and weak layers in lung/immune/external-lung.
  3. Add source-disjoint and target-disjoint gene split reruns with the same H1 summary.
  4. Emit one machine summary JSON + per-layer CSV with per-null/per-split p-values.
- Success criterion: majority of tested layers remain significant (`Fisher p < 0.05`) under at least one stronger null and both split regimes.

### 2) Backup Experiment
- Experiment: biological anchoring of high-persistence genes (`N18`).
- Trigger: use if primary run is compute-heavy or blocked by null implementation details.
- Minimal implementation:
  1. Select top layer per domain from current outputs (lung L0, immune L7, external-lung L0).
  2. Compute H1-contribution ranking (ablation-drop proxy is acceptable for first pass).
  3. Run TRRUST/DoRothEA/GO enrichment with FDR.
- Success criterion: at least one regulator/pathway family enriched in at least two domains.

### 3) Stretch Experiment
- Experiment: residual-level cross-model alignment (`N16`).
- Minimal implementation:
  1. Surface matched-gene residual tensors for scGPT and Geneformer in shared domains.
  2. Run CKA and Procrustes with gene-label permutation nulls.
  3. Compare alignment strength on all genes vs topology-core genes.
- Success criterion: above-null alignment in at least two domains, with stronger effect on topology-core genes.

## Recovery Plan (Only If Next Gate Fails)
- Keep iteration scope to one robust packet:
  1. One domain (immune), one family (topology), one stronger null.
  2. Produce at least one CSV and one JSON artifact with explicit pass/fail thresholds.
  3. Log blocker root cause separately (data access, runtime, or statistical failure).
- This keeps screening velocity while preserving scientific validity.
