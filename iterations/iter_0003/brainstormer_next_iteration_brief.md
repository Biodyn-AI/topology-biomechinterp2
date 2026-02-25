# Brainstormer Next Iteration Brief — iter_0003

## Gate Context
- Current gate is passed (`passed_min_research_gate=true`), so next iteration should maximize screening value, not recovery triage.

## Prioritized Plan

### 1) Primary Experiment (do first)
- Experiment: cross-domain topology replication (`N01`).
- Goal: test whether the strong lung H1 signal appears in immune and external-lung.
- Minimal implementation:
  1. Extend `run_iter0003_screen.py` logic to include immune/external-lung seed runs.
  2. Keep protocol fixed (`n_points=350`, `PCA=20`, `n_null=20`, ripser H1 summary).
  3. Emit domain-layer summary CSV + iteration-level JSON.
- Success criterion: at least 2 domains show majority-layer significance (`>50%` layers with Fisher p `<0.05`) and positive mean delta.

### 2) Backup Experiment (if primary stalls)
- Experiment: stronger null stress test on current lung signal (`N03`).
- Goal: rule out null-specific artifact risk quickly.
- Minimal implementation:
  1. Focus on lung layers `L0/L1/L7/L9`.
  2. Add at least two stronger nulls (distance permutation + kNN rewiring).
  3. Report per-layer empirical p-values against each null.
- Success criterion: top layers remain significant under at least one stronger null family.

### 3) Stretch Experiment
- Experiment: residual-level cross-model alignment (`N06`).
- Goal: move H02 from low-dimensional feature summaries to direct manifold alignment.
- Minimal implementation:
  1. Surface matched-gene residual vectors for Geneformer and scGPT.
  2. Run CKA/Procrustes + permutation nulls per domain.
  3. Compare alignment magnitude to current feature-summary baseline.
- Success criterion: significant above-null alignment in at least 2 of 3 domains.

## Fast Recovery Plan (only if next gate fails)
- Run a minimal salvage packet in one pass:
  1. Keep only one domain (lung) and one hypothesis family (topology).
  2. Produce one machine artifact table + one summary JSON with explicit null calibration.
  3. Document failure cause (data access vs statistical failure) in hypothesis screen.
- This keeps iteration velocity while preserving scientific screening integrity.
