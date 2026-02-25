# Executor Next Steps — iter_0020

1. **Stress-test H43 with stronger transfer criterion**
- Re-run `H43` on seeds `{42,43,44}` with the same STRING+ontology support construction.
- Add an out-of-domain holdout check (train on two domains, evaluate on held-out third) and require positive interaction + non-negative AUROC delta in at least `2/3` held-out evaluations.

2. **Calibrate H44 significance resolution and utility linkage**
- Increase `H44` permutation budget (`>=160` per row) to move beyond the current p-value floor.
- Add a downstream linkage test: regress edge-level geometry uplift on layer-wise zigzag lifetime excess to verify decision relevance.

3. **Retire or redesign H45 quickly**
- Treat current H45 formulation as non-promotable.
- If retained, enforce a materially changed metric definition (larger holdout blocks or pooled robust rank-based targets) and drop trimmed leave-layer-out rows that are structurally underpowered.
