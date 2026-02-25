# Executor Next Steps (iter_0051)

1. Promote `H136` to a robustness packet.
- Run seeds `{42,43,44}` with splits `{source_disjoint,target_disjoint,dual_axis_disjoint}` at layers `{7,11}`.
- Keep/stop gate: positive mean null-gap in at least `3/9` domain-splits and at least one hard split (`immune/source` or `lung/dual_axis`) non-negative.

2. Retire the tested `H137` endpoint.
- Do not spend another iteration on correspondence-free descriptor alignment using current Geneformer edge-only artifacts.
- Re-open cross-model only with materially different inputs (residual embeddings or perturbation-coupled anchors).

3. Close `H138` additive sheaf-hardening form.
- No further local tuning on current feature set.
- If revisited, require a major method change: hard-slice-targeted causal chart construction and hard-slice-specific adversarial nulls.
