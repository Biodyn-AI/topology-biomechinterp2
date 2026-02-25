# Executor Next Steps — iter_0013

1. Promote `H24` with a stricter unsupervised control:
- Run a correspondence-free variant (e.g., Sinkhorn/GW or layer-DTW + CCA warm start) and require improvement over both random-map and Procrustes/PCA baselines on distance/Jaccard/top-1.

2. Resolve `H22` domain-heterogeneity before retirement/promotion:
- Repeat the split×phase test with the same protocol but add per-domain biological anchors (TRRUST/STRING confidence strata) to test whether immune-only late inversion is biology-specific or noise.

3. Retire current `H23` curvature direction unless rescued with a materially changed method:
- If retried, switch from edge-level Forman curvature to path-level/triangle-aware curvature proxies with explicit degree-matched controls; otherwise mark branch as retired after this second non-supportive outcome.

4. Keep null-sensitivity rewiring branch retired by default:
- No new rewiring-survival reruns without explicit rescue rationale and method change.
