# Next Steps — iter_0022

1. Promote `H50` to multiseed robustness.
   - Run the same directed/signed topology packet for seeds `42/43/44` and require consistent positive `delta_AUROC(directed-signed - distance-only)` in both disjoint splits with stronger directed nulls (degree-preserving edge swaps).

2. Keep `H49` active but narrow the objective.
   - Preserve the bifiltration method (already robust on edge discrimination) and retest utility linkage using explicit split-specific utility targets and held-out-domain calibration; do not claim utility coupling until placebo-calibrated support appears.

3. Retire the current `H51` motif-overlap formulation.
   - Degree-null enrichment improved, but module-shuffle control remained null (`0/18` significant module rows). Re-open cross-model work only under a materially changed utility/topology-transfer objective.
