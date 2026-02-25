# Executor Next Steps — iter_0006

1. Calibrate the rewiring null to reduce scale mismatch:
   - compare observed Euclidean PH against observed geodesic PH on the original kNN graph,
   - then test rewired geodesic PH as null against that same geodesic observed baseline.
2. Re-run immune full-layer split map with a connectivity-stable graph design:
   - increase kNN density target or use mutual-kNN + guaranteed-connectivity augmentation with explicit bridge-count reporting,
   - test sensitivity of conclusions to `k` and to bridge usage strata.
3. Biological anchoring on split-robust immune layers from this iteration (`7, 9, 10, 11` under feature-shuffle):
   - rank high-contribution genes,
   - run TRRUST/GO enrichment with FDR control,
   - check whether enriched modules are consistent across source/target splits.
4. Extend this full-layer split protocol to external-lung for cross-domain comparison with matched null calibration settings.
