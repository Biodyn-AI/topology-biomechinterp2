# Executor Next Steps — iter_0016

1. Promote `H32` to full robustness mode:
- Expand from seed42-only to seeds 42/43/44.
- Keep both disjoint splits and layers `0,3,7,11`.
- Add a diffusion-adjusted combo baseline to test whether convexity/detour remains incremental beyond diffusion features.

2. Run one targeted rescue for `H31`:
- Focus on immune + external_lung where signal is strongest.
- Add stronger nuisance controls (explicit gene-level prevalence proxies and calibration diagnostics).
- Use the same stratified shuffle null to test whether immune robustness survives additional covariates.

3. Reframe `H33` before further spend:
- Keep non-GW cycle objective only if augmented with biological anchors (e.g., trusted TF-target anchor regularization).
- Gate continuation on positive edge-transfer delta (not cycle metrics alone).
