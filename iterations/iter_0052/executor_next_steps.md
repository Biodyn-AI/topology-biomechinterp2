# Next Steps: iter_0052

1. **Promote H139 with targeted coverage recovery**
- Keep `H139` active.
- Recover missing `external_lung/dual_axis_disjoint` seed coverage by explicitly relaxing that slice's node/sample thresholds (as already done for lung dual-axis) and rerun only that hard slice.
- Gate: require at least `2/3` seed rows evaluable and non-negative mean null-gap before claiming full dual-axis robustness.

2. **Scale-check H140 across seeds (cheap extension)**
- Extend `H140` neighborhood-scaling test from seed42 to seeds `43/44` while keeping the same bounded `k={8,12,16}` setup.
- Report per-split sign consistency of `mean_delta_gain_vs_swap` across seeds.
- Gate: keep if at least `5/8` evaluable domain-splits remain gain-positive in at least two seeds.

3. **Use H141 to drive selective rescue, not broad tuning**
- Treat `H141` as a failure map: prioritize negative strict-margin slices (`external_lung/*`, `lung/source`, `lung/target`, `immune/dual`) for rescue.
- Rescue rule: only one materially changed method next (for example, stronger local charting or density-calibrated anisotropy) and evaluate on those failing slices first.

4. **Do not reopen retired branches this iteration**
- Keep correspondence-free cross-model alignment and additive ontology-sheaf hardening retired unless there is a clearly different representation or objective, with explicit rescue rationale.
