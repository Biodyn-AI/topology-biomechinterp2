# Executor Next Steps: iter_0042

1. Retire the tested `N546` rescue endpoint in its current form (multi-seed Jacobian-alignment variant), because immune null-gap remained negative across all seeds and seed-level gate did not clear.
2. Retire the tested `N539` vineyards additive utility formulation (`0/6` positive mean null-gap domain-splits).
3. Keep one rescue-once slot for `N551` only with materially changed controls (larger null budget plus state-occupancy/transition-matched shuffles) and require positive mean null-gap in at least `2/6` domain-splits for promotion.
4. Rotate remaining budget to a different non-retired family in the next iteration (for example `module_structure` or `split_robustness`) to avoid over-investing in repeatedly non-robust topology derivatives.
