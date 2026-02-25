# Next Steps: iter_0041

1. Promote `H108` to a robustness follow-up packet:
- run seeds `42/43/44` and test split-aware module partitions,
- keep the same perturbation-response endpoint and null package,
- require positive `null_gap_q95_response_concordance` in `>=2/3` domains for at least `2/3` seeds.

2. Retire `H106` (`N538`) in current form:
- fail-fast criterion already triggered (`0/6` positive mean null-gap domain-splits),
- do not re-run without a materially changed filtration objective.

3. Keep `H107` as a single rescue candidate only if method changes materially:
- current formulation has directional gain but `0/6` positive null-gap domain-splits,
- a valid rescue must change either state construction (non-quantile, biologically anchored) or null design to avoid transition-shuffle dominance.
