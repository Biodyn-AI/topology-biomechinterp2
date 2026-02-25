# Executor Next Steps — iter_0015

1. Promote one refinement of `H28` with a harder incremental-value test:
- Fit edge models with coexpression + degree covariates and test whether diffusion scores add AUROC/calibration beyond those covariates under source/target-disjoint splits.
- Keep permutation calibration, but switch the target metric from raw AUROC deltas to incremental AUROC/log-loss.

2. Retire GW-as-primary correspondence recovery direction (`H27` + `H29`) unless method changes materially beyond GW:
- Use CCA/spectral alignment as the new baseline family for cross-model mapping.
- Only revisit GW with explicit new mechanism (e.g., multi-domain cycle consistency or supervised anchors).

3. Replace failed thinness variant with a different topology mechanism next iteration:
- Run one of `N113` (diffusion-time persistence vineyards) or `N125` (module-restricted diffusion gains) with explicit nulls.
- Keep cheap-screen budget but require at least two seeds if initial signal appears positive in one split.
