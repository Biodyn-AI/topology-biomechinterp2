# Executor Next Steps — iter_0012

1. Promote map-aware cross-model transfer (`H20`) and run a stronger unsupervised rescue by replacing plain Hungarian OT with entropic Sinkhorn (or CCA-warm-start + OT), then require OT top-1 recovery and transfer AUROC gains in at least `2/3` domains.

2. Deprioritize confidence-monotonicity as currently defined (`H19`), and run only one rescue with biologically independent confidence anchors (e.g., STRING score tiers) plus prevalence-adjusted statistics; retire if direction remains non-monotonic.

3. Reframe local reconstruction mechanism (`H21`) as an inverse-coupling hypothesis by testing whether higher local reconstruction error predicts lower geodesic lift, especially in target-disjoint splits, then replicate in lung/external-lung before promotion.
