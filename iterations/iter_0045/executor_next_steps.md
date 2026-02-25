# Executor Next Steps: iter_0045

1. Promote `H118` to a stricter robustness pass with higher null budgets (`>=64`), and add one additional control that randomizes TRRUST sign while preserving source TF identity and target degree.
2. Re-test `H118` on an orthogonal split regime (`dual_axis_disjoint`) to verify that the signed motif-community effect is not split-specific.
3. Retire the `H119` disagreement-gated transfer endpoint for now (near-zero utility and only `1/6` positive mean null-gap domain-splits); only revisit with a materially different target objective.
4. Keep `H120` active as a neutral branch and run one inexpensive rescue: interaction-only curvature features (`curvature x support_margin`) with higher permutation counts before promotion.
