# Executor Next Steps — iter_0014

1. Promote `H25` with stricter robustness controls.
- Run diffusion sweep on dual-axis disjoint splits where feasible and add coexpression-matched label-shuffle null.
- Gate: retain positive domain-level mean delta in `>=2/3` domains with combined Fisher `p<0.05` after new null.

2. Rescue `H27` with constrained correspondence-aware transport.
- Replace plain unseeded GW with CCA-seeded GW / one-to-one regularized transport and report unique-match rate in addition to top-1.
- Gate: top-1 retrieval significantly above random in at least one domain and edge-transfer AUROC significant in `>=2/3` domains.

3. Re-test `H26` with independent external priors.
- Add STRING edge support (once local table is available) and rerun geometry x multi-prior interaction with the same degree-stratified permutation/bootstrap protocol.
- Gate: positive interaction coefficient with `p<0.05` in both disjoint splits for at least one domain.
