# Executor Next Steps - iter_0035

1. Promote `H88` to one targeted robustness pass focused on `immune/source_disjoint` with higher null resolution and stability-regularized sparsity, keeping the gate at `>=5/6` positive mean null-gap domain-splits and adding a descriptor-core stability target (`Jaccard >= 0.6`).
2. Retire the tested `H89` phase-boundary utility formulation (`0/6` positive mean null-gap domain-splits); only reuse these descriptors as diagnostics inside stronger backbone models.
3. Retire the tested `H90` additive stability formulation (`0/6` positive mean null-gap domain-splits); if reused, shift to perturbation stress-testing of already-positive hypotheses instead of direct predictive lift.
4. Keep next packet breadth-first with at most one `H88` carry-over slot plus two materially novel slots outside retired standalone intrinsic-dimension and additive-stability utility formulations.
