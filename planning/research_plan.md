# Research Plan: Geometric and Topological Hypothesis Screening

## Objective
Identify whether scGPT/Geneformer residual representations contain robust geometric/topological structures linked to biological knowledge (TRRUST/GO/STRING/perturbation) beyond trivial baselines.

## Core strategy
Run bounded, high-throughput hypothesis screens across diverse method families with strict controls and rapid iteration.

## Initial hypothesis slate
1. Persistent-homology signal in gene embedding neighborhoods predicts regulatory proximity better than shuffled controls.
2. Geodesic distances on embedding kNN graphs align with TRRUST/STRING better than Euclidean distances.
3. Local intrinsic dimensionality shifts by layer and tracks known biological signal peaks.
4. Cross-model manifold alignment (scGPT vs Geneformer) identifies shared biologically meaningful subspaces.
5. Topological/community features provide incremental predictive value over gene-level baselines.

## Iteration protocol
1. Select 1 primary + 1 backup hypothesis.
2. Execute bounded experiment with explicit command trace.
3. Produce machine artifacts and update hypothesis screen JSON.
4. Update paper section for current iteration and compile PDF.
5. Brainstormer proposes next high-value directions and failure-contingency experiments.

## Success criteria
- At least one hypothesis shows reproducible positive effect with controls.
- Evidence remains directional across seeds/splits.
- Biological anchor is explicit and artifact-backed.

## Stop criteria for a hypothesis family
- Two to three consecutive negative/inconclusive runs with controls and no practical improvement path.
