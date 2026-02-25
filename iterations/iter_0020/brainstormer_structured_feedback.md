# Brainstormer Structured Feedback - iter_0020

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0020/executor_research_validation.json`).
- Implication: full hypothesis expansion is valid; no recovery-only fallback packet is required for this pass.

## Iteration Readout (What Matters)
- `H43` (`module_structure`) is directionally strong for biological interaction structure (mean interaction coefficient `+0.23253`, Fisher-significant `4/6` domain-splits) but has near-zero predictive lift (mean AUROC delta `-1.84e-05`).
- `H44` (`topology_stability`) is the strongest structural signal this iteration (true zigzag positive in `12/12` rows; domain Fisher `3/3`) but currently capped by low permutation resolution (`40` nulls; p-value floor `0.02439`).
- `H45` (`intrinsic_dimensionality`) remains non-promotable (winsorized positive but bootstrap-failed; trimmed unstable/negative).

## Stale Direction Triage

### `retire_now`
1. Rewiring-survival null lineage (`H07`, `H09`, `H12`; plus bridge explanation `H11`) - repeated negatives/inconclusive across multiple redesigns with low rescue upside.
2. GW-first correspondence recovery (`H27`, `H29`) - repeated failure to recover useful mappings or transfer gains.
3. Raw curvature/thinness direct-score variants (`H23`, `H30`) - repeated underperformance versus geodesic/diffusion baselines.
4. Coarse tiered concentration variants (`H19`, `H37`) - repeated directional failure or zero significance despite multiple tries.
5. Current `H45` metric form (winsorized/trimmed delta-R2 with tiny holdouts) - unstable and not decision-grade.

### `rescue_once_with_major_change`
1. Cross-model alignment (`H33`, `H36`) only if objective is changed to utility-first or topology-signature transfer, not correspondence quality.
2. Intrinsic-dimension mechanism branch only if target is changed from unstable delta-R2 to rank/ordering or monotone utility coupling with larger holdout blocks.
3. H43 branch only if next runs enforce strict held-out-domain utility gates (otherwise it stalls as interpretive-only signal).

## Navigation Guidance for Next Loop
- Push toward joint topology x biology hypotheses that can directly predict utility, not standalone descriptive topology.
- Use multiseed and held-out-domain checks by default for promotable claims.
- Force nulls that preserve structure (degree, support strata, layer effects) to avoid both trivial and over-adversarial controls.

## Selected Immediate Candidates
- High-probability discovery: `N198` support-weighted zigzag utility coupling.
- High-risk/high-reward: `N200` two-parameter (distance x support) bifiltration topology.
- Cheap broad-screen: `N208` cross-model motif-overlap concordance.
