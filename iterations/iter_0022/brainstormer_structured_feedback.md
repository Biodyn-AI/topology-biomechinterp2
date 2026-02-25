# Brainstormer Structured Feedback - iter_0022

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0022/executor_research_validation.json`).
- Implication: full next-iteration hypothesis expansion is valid; no recovery-only packet is required.

## Iteration Readout (What Matters)
- `H50` is the strongest active lead: mean `delta_AUROC=+0.01585`, positive in `11/12` rows, Fisher-significant domain-splits `6/6`.
- `H49` is robust for discrimination but not mechanism-linked yet: mean `delta_AUROC=+0.00599` with Fisher-significant `6/6`, but utility coupling is weak or negative (`spearman`: external-lung `+0.258`, immune `-0.371`, lung `-0.401`; placebo significant `0/3`).
- `H51` remains non-promotable in current form: degree-null enrichment broadened, but module-shuffle control failed (`0/18`) and gene-level null variance is often degenerate in lung/external-lung rows.

## Stale Direction Triage

### `retire_now`
1. Rewiring-survival lineage (`H05-H12`) - repeated negatives under multiple stronger-null redesigns.
2. GW-first correspondence recovery (`H27`, `H29`) - repeated failure on both mapping quality and transfer utility.
3. Raw curvature/thinness direct-score variants (`H23`, `H30`) - repeated underperformance against geodesic/diffusion baselines.
4. Coarse tier/monotonic concentration variants (`H19`, `H37`) - repeated directional failure.
5. Weighted-zigzag coupling objective (`H46`) - explicit objective failure (`0/36` weighted wins).
6. Current motif-overlap objective (`H48`, `H51`) as a descriptive endpoint - repeated mixed outcomes without robust module-level attribution.
7. OOS `delta_R2` ID formulation (`H42`, `H45`) - unstable and contradictory holdout behavior.

### `rescue_once_with_major_change`
1. `H49` utility-coupling claim - rescue only with held-out utility target redesign (not layer-order correlation).
2. Cross-model alignment (`H33`, `H36`) - rescue only if objective is topology/utility transfer, not correspondence quality.
3. Curvature family - rescue only as interaction features (curvature x support/geodesic), not standalone scores.

## Navigation Guidance
- Exploit `H50` immediately with multiseed replication and stronger directed nulls before opening many new branches.
- Shift cross-model work from motif-count enrichment to topology-signature transfer (persistence images, path-homology signatures, tangent-space transfer).
- For every new direction, require both:
  - one topology/geometric stress null that preserves local graph statistics, and
  - one biological-anchor falsification control (TRRUST/GO/STRING/cell ontology shuffle).

## Selected Immediate Candidates
- High-probability discovery: `N226` (multiseed directed/signed topology replication + hard nulls).
- High-risk/high-reward: `N230` (directed path-homology signatures and cross-model transfer utility).
- Cheap broad-screen: `N233` (local linearity rupture index + ID-disagreement edge screen).
