# Brainstormer Structured Feedback - iter_0021

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0021/executor_research_validation.json`).
- Implication: full hypothesis expansion is valid; no recovery-only packet is required this iteration.

## Iteration Readout (What Matters)
- `H46` is decisively negative for its stated objective: weighted zigzag did not beat unweighted coupling (`weighted>unweighted` in `0/36`; domain better count `1/3`).
- `H47` is the strongest live signal: `mean delta_AUROC=+0.00566`, positive in `24/24` rows, Fisher-significant in `6/6` domain-split summaries.
- `H48` is narrow and sparse: enrichment appears only in immune (`4/18` significant rows overall), while lung/external-lung are null with many zero-variance null rows.

## Stale Direction Triage

### `retire_now`
1. Rewiring-survival null lineage (`H07`, `H09`, `H11`, `H12`) - repeated negatives and confounded bridge diagnostics with no rescue trajectory.
2. GW-first correspondence recovery (`H27`, `H29`) - repeated mapping/transfer failures under controlled nulls.
3. Raw curvature/thinness direct-score variants (`H23`, `H30`) - repeated underperformance against geodesic/diffusion baselines.
4. Coarse support-tier monotonic concentration (`H19`, `H37`) - repeated directional failure.
5. Current weighted-threshold zigzag formulation (`H46`) - objective failure on full multi-seed packet.
6. Current OOS ID delta-R2 family (`H42`, `H45`) - unstable or contradictory holdout behavior.

### `rescue_once_with_major_change`
1. Cross-model motif branch (`H48`) only with a denser motif basis and anti-sparsity design (module-level aggregation or larger k).
2. Cross-model alignment branch (`H33`, `H36`) only if objective is utility/topology transfer, not correspondence quality.
3. Biological-support interaction branch (`H40`, `H43`) only with explicit held-out-domain utility lift gates.

## Navigation Guidance
- Treat `H47` as the default exploitation lane (robustness, ablations, transfer tests).
- Push exploration toward topology that is not simple thresholding (directed/signed filtrations, vineyard dynamics, mapper-style summaries).
- Force every new branch to include one biological anchor and one hard null that preserves local graph structure.

## Selected Immediate Candidates
- High-probability discovery: `N212` (multi-seed robustness + utility anchoring of bifiltration gains).
- High-risk/high-reward: `N213` (directed/signed persistence branch).
- Cheap broad-screen: `N224` (expanded cross-model motif fingerprint with anti-sparse design).
