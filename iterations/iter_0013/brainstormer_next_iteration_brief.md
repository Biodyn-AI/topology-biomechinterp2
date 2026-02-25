# Brainstormer Next Iteration Brief — iter_0013 -> iter_0014

## Gate Status
- Current validation passed: `passed_min_research_gate=true`.
- Proceed with normal execution packet.

## Minimal Executable Plan for iter_0014

### 1) Cheap broad-screen first (stabilize throughput)
- Candidate: `N105` (diffusion-distance sweep).
- Objective: quickly test whether another manifold metric beats both Euclidean and shortest-path geodesic across domains.
- Protocol:
1. For each domain and split, compute diffusion distances over a small time grid (for example `t={1,2,4,8}`).
2. Compute edge-level AUROC per layer and compare against existing Euclidean/geodesic baselines.
3. Use label permutations for per-layer significance.
- Required artifacts:
  - `h25_diffusion_distance_by_seed_layer_split.csv`
  - `h25_diffusion_distance_domain_summary.csv`
  - `h25_diffusion_distance_null_summary.csv`
- Promotion gate: at least one diffusion-time band outperforms both baselines in `>=2/3` domains.

### 2) High-probability discovery run
- Candidate: `N110` (multi-prior biological anchoring).
- Objective: biologically anchor the promoted geometry/alignment signals.
- Protocol:
1. Build an edge table with TRRUST support, STRING confidence, GO co-membership, and geometry/alignment features.
2. Fit split-aware mixed-effects models with interaction terms (`geometry x prior_support`).
3. Calibrate significance using prior-label permutations and degree-stratified bootstrap.
- Required artifacts:
  - `h26_bio_anchor_edge_table.csv`
  - `h26_bio_anchor_model_summary.csv`
  - `h26_bio_anchor_permutation_null.csv`
- Promotion gate: positive and significant interaction terms in both splits (`p < 0.05`, permutation-calibrated).

### 3) High-risk/high-reward run
- Candidate: `N108` (unseeded Gromov-Wasserstein alignment).
- Objective: test correspondence-free cross-model structure recovery.
- Protocol:
1. Build per-domain scGPT/Geneformer kNN graphs on matched gene sets.
2. Run unseeded GW alignment with a fixed hyperparameter grid.
3. Evaluate map quality (top-1/cycle consistency) and downstream transfer metrics (distance Spearman, kNN Jaccard, edge-transfer AUROC).
- Required artifacts:
  - `h27_gw_alignment_domain_summary.csv`
  - `h27_gw_alignment_null_summary.csv`
  - `h27_gw_alignment_map_quality.csv`
- Promotion gate: beat random graph correspondence controls in at least one domain, with directional consistency in `>=2/3` domains.

## Execution Order
1. Run `N105` first to guarantee an early valid machine artifact and gate safety.
2. Run `N110` second as the primary discovery packet.
3. Run `N108` last due complexity/runtime risk.

## Scope Guardrails
1. Do not rerun retired branches (`H07/H09/H12`, distortion rescue, raw Forman `H23`, plain Hungarian OT).
2. Any rescue of `H19/H22/H15/H11` must use materially changed design and controls.
3. Keep per-hypothesis null/control artifacts mandatory; no metric-only runs.

## Contingency if a Future Gate Fails
1. Execute only `N105` in one domain with both splits and one permutation control.
2. Emit mandatory files (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, one machine CSV summary).
3. Resume full 3-run packet after gate validity is restored.
