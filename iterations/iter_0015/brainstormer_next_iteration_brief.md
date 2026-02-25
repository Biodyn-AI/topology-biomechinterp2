# Brainstormer Next Iteration Brief — iter_0015 -> iter_0016

## Gate Status
- Current validation passed: `passed_min_research_gate=true`.
- Run a normal 3-hypothesis packet.

## Minimal Executable Plan for iter_0016

### 1) High-probability discovery run
- Candidate: `N127` (diffusion incremental-value under covariate adjustment).
- Objective: determine whether diffusion adds signal beyond `{degree, coexpression, euclidean, geodesic}`.
- Protocol:
1. Reuse current per-domain/per-seed/per-layer edge tables (`immune/lung/external_lung`, layers `0/3/7/11`, source+target disjoint).
2. Fit nested models: `baseline_covariates` vs `baseline_covariates + diffusion_features(t=1,2,4,8)`.
3. Compute per-row incremental AUROC, log-loss, and calibration gain.
4. Calibrate gains with stratified permutations within degree x coexpression bins.
- Required artifacts:
  - `h31_diffusion_incremental_by_seed_layer_split.csv`
  - `h31_diffusion_incremental_domain_summary.csv`
  - `h31_diffusion_incremental_null_summary.csv`
- Promotion gate:
  - Incremental gain positive and permutation-significant in `>=2/3` domains.

### 2) Cheap broad-screen run
- Candidate: `N133` (geodesic convexity-deficit/detour features).
- Objective: test a new geometric mechanism with low implementation overhead.
- Protocol:
1. Compute edge detour ratio (`shortest_path / euclidean`) and endpoint convexity-deficit features from existing kNN geodesic graphs.
2. Evaluate edge AUROC by domain/split/layer and compare to geodesic baseline.
3. Calibrate with endpoint-matched permutations (degree + geodesic-length + coexpression bins).
- Required artifacts:
  - `h32_convexity_detour_by_seed_layer_split.csv`
  - `h32_convexity_detour_domain_summary.csv`
  - `h32_convexity_detour_null_summary.csv`
- Promotion gate:
  - Above-chance AUROC in both splits for at least two domains.

### 3) High-risk/high-reward run
- Candidate: `N137` (tri-domain cycle-consistent alignment).
- Objective: replace failed GW correspondence recovery with a materially different alignment objective.
- Protocol:
1. Build CCA/spectral embeddings for each domain pair.
2. Jointly optimize maps with cycle-consistency penalty across `immune`, `lung`, `external_lung`.
3. Evaluate cycle error, distance structure transfer, and edge-transfer AUROC against pairwise-only baseline.
4. Use random cycle-consistent map controls for null calibration.
- Required artifacts:
  - `h33_cycle_consistent_alignment_domain_summary.csv`
  - `h33_cycle_consistent_alignment_null_summary.csv`
  - `h33_cycle_consistent_alignment_map_quality.csv`
- Promotion gate:
  - Cycle error reduction and transfer-AUROC gain over pairwise baseline in `>=2/3` domains.

## Execution Order
1. Run `N127` first (highest confidence branch recovery).
2. Run `N133` second (cheap orthogonal screen to keep iteration productive even if run 3 is unstable).
3. Run `N137` last (highest implementation/tuning risk).

## Scope Guardrails
1. Do not re-open GW-primary correspondence methods unless the objective is materially different from `H27/H29`.
2. Do not iterate on triangle-thinness as currently defined.
3. Every run must emit both row-level metrics and explicit null-summary artifacts.

## Contingency if a Future Gate Fails
1. Execute only `N127` on `immune` and `lung` with both disjoint splits.
2. Emit mandatory files: `executor_iteration_report.md`, `executor_hypothesis_screen.json`, and one machine summary CSV.
3. Restore full 3-run packet after validation returns to pass state.
