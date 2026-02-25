# Brainstormer Next Iteration Brief — iter_0014 -> iter_0015

## Gate Status
- Current validation passed: `passed_min_research_gate=true`.
- Run a normal 3-hypothesis packet.

## Minimal Executable Plan for iter_0015

### 1) High-probability discovery run
- Candidate: `N116` (diffusion uplift under coexpression-matched stronger null).
- Objective: confirm that `H25` survives a materially stronger, biology-aware null.
- Protocol:
1. Recompute diffusion-vs-baseline deltas for domains `immune/lung/external_lung`, layers `0/3/7/11`, both disjoint splits, and seeds `42/43/44`.
2. Use coexpression-bin- and degree-matched label permutations (plus current random-label null for reference).
3. Summarize per-row and per-domain significance (upper-tail p-values and Fisher combine).
- Required artifacts:
  - `h28_diffusion_coexp_by_seed_layer_split.csv`
  - `h28_diffusion_coexp_domain_summary.csv`
  - `h28_diffusion_coexp_null_summary.csv`
- Promotion gate:
  - Positive mean delta in `>=2/3` domains and Fisher `p<0.05` under coexpression-matched null.

### 2) High-risk/high-reward run
- Candidate: `N121` (CCA-seeded, one-to-one-regularized GW).
- Objective: rescue correspondence recovery for cross-model mapping.
- Protocol:
1. Use `H24` CCA latent alignment as initialization.
2. Run GW with one-to-one regularization (or projection-to-assignment post-step) and annealed entropy.
3. Evaluate map quality (`top1`, `mutual_top1`, unique-match rate), geometry transfer, and edge-transfer AUROC.
4. Compare against unseeded GW (`H27`) and identity mapping baselines.
- Required artifacts:
  - `h29_seeded_gw_domain_summary.csv`
  - `h29_seeded_gw_null_summary.csv`
  - `h29_seeded_gw_map_quality.csv`
- Promotion gate:
  - Significant map recovery above random in `>=2/3` domains and transfer AUROC significance in `>=2/3` domains.

### 3) Cheap broad-screen run
- Candidate: `N118` (triangle thinness / hyperbolicity edge screen).
- Objective: test a low-cost geometric mechanism orthogonal to distance metrics.
- Protocol:
1. Compute edge-local geodesic triangle thinness features around endpoints per domain/split/layer.
2. Score edge discrimination AUROC and calibrate with endpoint-matched permutations.
3. Compare with diffusion/geodesic baseline layers to detect complementarity.
- Required artifacts:
  - `h30_hyperbolicity_by_seed_layer_split.csv`
  - `h30_hyperbolicity_domain_summary.csv`
  - `h30_hyperbolicity_null_summary.csv`
- Promotion gate:
  - Above-chance AUROC in both splits for at least two domains.

## Execution Order
1. Run `N116` first (highest promotion probability).
2. Run `N118` second (cheap orthogonal screen; keeps throughput if `N121` stalls).
3. Run `N121` last (highest compute and tuning risk).

## Scope Guardrails
1. Do not rerun retired methods without a materially changed definition (`H07/H09/H12`, distortion rescue, raw Forman `H23`, plain unseeded OT/GW baseline as primary method).
2. Any biological interaction claim must include independent priors (include STRING when available) and leakage-safe feature sets.
3. Every hypothesis must output both raw-by-row metrics and explicit null summaries.

## Contingency if a Future Gate Fails
1. Execute only `N116` on one domain and both splits with one stronger-null control.
2. Emit mandatory files (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, and one machine summary CSV).
3. Restore full 3-run packet after validation returns to pass state.
