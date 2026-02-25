# Brainstormer Next Iteration Brief — iter_0017 -> iter_0018

## Gate Status
- Current validation passed: `passed_min_research_gate=true`.
- Proceed with a normal 3-hypothesis execution packet.

## Minimal Executable Plan for iter_0018

### 1) High-probability discovery run
- Candidate: `N165` (biological consensus-tier interaction for H34 geometry features).
- Objective: determine whether small pooled `H34` gains are concentrated in biologically plausible edge subsets.
- Protocol:
1. Build a 0-3 consensus tier per edge (TRRUST, STRING, GO co-membership).
2. Fit nested models per domain/split/layer/seed:
   - baseline: `geodesic+diffusion+detour+convexity`
   - extended: baseline + `tier` + `detour*tier` + `convexity*tier`
3. Report per-row and per-domain interaction coefficients, AUROC delta, log-loss delta.
4. Calibrate with tier permutations within degree x coexpression x geodesic strata.
- Required artifacts:
  - `h37_consensus_tier_geometry_by_seed_layer_split.csv`
  - `h37_consensus_tier_geometry_domain_summary.csv`
  - `h37_consensus_tier_geometry_null_summary.csv`
- Promotion gate:
  - Positive significant interaction in `>=2/3` domains and stronger uplift in highest tier vs lowest tier.

### 2) Cheap broad-screen run
- Candidate: `N160` (intrinsic-dimension variance/skew mechanism screen).
- Objective: test whether `H35` asymmetry is distributional (variance/skew) rather than mean-breakpoint only.
- Protocol:
1. For each domain/split/layer/seed, compute TWO-NN and participation-ratio distribution moments (mean/variance/skew).
2. Regress edge AUROC and breakpoint shift on these moments with split interaction terms.
3. Use layer-order permutation and estimator-swap controls.
- Required artifacts:
  - `h38_id_distribution_moments_by_seed_layer_split.csv`
  - `h38_id_distribution_moments_domain_summary.csv`
  - `h38_id_distribution_moments_null_summary.csv`
- Promotion gate:
  - External-lung replicate plus at least one additional domain showing consistent directional signal.

### 3) High-risk/high-reward run
- Candidate: `N162` (anchor-causality stress test for H36).
- Objective: decide if anchor regularization is causal for transfer gains or only incidental to objective design.
- Protocol:
1. Freeze the `H36` lambda selection/eval split.
2. Run matched anchors, anchor-dropout levels (25/50/75%), and anchor-mismatch controls (wrong-domain/wrong-TF anchors).
3. Compare target AUROC and delta-vs-baseline across controls.
4. Keep label-permutation null for sanity, add random-anchor same-size baseline.
- Required artifacts:
  - `h39_anchor_causality_dropout_mismatch_domain_summary.csv`
  - `h39_anchor_causality_dropout_mismatch_curve.csv`
  - `h39_anchor_causality_dropout_mismatch_null_summary.csv`
- Promotion gate:
  - Clear monotonic degradation under dropout/mismatch in all domains.

## Execution Order
1. `N165` first (highest expected discovery probability, low cost).
2. `N160` second (cheap mechanism discriminator).
3. `N162` third (highest upside and decisive triage value).

## Scope Guardrails
1. Do not reopen GW-primary mapping variants.
2. Do not run another pooled `H34`/`H31` rerun without biological stratification.
3. If `N162` remains anchor-invariant, retire anchor-causality claim and keep only utility-oriented alignment metrics.

## If a Future Gate Fails (Recovery Packet)
1. Run seed42-only `N165` on immune+lung with both splits and layers `0/3/7/11`.
2. Run seed42-only `N160` with reduced permutations (`<=100`) for rapid validity.
3. Emit mandatory narrative artifacts plus one machine summary JSON and resume full packet after gate recovery.
