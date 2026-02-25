# Brainstormer Next Iteration Brief — iter_0011 -> iter_0012

## Gate Status
- Current iteration gate passed: `passed_min_research_gate=true`.
- Proceed with standard hypothesis execution (no immediate recovery patch required).

## Minimal Executable Plan for iter_0012

### 1) Primary run (high-probability discovery)
- Candidate: `N81` (confidence-stratified biological anchoring of H16).
- Objective: convert `H16` geometric-community signal into confidence-aware biological evidence.
- Protocol:
1. Join edge list with TRRUST/DoRothEA/STRING confidence tiers.
2. Recompute same-community enrichment per confidence bin, layer, split, seed.
3. Test monotonic slope and split replication.
- Required artifacts:
  - `h19_confidence_community_by_seed_layer_split_bin.csv`
  - `h19_confidence_community_layer_split_summary.csv`
  - `h19_confidence_community_monotonicity_tests.csv`
- Promotion gate: monotonic positive slope in both source and target disjoint splits.

### 2) Upside run (high-risk/high-reward)
- Candidate: `N78` (cross-model OT/Procrustes transfer).
- Objective: test model-invariant geometric structure.
- Protocol:
1. Build shared-gene alignment map (Procrustes baseline, OT variant).
2. Transfer geodesic neighborhoods/scoring between models.
3. Evaluate transferred geodesic-vs-euclidean lift and neighborhood overlap by domain.
- Required artifacts:
  - `h20_cross_model_transfer_alignment_summary.csv`
  - `h20_cross_model_transfer_by_domain_layer.csv`
  - `h20_cross_model_transfer_null_summary.csv`
- Promotion gate: transferred geodesic lift stays positive and significant above random-map controls in >=2 domains.

### 3) Cheap broad-screen run
- Candidate: `N75` (local reconstruction error screen).
- Objective: quickly test nonlinearity as mechanism for geodesic advantage.
- Protocol:
1. Compute local neighborhood reconstruction error per gene/layer/split.
2. Aggregate to edge features and fit split-aware trend tests.
3. Validate with permutation null.
- Required artifacts:
  - `h21_local_reconstruction_edge_features.csv`
  - `h21_local_reconstruction_trend_summary.csv`
- Promotion gate: positive and significant trend in both disjoint splits.

## Execution Ordering
1. Start with `N81` to secure biologically interpretable yield.
2. Run `N75` in parallel or immediately after `N81` as low-cost mechanism screen.
3. Run `N78` once alignment inputs are validated, because it is the highest-cost branch.

## Scope Guardrails
1. Do not reopen rewiring-survival sweeps (`H07/H09/H12`) unless explicitly authorized as a one-off rescue.
2. Do not run coarse-bin-only cross-model trend analyses; use per-edge or transfer-based tests.
3. Keep every experiment decision-graded with one explicit null and a pass/fail promotion rule.

## Contingency if a Future Gate Fails
- Recovery packet (single iteration, single domain, fast turnaround):
1. Execute only `N75` with permutation null and both disjoint splits.
2. Emit mandatory files: one machine CSV, `executor_hypothesis_screen.json`, `executor_iteration_report.md` with command trace.
3. Resume full portfolio only after a passing research gate is re-established.
