# Brainstormer Next Iteration Brief — iter_0012 -> iter_0013

## Gate Status
- Current executor validation passed: `passed_min_research_gate=true`.
- Proceed with normal execution; no emergency recovery patch is required.

## Minimal Executable Plan for iter_0013

### 1) Primary run (high-probability discovery)
- Candidate: `N95`.
- Objective: rescue biological anchoring with a materially improved test.
- Protocol:
1. Build a unified edge-prior table from DoRothEA + TRRUST + STRING confidence and agreement status.
2. Recompute community/geometry effects (`same-community`, geodesic gain, curvature bins) with prevalence- and degree-adjusted mixed models.
3. Evaluate by split and layer with permutation-calibrated p-values.
- Required artifacts:
  - `h22_bio_anchor_adjusted_edge_table.csv`
  - `h22_bio_anchor_adjusted_model_summary.csv`
  - `h22_bio_anchor_adjusted_permutation_null.csv`
- Promotion gate: adjusted effect remains positive in both disjoint splits (`p < 0.05`).

### 2) Upside run (high-risk/high-reward)
- Candidate: `N92`.
- Objective: make unsupervised cross-model alignment non-trivial.
- Protocol:
1. Learn shared latent space (CCA/PCA) for matched genes.
2. Run Sinkhorn OT in latent space and compare against Hungarian OT and Procrustes baselines.
3. Score map quality (top-1/cycle-consistency) and transfer quality (Jaccard/AUROC) per domain.
- Required artifacts:
  - `h23_unsupervised_alignment_domain_summary.csv`
  - `h23_unsupervised_alignment_null_summary.csv`
  - `h23_unsupervised_alignment_map_quality.csv`
- Promotion gate: OT map-quality and transfer-quality both beat Hungarian + random nulls in `>=2/3` domains.

### 3) Cheap broad-screen run
- Candidate: `N90`.
- Objective: test whether H21 split asymmetry is a reproducible depth-phase phenomenon.
- Protocol:
1. Compute layerwise local-linearity/reconstruction metrics in immune, lung, and external-lung.
2. Fit split x layer-band interaction models (early/mid/late) against edge-label AUROC and geodesic lift.
3. Validate with layer-order and split-label permutations.
- Required artifacts:
  - `h24_phase_transition_by_seed_layer_split.csv`
  - `h24_phase_transition_model_summary.csv`
  - `h24_phase_transition_null_summary.csv`
- Promotion gate: late-layer target interaction is negative and significant in at least two domains.

## Execution Order
1. Run `N95` first (highest expected discovery yield and directly resolves current H19 failure mode).
2. Run `N90` next (cheap discriminator that can immediately reframe H21 strategy).
3. Run `N92` last (highest complexity and runtime; strongest upside if successful).

## Scope Guardrails
1. Do not rerun rewiring-survival (`H07/H09/H12`) or distortion rescue unless explicitly re-authorized.
2. Do not run plain Hungarian OT again as a standalone unsupervised claim.
3. Require two-gate evaluation for unsupervised alignment: map quality and biological transfer.

## Contingency If a Future Gate Fails
1. Execute only `N90` in one domain with both splits and one permutation null.
2. Emit mandatory files (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, one machine CSV).
3. Resume full portfolio only after gate validity is restored.
