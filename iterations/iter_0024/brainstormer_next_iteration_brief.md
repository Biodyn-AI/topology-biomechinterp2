# Brainstormer Next Iteration Brief - iter_0024

## Objective
Exploit the strongest live signal (`H55`) with a biologically anchored rescue of its two failure slices, while opening one new cross-model topology-transfer path and one cheap geometric broad-screen.

## Gate Note
- Current gate status is valid (`passed_min_research_gate=true`), so execute a full 3-slot packet.

## Required Execution Packet (iter_0025)

### Slot A - High-probability discovery (`N255` -> proposed `H58`)
- Hypothesis: biologically weighted directed/signed topology removes the `source_disjoint` failure slices without eroding global gains.
- Scope: `3 domains x 3 seeds x 2 splits x layers {7,11}` (match `H55` footprint first; expand if gate passes).
- Core method:
  1. Start from `H55` directed/signed scores.
  2. Add multiplicative/support-aware weights from STRING, TRRUST, GO coherence.
  3. Compare weighted vs unweighted directed/signed against distance-only baseline.
- Required outputs:
  - `iterations/iter_0025/h58_weighted_directed_signed_by_seed_layer_split.csv`
  - `iterations/iter_0025/h58_weighted_directed_signed_domain_summary.csv`
  - `iterations/iter_0025/h58_weighted_directed_signed_null_summary.csv`
  - `iterations/iter_0025/h58_weighted_directed_signed_failure_slice_summary.csv`
- Null/control:
  - Biological-weight shuffle within degree/coexpression bins.
  - Random stratum assignment placebo.
  - Existing directed/signed null family carried forward.

### Slot B - High-risk/high-reward (`N263` -> proposed `H59`)
- Hypothesis: cross-model transfer works when alignment is performed on topology signatures (persistence embeddings), not raw coordinates or explicit correspondences.
- Scope: seed42 pilot on `immune/lung/external_lung`, both split regimes, layers `{7,11}`; expand seeds only if pilot passes.
- Core method:
  1. Build persistence-signature vectors per gene/module from scGPT and Geneformer.
  2. Learn alignment in signature space (CCA/Procrustes).
  3. Evaluate transferred edge priors on held-out domains.
- Required outputs:
  - `iterations/iter_0025/h59_cross_model_topology_signature_transfer_by_domain_layer.csv`
  - `iterations/iter_0025/h59_cross_model_topology_signature_transfer_summary.csv`
  - `iterations/iter_0025/h59_cross_model_topology_signature_transfer_null_summary.csv`
- Null/control:
  - Random-map alignment.
  - Within-model layer shuffle.
  - Signature-destroying permutation control.

### Slot C - Cheap broad-screen (`N262` -> proposed `H60`)
- Hypothesis: endpoint intrinsic-dimension jump is a low-cost, broad predictor of regulatory edges and can complement geodesic/topological scores.
- Scope: `3 domains x 3 seeds x 2 splits x layers {0,3,7,11}`.
- Core method:
  1. Estimate multiscale local ID (TWO-NN + MLE).
  2. Derive endpoint ID-jump and short-path ID-gradient features.
  3. Test incremental value over geodesic baseline and over `H55` family features where available.
- Required outputs:
  - `iterations/iter_0025/h60_id_jump_by_seed_layer_split.csv`
  - `iterations/iter_0025/h60_id_jump_domain_summary.csv`
  - `iterations/iter_0025/h60_id_jump_null_summary.csv`
- Null/control:
  - Endpoint swap within degree bins.
  - Estimator-randomization placebo.
  - Label permutation.

## Promotion / Continuation Gates
1. `H58` promote gate:
   - `lung/source_disjoint` and `external_lung/source_disjoint` mean `delta_AUROC >= 0`.
   - Global `H55`-style signal remains strong (`>=4/6` domain-splits Fisher-significant).
2. `H59` continue gate:
   - Transfer utility beats random-map null in `>=2/3` domains with significant aggregate evidence.
3. `H60` keep gate:
   - Positive incremental value in `>=4/6` domain-splits and at least one significant domain aggregate.

## Stop / Retire Rules
1. If `H58` fails to improve either failure slice, do not run another weighting tweak; pivot to filtration redesign (`N256`) directly.
2. If `H59` fails its pilot gate, retire cross-model transfer attempts based on alignment-only objectives for at least one full cycle.
3. If `H60` is negative across most domain-splits, retire absolute-ID and ID-jump edge endpoints and focus on non-ID geometry.

## Minimal Recovery Plan (Only if next executor gate unexpectedly fails)
1. Run a reduced `H58` on seed42 with layers `{7,11}` across all domains to produce one valid topology artifact quickly.
2. Run reduced `H60` on seed42 layer `11` across all domains for a second independent result.
3. Emit required report triad (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, `executor_research_validation.json`) before any expansion.
