# Brainstormer Hypothesis Roadmap - iter_0049

## Retire / Deprioritize
1. Cross-model map-learning transfer chain (`H119/H122/H125/H131`) -> `retire_now`.
Reason: repeated null-gap collapse after major objective resets (`0/3` in current run).

2. Cross-model perturbation-concordance chain (`H108/H109`) -> `rescue_once_with_major_change`.
Constraint: only reopen with correspondence-free invariants; do not run another direct mapping utility endpoint.

3. Local path-geometry additive chain (`H120/H126/H129/H132`) -> `retire_now`.
Reason: repeated directional lift without strict-null survival; current and prior runs both at `0/6` mean null-gap support.

4. Additive graph-topology surrogate chain (`H95/H97/H128`) -> `retire_now`.
Reason: low effect size and repeated rewiring-null failure.

5. Signed motif-community hardening lineage (`H116/H118/H123/H124/H127/H130`) -> `keep_active_high_priority`.
Reason: only branch with stable directional strength and clear, narrow failure slices.

## New Hypothesis Portfolio
| ID | Area | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N661` | Topology (bifiltration PH) | Regulatory positives lie on stable ridges in a signed-support x geodesic-radius bifiltration not captured by single-axis features. | Build 2-parameter edge-local filtrations at layers `7,11`, extract ridge mass and persistence-surface moments, add over `H70`. | Positive mean null-gap in `>=3/6` domain-splits and at least one target-disjoint pass. | Axis permutation within degree bins, sign shuffle by TF strata, label permutation. | high | high |
| `N662` | Topology stability (vineyards) | True edges show lower barcode-trajectory volatility across depth than matched negatives. | Compute vineyard slope/acceleration summaries across layers `0,3,7,11`; model over `H70` with bootstrap stability filtering. | Positive mean null-gap in `>=2/6` and stable effect under bootstrap resamples. | Layer-order permutation, barcode component shuffle, label permutation. | medium | medium |
| `N663` | Topology + biology (witness complexes) | TRRUST-anchored witness complexes around TF neighborhoods recover null-robust local topology signal. | Use TRRUST TFs as landmarks to build edge-local witness complexes at layer `11`; derive Betti/lifetime descriptors and evaluate over `H70`. | Positive mean null-gap in `>=4/9` on module-slice packet. | Landmark swap within TF-degree strata, witness assignment shuffle, label permutation. | high | medium |
| `N664` | Topology (filtration variant) | Morse-style filtration on residual-energy gradients captures discriminative topological events missed by current support-based filtrations. | Construct superlevel filtrations from residual-energy gradient fields per edge neighborhood; add persistence image features over `H70`. | Positive mean delta in `>=5/6` and positive mean null-gap in `>=2/6`. | Gradient-field phase randomization, neighborhood rewiring, label permutation. | medium | medium |
| `N665` | Manifold geometry (curvature flow) | Directed Ricci-flow asymmetry along source-target geodesics separates true regulatory edges. | Compute Ollivier/Forman curvature trajectories on directed path neighborhoods; extract asymmetry and flow-stability terms over `H70`. | Positive mean null-gap in `>=2/6`, strongest in source-disjoint splits. | Path reversal, transport-plan shuffle within degree bins, label permutation. | medium | high |
| `N666` | Manifold geometry (local linearity) | Positives occupy neighborhoods with lower codimension jump between source and target tangent spaces than negatives. | Estimate local PCA codimension at path waypoints; build codimension-jump and jerk features at layers `7,11` over `H70`. | Positive mean null-gap in `>=2/6` and consistent sign across domains. | Chart-basis rotation, waypoint order shuffle, label permutation. | medium | low |
| `N667` | Intrinsic dimension (cheap) | Intrinsic-dimension phase transitions along directed paths are enriched for negative edges and improve discrimination when used as interaction terms. | Compute TWO-NN ID profiles along source-target geodesics; derive sign-flip count, hysteresis, and slope asymmetry features over `H70`. | Positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6`. | ID-profile permutation along path, endpoint swap in distance bins, label permutation. | medium | low |
| `N668` | Cross-model alignment (correspondence-free) | Cross-model agreement is detectable as distributional similarity of topological invariants without learning gene-level maps. | Build matched domain/split persistence images in scGPT and Geneformer, score sliced-Wasserstein/MMD similarity, and test whether invariant agreement predicts edge utility above random pairing. | Positive domain null-gap in `>=2/3` and immune non-negative. | Cross-model pairing shuffle, anchor-set shuffle within degree bins, barcode-lifetime permutation, label permutation. | high | high |
| `N669` | Cross-model structure transfer | Perturbation-induced homology response signatures are shared across models even when static transfer fails. | Apply matched perturbation panels to both models, compute delta-persistence fingerprints, and score concordance features for edge prediction. | Positive domain null-gap in `>=1/3` in pilot with at least one immune-null nonnegative row. | Perturbation schedule shuffle, response-time permutation, model-pair randomization, label permutation. | high | high |
| `N670` | Biological anchoring (hard-slice rescue) | Hard-slice failures in `H130` come from unmodeled ontology-barrier effects interacting with GO semantics and STRING support. | Extend `H130` with ontology barrier energy, GO semantic residuals, STRING triangle-closure terms, and interaction terms; evaluate full 3-seed x 3-domain x 3-split packet at layer `11`. | Hard slices flip to non-negative null-gap and overall positive mean null-gap in `>=4/9` domain-splits. | Conditional randomization preserving TF/target degree, GO-depth bin, ontology-barrier quantile, STRING bin; label permutation. | high | medium |
| `N671` | Biological anchoring (network motifs) | Signed feed-forward and feedback motif closure weighted by STRING confidence provides mechanistic lift beyond pairwise motif terms. | Construct edge-local signed motif closure counts (FFL, feedback loops, bi-fans) with STRING weighting; add over `H130` residuals. | Positive mean null-gap in `>=3/9`, especially `lung/dual_axis_disjoint`. | Degree-matched motif rewiring, sign shuffle preserving TF polarity rates, label permutation. | high | medium |
| `N672` | Topology + cell ontology | Cell-ontology sheaf obstruction scores identify edges crossing incompatible local regulatory contexts. | Partition neighborhoods by ontology chart, compute sheaf consistency/obstruction summaries, and test additive lift over `H130` on hard slices first. | Positive null-gap on both hard slices and at least one additional split per domain. | Ontology chart relabeling preserving sizes, section-order shuffle, label permutation. | medium | medium |
| `N673` | Algorithmic signatures (compression) | True regulatory edges have lower description length for multi-layer geometric-topological token streams than negatives. | Tokenize per-edge descriptor trajectories across layers, compute MDL/compression-gap features, and test over `H70`. | Positive mean delta in `>=4/6` with positive mean null-gap in `>=2/6`. | Token shuffle within layer bins, random codebook baseline, label permutation. | medium | low |
| `N674` | Mechanistic motifs (counterfactual edits) | Positives require larger minimal edit distance to transform their trajectory motifs into null-like automata than negatives. | Fit compact motif automata on positives, compute per-edge counterfactual transition-edit cost, and add to `H70`/`H130` on pilot slices. | Positive null-gap in `>=2/6` and strongest effect in external-lung. | State-label permutation, transition rewiring preserving out-degree, label permutation. | medium | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N670` (hard-slice ontology-barrier rescue on top of `H130`).
Why: strongest active lineage plus localized failure signatures; this is the shortest path to a null-robust gain.
Keep gate: positive mean null-gap in `>=4/9` domain-splits, `lung/dual_axis_disjoint > 0`, `immune/source_disjoint >= 0`.

2. High-risk/high-reward candidate: `N668` (correspondence-free cross-model topological invariant alignment).
Why: avoids repeatedly failed map-learning endpoints and tests a genuinely new transfer objective.
Keep gate: positive domain null-gap in `>=2/3` and immune domain non-negative.

3. Cheap broad-screen candidate: `N667` (intrinsic-dimension phase descriptor screen).
Why: low implementation cost, broad geometric coverage, high pruning value if null-negative.
Keep gate: positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6`.
