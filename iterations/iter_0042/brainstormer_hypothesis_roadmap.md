# Brainstormer Hypothesis Roadmap - iter_0042

## Retire / Deprioritize
1. Static cross-model concordance and Jacobian alignment endpoints (`H96/H99/H102/H109`) -> `retire_now`.
Evidence: repeated domain-level null-gap failure, with immune failing across all seeds in `H109`.

2. Additive filtration refinements on top of `H93` (`H94/H100/H103/H106`) -> `retire_now`.
Evidence: repeated `0/6` positive mean null-gap domain-splits.

3. Vineyard additive utility formulation (`H110/N539`) -> `retire_now`.
Evidence: mean utility effectively null and robustness `0/6`.

4. Bridge-curvature additive utility branch (`H95/H97`) -> `retire_now` for promotion claims.
Evidence: repeated directional-only signal with zero null-gap survival.

5. Standalone/additive intrinsic-dimension utility (`H89/H98` and related) -> `retire_now`.
Evidence: long negative tail with no robust keep-gate success.

6. Bio-motif grammar branch (`H107/H111`) -> `rescue_once_with_major_change`.
Required major change: occupancy+transition matched sequence nulls, multiseed replication, and explicit transition-structure diagnostics.

7. Cross-model branch -> `rescue_once_with_major_change` only for disagreement-conditioned transfer objectives (not raw concordance).

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N552` | topology (depth zigzag PH) | True regulatory edges are enriched for long-lived zigzag persistence bars across layer transitions that negatives do not sustain. | Build edge-local complexes at layers `{0,3,7,11}`, run zigzag persistence on alternating add/remove simplices, and add long-bar mass + birth-depth entropy features over `H93/H70`. | Positive mean `delta_AUROC` in `>=3/6` domain-splits and positive mean null-gap in `>=2/6`, with strongest gain in late-depth transitions. | Layer-order permutation, local-complex node relabeling within degree bins, label permutation. | high | high |
| `N553` | topology stability (landscape variance) | Positives show lower cross-seed variance of persistence landscapes under graded perturbation than negatives. | For seeds `{42,43,44}`, compute landscape curves across perturbation strengths `{0,0.25,0.5,0.75,1.0}` and use Fr\'echet variance/slope descriptors as additive features to `H93`. | Landscape-variance features reduce weak-split failures and produce positive null-gap in `>=3/6` domain-splits. | Strength-order permutation, perturbation-panel shuffle, label permutation. | medium | medium |
| `N554` | topology (bifiltration surface) | A directional-sign bifiltration surface captures regulatory asymmetry missed by one-axis weighted filtrations. | Construct two-parameter filtrations (axis1=`support sign margin`, axis2=`confidence-weighted geodesic`), summarize Hilbert surface moments and Euler characteristic transform slices. | Utility lift over `H93` with positive mean null-gap in `>=3/6`, especially in immune source-disjoint. | Axis permutation, sign-state shuffle within degree strata, label permutation. | high | high |
| `N555` | topology (local homology boundary) | Regulatory edges crossing GO-module boundaries induce characteristic local homology rank shifts around endpoints. | Define GO-boundary neighborhoods per endpoint, compute local Betti rank changes between boundary/core neighborhoods, and test interactions with `H93`. | Positive boundary-interaction coefficient and `>=3/6` positive mean null-gap domain-splits. | Module-membership shuffle preserving module size, boundary-threshold randomization, label permutation. | medium | medium |
| `N556` | manifold geometry (curvature drift) | True directed edges follow a more coherent curvature drift profile along shortest geodesic paths than negatives. | Compute Ollivier-Ricci/Forman proxies along source-target geodesics, extract drift slope/asymmetry/variance, and evaluate additive utility over `H70`. | Consistent drift-sign separation and positive null-gap in `>=3/6` domain-splits. | Endpoint swap within distance bins, neighborhood permutation preserving degree, label permutation. | medium | medium |
| `N557` | manifold geometry (holonomy) | Positives reside in neighborhoods with lower tangent-space holonomy defect around local loops. | Build triangle/quad loops around each edge, parallel-transport local PCA tangents, and score holonomy defect moments. | Lower holonomy defect in positives and positive mean null-gap in `>=2/6` domain-splits. | Loop rewiring preserving loop length distribution, tangent-basis random rotation, label permutation. | medium | high |
| `N558` | manifold geometry (local linearity acceleration) | True edges have smoother depth-wise tangent-subspace evolution (low principal-angle acceleration). | Compute principal angles between endpoint tangent spaces at layers `{0,3,7,11}`, derive drift/acceleration/jerk features, and compare to `H70` baseline. | Positive utility concentrated in late layers and positive null-gap in `>=2/6` domain-splits. | Layer-order permutation, neighborhood assignment shuffle, label permutation. | medium | low |
| `N559` | manifold geometry (ID hysteresis) | Positives show lower intrinsic-dimension hysteresis between forward and reverse neighborhood-scale sweeps. | Estimate local ID over radii `k={4,6,8,10,12,16}` in forward vs reverse sweep, compute area-between-curves and endpoint-gap features. | Positive edges have lower hysteresis; broad directional lift with positive mean delta in `>=4/6` domain-splits. | Radius-order permutation, local-neighborhood reshuffle, label permutation. | medium | low |
| `N560` | cross-model transfer (disagreement-conditioned) | Cross-model signal becomes robust when agreement features are gated by disagreement strata instead of pooled globally. | Reuse `H108/H109` perturbation descriptors, stratify module/edge items by disagreement quantile, and fit gated transfer model per domain. | Immune null-gap shifts toward zero/positive; `>=2/3` domains with positive null-gap. | Disagreement-bin permutation within size/variance strata, ungated baseline, random mapping control. | high | medium |
| `N561` | cross-model alignment (trajectory CCA) | Canonical alignment of perturbation trajectories transfers better than alignment of static role vectors. | Build trajectory tensors from perturbation panels, learn CCA components on two domains, and test held-out-domain response-rank concordance. | Higher held-out concordance than raw alignment and positive null-gap in at least one held-out test. | Trajectory time-index permutation, module-label shuffle, random gene mapping. | high | high |
| `N562` | cross-model transfer (null-separation invariance) | Transferability exists in null-separation margins even when raw utility transfer fails. | Train model to predict whether scGPT slice exceeds q95 null-gap using Geneformer-derived descriptors and evaluate leave-one-domain-out. | Margin-class AUROC `>0.60` and positive calibration gain on held-out domain. | Domain-label permutation, random-teacher baseline, label permutation. | medium | medium |
| `N563` | biological anchoring (TRRUST sign motifs) | TRRUST sign-consistent TF motifs are enriched in high-persistence/high-geometry-support edge neighborhoods. | Stratify by TRRUST sign consistency and motif class (feed-forward/feedback), add motif-conditioned persistence interactions over `H93/H70`. | Positive motif-interaction effect and `>=3/6` positive mean null-gap domain-splits. | Motif-sign permutation within TF degree bins, motif-membership shuffle, label permutation. | high | medium |
| `N564` | biological anchoring (GO/STRING cocycle) | Edges supported by GO+STRING cocycle consistency exhibit stronger topological signal than support-matched controls. | Build cocycle-consistency score from GO co-annotation + STRING edges in local 3-node motifs, test interaction with topology features. | Positive interaction and recovery of weak source-disjoint slices. | Support-matched random motif control, cocycle sign randomization, label permutation. | medium | medium |
| `N565` | algorithmic motif (semi-Markov bio-grammar) | Regulatory edges follow layerwise semi-Markov state dynamics that second-order FSM misses. | Extend `H111` with dwell-time bins, transition entropy, and forbidden-transition penalties; run across all domains/splits with multiseed check. | Keeps strong directional gain and raises positive mean null-gap from `1/6` to `>=2/6` (target `>=3/6`). | Occupancy+transition-count matched sequence shuffles, layer-order permutation, label permutation. | high | medium |
| `N566` | algorithmic mechanism (perturbation automaton) | Under controlled perturbations, positives exhibit lower automaton entropy growth and fewer irreversible state cycles. | Convert perturbation-response trajectories into discrete automaton states per edge/module and model entropy-rate + cycle-count descriptors over `H70/H93`. | Positive utility and positive mean null-gap in `>=2/6` domain-splits. | Perturbation-schedule permutation, trajectory phase randomization, label permutation. | medium | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N565` (semi-Markov bio-grammar rescue).
Why: `H111` already shows strong directional lift (`6/6` positive deltas), so redesigning nulls/features is the shortest path to a robust claim.
Execution gate: positive mean null-gap in `>=2/6` domain-splits (stretch `>=3/6`) and no domain with strongly negative collapse (`< -0.02`).
Fail-fast: retire if positive mean delta drops below `4/6` domain-splits.

2. High-risk/high-reward candidate: `N552` (depth zigzag persistence).
Why: introduces a genuinely new topological object not exhausted by prior additive PH variants.
Execution gate: positive mean null-gap in `>=2/6` domain-splits with at least one immune/lung split positive.
Fail-fast: if utility mean is near zero and null-gap positive is `0/6`, retire immediately.

3. Cheap broad-screen candidate: `N559` (ID hysteresis).
Why: low implementation cost using existing embeddings and neighborhood machinery; orthogonal manifold signal that has not been screened in hysteresis form.
Execution gate: positive mean delta in `>=4/6` domain-splits and at least `1/6` positive mean null-gap.
Fail-fast: if positive mean delta is `<=2/6`, stop lineage.
