# Brainstormer Hypothesis Roadmap - iter_0045

## Retire / Deprioritize
1. Disagreement-gated cross-model transfer utility (`H119` endpoint family) -> `retire_now`.
Evidence: `mean delta=+0.00060`, positive mean null-gap in `1/6` domain-splits.

2. Repeated additive PH rescue variants (`H100/H101/H103/H106/H110/H113` style) -> `retire_now`.
Evidence: multiple consecutive near-zero/negative robustness outcomes under varied null packages.

3. Coarse tokenized grammar family (`H104/H107/H111/H112`) -> `rescue_once_with_major_change`.
Major change requirement: biologically anchored state alphabet + occupancy-matched nulls + dwell-time features.

4. Standalone/additive intrinsic-dimension utilities (`H98/H114` style) -> `retire_now`.
Evidence: persistent failure to clear null-gap gates.

5. Tangent-acceleration formulation (`H115`) -> `retire_now`.
Evidence: negative direction in the tested formulation.

6. Geodesic curvature drift (`H120`) -> `rescue_once_with_major_change`.
Major change requirement: explicit direction-asymmetry modeling (source-to-target vs reverse), not pooled curvature summaries.

7. Signed motif-community (`H118`) -> `prioritize_hardening`.
Reason: strongest current signal with partial null survival; this is the best near-term discovery route.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N600` | module_structure + biological anchoring | The signed TRRUST motif-community effect remains robust under stricter nulls and orthogonal split stress. | Re-run `H118` with seeds `{42,43,44}`, splits `{source_disjoint,target_disjoint,dual_axis_disjoint}`, null budget `>=64`, plus TF-identity-preserving sign randomization and motif-decoy controls. | Positive mean null-gap in `>=5/9` domain-splits with direction positive in `9/9`. | Label permutation, TF-identity sign shuffle, motif-presence decoy shuffle matched on TF degree and target degree. | high | medium |
| `N601` | topology (multiparameter PH) | True regulatory edges occupy distinct signed-support x geodesic-distance bifiltration structure. | Build two-parameter filtrations per edge-neighborhood, extract Hilbert-surface summaries and rank-invariant slices, then test additive utility over `H70`. | Positive mean null-gap in `>=4/9` domain-splits and better split symmetry than one-parameter PH features. | Axis permutation within degree bins, bifiltration slice shuffle, label permutation. | high | high |
| `N602` | topology (directed path homology) | Directed/signed path-homology signatures around TF→target neighborhoods distinguish true edges. | Construct directed local complexes (layers `{7,11}`), compute path-homology `H1/H2` descriptors and directed cycle ratios, evaluate vs `H70`. | Positive mean null-gap in `>=3/6` domain-splits with strongest gain in source-disjoint settings. | Direction shuffle preserving in/out degree, sign shuffle, label permutation. | high | high |
| `N603` | topology (stability envelope) | Positive-edge topology is more stable to witness-complex sparsification than negatives. | Compute PH features over witness complexes across landmark fractions `{0.1,0.2,0.3,0.4,0.5}` and use stability slopes/curvature as features. | Positives show flatter degradation slope and positive mean null-gap in `>=3/6`. | Landmark resampling controls, fraction-order permutation, label permutation. | medium | medium |
| `N604` | topology (cohomology localization) | Long-lived cocycles localize near biologically supported motif modules around real edges. | Compute representative cocycles for top persistent classes; quantify overlap with TRRUST/STRING module neighborhoods and add overlap interactions to `H70`. | Positive overlap interaction and positive mean null-gap in `>=3/6`. | Module label shuffle preserving size/degree, cocycle support shuffle, label permutation. | medium | high |
| `N605` | manifold geometry (directional geodesics) | Source→target versus target→source geodesic asymmetry explains the split-specific robustness pattern seen in `H120`. | Reuse path pipeline to compute directional asymmetry features (curvature, path length, drift) per edge at layers `{7,11}` and run a cheap broad screen. | Source-disjoint null-gap improves from mostly negative to mixed/positive, with positive mean null-gap in `>=2/6`. | Direction-flip control within path-length bins, endpoint swap control, label permutation. | high | low |
| `N606` | manifold geometry (multi-scale curvature) | True edges show scale-stable curvature sign and lower curvature volatility across neighborhood radii. | Estimate Ollivier/Forman curvature at radii `{10,20,30}` and use sign consistency, slope, and variance features. | Positive mean null-gap in `>=3/6` and consistent sign trend in at least two domains. | Neighborhood membership shuffle within degree bins, radius-order permutation, label permutation. | medium | medium |
| `N607` | manifold geometry (local linearity defect) | True edges traverse locally more linear segments than negatives despite global nonlinearity. | Fit local linear reconstructions along edge geodesics, extract residual mean/anisotropy/change-rate, and test over `H70`. | Lower defect for positives and positive mean null-gap in `>=3/6`. | Segment-order shuffle, matched-distance random endpoints, label permutation. | medium | low |
| `N608` | manifold geometry (ID-curvature coupling) | Intrinsic-dimension valley depth helps only through interaction with curvature drift, not as a standalone additive term. | Compute ID valley descriptors and curvature drift, fit interaction-only models (drop main effects), and compare to `H70`. | Interaction terms survive nulls where additive ID failed, with positive mean null-gap in `>=2/6`. | ID profile shuffle along path, interaction-break permutation, label permutation. | medium | medium |
| `N609` | cross-model alignment (topological transport) | Cross-model transfer improves when alignment is learned in persistence-landscape space instead of raw embedding space. | Build module-level persistence landscapes in scGPT/Geneformer, align with entropic OT, and use transport residuals + concordance as transfer features. | Domain null-gap positive in `>=2/3` domains and immune no longer strongly negative. | Random module mapping, landscape-bin permutation, label permutation. | high | high |
| `N610` | cross-model transfer (ontology-regularized) | Cell-ontology regularization makes cross-model module transport robust in immune without hurting lung domains. | Learn alignment with ontology penalty (train on two domains, evaluate held-out domain and immune), compare against unconstrained transport. | Immune null-gap becomes non-negative and global domain pass `>=2/3`. | Ontology-label permutation preserving group sizes, unconstrained baseline, random mapping. | high | medium |
| `N611` | biological anchoring (TRRUST/STRING) | Edges with both signed TRRUST motif support and high STRING confidence show enriched long-bar local topology. | Add interaction features between motif support, STRING strata, and local long-bar mass from PH; test over `H70`. | Positive interaction coefficient and positive mean null-gap in `>=4/9`. | TRRUST sign shuffle, STRING-bin permutation, label permutation. | high | medium |
| `N612` | biological anchoring (GO + cell ontology) | GO/cell-ontology boundary crossings induce geometric barrier signatures enriched in true regulatory edges. | Define boundary-crossing features (distance inflation, curvature spikes, boundary dwell), evaluate across domains and splits. | Positive mean null-gap in `>=3/6`, especially external-lung and lung. | Boundary-membership permutation preserving term sizes, boundary-depth matched randomization, label permutation. | medium | medium |
| `N613` | algorithmic signatures (mechanistic automata) | Biologically anchored automata with dwell-time and transition-entropy statistics outperform coarse grammar tokens. | Build states from sign-consistency x support x community-status across layer transitions; extract dwell, entropy, and motif-likelihood features. | Positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6`. | State-occupancy matched shuffle, layer-order permutation, label permutation. | high | low |
| `N614` | algorithmic signatures (causal cones) | True edges sit in asymmetric local fan-in/fan-out influence cones detectable from directed neighborhood structure. | Approximate local directed influence cones and cone-overlap metrics around each edge, then test additive/interactions over `H70`. | Positive mean null-gap in `>=3/6` with strongest effects in source-disjoint splits. | Edge-direction randomization preserving degree, cone membership shuffle, label permutation. | medium | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N600` (strict hardening of `H118`).
Why: current strongest signal, multiseed direction already stable, and failure mode is mostly null calibration depth/split stress.
Keep gate: positive mean null-gap in `>=5/9` domain-splits and positive direction in `9/9`.

2. High-risk/high-reward candidate: `N609` (cross-model persistence-landscape transport).
Why: this is a true objective reset for cross-model alignment and could convert repeated transfer negatives into a topology-grounded win.
Keep gate: positive domain null-gap in `>=2/3` domains with immune not negative.

3. Cheap broad-screen candidate: `N605` (directional geodesic asymmetry screen).
Why: directly targets the exact `H120` asymmetry failure using mostly existing infrastructure.
Keep gate: positive mean null-gap in `>=2/6` domain-splits and at least one source-disjoint split crossing zero.
