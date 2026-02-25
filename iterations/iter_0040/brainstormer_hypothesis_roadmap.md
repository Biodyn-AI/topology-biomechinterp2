# Brainstormer Hypothesis Roadmap - iter_0040

## Retire / Deprioritize
1. Cross-model static concordance / OT-depth warp endpoint family (`H65/H68/H71/H74/H77/H80/H83/H86/H96/H99/H102`) -> `retire_now`.
2. Derivative-spectrum additive + interaction-only-on-same-backbone forms (`H101/H103`) -> `retire_now`.
3. Conditioned-null-as-rescue expectation (`H105`) -> `retire_now` (keep conditioned nulls only as stricter controls).
4. Bridge-curvature additive lineage (`H95/H97`) -> `retire_now`.
5. Standalone/additive intrinsic-dimension utility lineage (`H54/H60/H63/H66/H89/H98`) -> `retire_now`.
6. Rewiring-survival-as-primary-objective lineage (`H07`-`H12`) -> `retire_now`.
7. Flat pooled module-support overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
8. Coarse tokenized depth-motif grammar as tested (`H104`) -> `rescue_once_with_major_change`.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null/control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N523` | topology | Edge labels are better separated by zigzag persistence across layers than by independent per-layer PH summaries. | Build zigzag complexes across layers `{0,3,7,11}` for edge-local neighborhoods and add zigzag Betti/lifetime features on top of `H93`. | Positive mean null-gap in `>=3/6` domain-splits with strongest gain in target-disjoint slices. | Layer-order scramble with marginal preservation, neighborhood membership shuffle, label permutation. | high | high |
| `N524` | topology | Two-parameter filtration (geodesic distance x signed-confidence) captures robust structure missed by one-parameter filtration. | Approximate bifiltration rank invariants on a fixed grid and summarize by Hilbert-surface descriptors added to `H91/H93`. | Positive mean null-gap in `>=3/6` domain-splits and lower split asymmetry than `H103/H104`. | Confidence-bin shuffle within degree strata, axis-swap control, label permutation. | high | medium |
| `N525` | topology | Positive edges concentrate in local witness-complex loops around endpoints. | Compute local witness-complex `H1/H2` lifetimes with landmark sampling around edge endpoints and test incremental AUROC over `H70`. | Positive-vs-negative lifetime gap and positive mean null-gap in `>=3/6` domain-splits. | Landmark identity shuffle preserving distance quantiles, endpoint-matched random pairs, label permutation. | medium | medium |
| `N526` | topology | Directed signed path-homology descriptors recover directionality that undirected topology misses. | Build directed kNN graphs with activation/repression channels and extract length-2/3/4 path-homology features as interactions with `H91`. | Positive directional contrast in both splits with positive mean null-gap in `>=3/6` domain-splits. | Direction swap, sign-flip within TF bins, degree-preserving directed rewiring, label permutation. | high | high |
| `N527` | manifold geometry | True regulatory edges lie on low-entropy, high-redundancy geodesic corridors. | Extract k-shortest-path multiplicity, corridor entropy, and detour volatility descriptors and test as additions to `H93`. | Lower corridor entropy for positives and positive mean null-gap in `>=3/6` domain-splits. | Degree-preserving rewiring, path-order randomization, endpoint-matched random pairs. | medium | medium |
| `N528` | manifold geometry | Source-target transport curvature asymmetry is predictive even when scalar curvature blends fail. | Compute Ollivier-Ricci (or Forman proxy) at endpoint neighborhoods and use asymmetry features plus interactions with `H91`. | Consistent asymmetry sign across domains and positive mean null-gap in `>=3/6` domain-splits. | Neighborhood measure shuffle preserving degree/radius, node swap within degree bins, label permutation. | medium | high |
| `N529` | manifold geometry | Positive edges traverse local-linear patches with controlled fracture rather than abrupt geometric breaks. | Estimate endpoint and corridor reconstruction error with local PCA and derive fracture-index features over `H70`. | Significant fracture-index interaction and positive mean null-gap in `>=3/6` domain-splits. | Patch-assignment shuffle within layer, corridor order permutation, label permutation. | medium | low |
| `N530` | manifold geometry | Intrinsic-dimension jumps matter only through interactions with confidence-weighted topology, not as standalone features. | Compute multi-radius ID jumps (`k={6,10,14}`) and fit interaction-only terms with `H93` weighted filtration gain. | Positive interaction coefficients in `>=4/6` domain-splits and positive mean null-gap in `>=3/6`. | Radius-order permutation, confidence shuffle within degree bins, label permutation. | high | medium |
| `N531` | cross-model alignment | scGPT and Geneformer align in perturbation-response ranks even when static structural concordance fails. | Apply matched perturbation library (module dropout, sign flip, local rewiring) and compare cross-model module response-rank concordance. | Positive domain null-gap in `>=2/3` domains despite weak static concordance baselines. | Perturbation-schedule permutation, module-label shuffle, random mapping baseline. | high | high |
| `N532` | cross-model alignment | Mapping learned on a high-confidence in-model topology core transfers to held-out non-core modules. | Train CCA/Procrustes on top-decile `H91/H93` core modules and evaluate retrieval/AUROC on non-core modules only. | Positive null-gap in `>=2/3` domains on held-out non-core evaluation. | Core-membership shuffle (size-matched), random-anchor map, depth-order permutation. | high | medium |
| `N533` | cross-model alignment | Persistent cycle generators correspond across models via shared-gene support beyond chance. | Extract top persistent generators per model/domain/layer, perform bipartite matching on shared genes, and score overlap concordance. | Generator-overlap score exceeds q95 null in `>=2/3` domains. | Gene-set size-preserving shuffle, layer permutation, random matching baseline. | medium | high |
| `N534` | biological anchoring | Separate TRRUST activation/repression channels in bifiltration rescue weak slices without harming strong ones. | Build sign-specific weighted-filtration channels with hierarchical shrinkage and compare against single-channel `H93`. | At least one weak slice flips to positive null-gap with zero strong-slice regressions. | Sign-label permutation, shared-channel baseline, label permutation. | high | low |
| `N535` | biological anchoring | GO parent-child contrast persistence is more robust than flat GO stratification. | Compute parent-minus-child persistence descriptors for GO hierarchy pairs and fit sparse group model on top of `H91/H93`. | Positive mean null-gap in `>=3/6` domain-splits with reduced domain heterogeneity vs `H94`. | GO hierarchy-edge rewiring preserving depth and module size, label permutation. | medium | medium |
| `N536` | biological anchoring | Cell Ontology partial pooling stabilizes cross-domain descriptor coefficients and lifts weakest splits. | Fit hierarchical logistic model with ontology-level random effects over `H91/H93+geometry` descriptors across all domains/splits. | Improved worst-split null-gap by `>=0.01` and positive mean null-gap in `>=4/6` domain-splits. | Ontology-label permutation (size-preserving), flat-model baseline, label permutation. | medium | medium |
| `N537` | algorithmic signatures | Positive edges follow a compact finite-state descriptor transition grammar across depth. | Discretize descriptor states across layers `{0,3,7,11}`, fit compact DFA/HMM on positives, and use likelihood-ratio scores as features. | Positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`. | Layer-order permutation preserving marginals, token shuffle within layer, label permutation. | medium | low |
| `N538` | biological anchoring | Adding STRING triad-closure priors to weighted filtration suppresses spurious uplift and improves robustness. | Combine STRING confidence with triangle-closure support into filtration weights and evaluate against `H93` backbone on layers `{7,11}` first. | Positive mean null-gap in `>=3/6` domain-splits with largest gains in currently marginal slices. | STRING-weight shuffle within degree bins, triad-count shuffle, label permutation. | high | low |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N538` (STRING triad-closure weighted filtration).
Why: closest to the strongest positive branch (`H93`), low engineering overhead, and directly targets robustness rather than null relaxation.
Pilot design: seed42, domains `{immune,lung,external_lung}`, splits `{source_disjoint,target_disjoint}`, layers `{7,11}`, null draws `>=24` per family.
Keep gate: positive mean null-gap in `>=3/6` domain-splits.
Fail-fast: if `0/6` after pilot, retire immediately.

2. High-risk/high-reward candidate: `N531` (cross-model perturbation-response alignment).
Why: fundamentally new cross-model target (response consistency instead of static concordance) with clear upside if it works.
Pilot design: seed42 pilot on shared genes/modules across all three domains, perturbation panel size `>=24` per domain.
Keep gate: positive domain-level null-gap in `>=2/3` domains.
Fail-fast: if `0/3`, re-retire cross-model branch for at least three loops.

3. Cheap broad-screen candidate: `N537` (finite-state descriptor motifs).
Why: very low cost and orthogonal to recent additive failures.
Pilot design: seed42 breadth run over all domains/splits using existing `H91/H93` descriptor traces, null draws `>=20`.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
Fail-fast: if positive mean delta `<3/6`, stop lineage.
