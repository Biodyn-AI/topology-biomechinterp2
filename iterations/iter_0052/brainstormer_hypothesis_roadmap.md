# Brainstormer Hypothesis Roadmap: iter_0052

## Retire / Deprioritize
| Direction | Status | Rationale |
|---|---|---|
| Cross-model descriptor/correspondence-free alignment endpoints (`H119/H122/H125/H131/H137`) | `retire_now` | Repeated negative domain-level null gaps after major resets |
| Additive ontology/module hardening endpoints (`H124/H127/H130/H135/H138`) | `rescue_once_with_major_change` | Directionality persists but strict-null survival repeatedly fails on hard slices |
| Standalone additive intrinsic-dimension uplift (`H114/H134` style) | `retire_now` | Repeated directional-only behavior with no strict null survival |
| Rank-surface/scalar PH surrogates (`H133` style) | `retire_now` | Decisive recent negatives and low rescue potential without changing filtration objective |
| Lightweight manifold descriptor add-ons (`H129/H132` style) | `rescue_once_with_major_change` | Permit one mechanism-level reset focused on strict-margin recovery; stop minor feature tuning |

## New Hypothesis Portfolio
| ID | Family | Hypothesis (one sentence) | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| Q01 | manifold_distance | Fixed-`k` neighborhoods are causing part of strict-margin failure, and density-adaptive neighborhoods will recover fail slices. | Rerun `H139` on fail slices (`external_lung/*`, `lung/source`, `lung/target`, `immune/dual`) using adaptive `k` (e.g., 8-20 by local density), seeds `42/43/44`, layer `11`. | Mean strict margin turns non-negative in at least `3/6` fail slices and external-lung dual-axis gains evaluable multi-seed support. | Same three `H139` nulls + shuffled `k` schedule within slice. | high | medium |
| Q02 | persistent_homology | A bifiltration over geodesic distance and support asymmetry captures robust cycles missed by single-axis descriptors. | Build coarse Betti-surface features (distance x support-margin bins) per slice and test lift over `H70`/`H139` on fail slices first. | Positive mean null-gap in at least `2/5` fail slices and better strict margins than `H139` baseline. | Support-bin permutation, filtration-axis swap, label permutation. | high | high |
| Q03 | topology_stability | True signal should persist as a smooth vineyard-like trajectory across neighborhood scale and seed, not just as per-`k` wins. | Extend `H140` to seeds `43/44`; compute per-slice continuity metrics across `k={8,12,16}` and seed-consistency score. | At least `5/8` evaluable splits gain-positive in `>=2` seeds with low cross-`k` variance. | Endpoint-swap control, seed-label shuffle, `k`-order permutation. | high | low |
| Q04 | manifold_geometry | Curvature sign transitions interacting with anisotropy (not raw curvature alone) mark regulatory edges. | Add Ollivier/Forman edge curvature and interaction terms with sectional features to the `H139` model on layer `11`. | Positive interaction terms and strict-margin improvement in lung source/target and immune dual-axis. | Curvature shuffle within degree-length strata; label permutation. | medium | medium |
| Q05 | intrinsic_dimension | Positive edges preferentially follow decreasing intrinsic-dimension flux along local geodesic routes. | Estimate node-level TWO-NN ID; build edge flux features from short geodesic paths and add to `H70+H139`. | Negative flux coefficient and new strict-positive support in at least `2` previously negative domain-splits. | Path-order permutation with path-length matching; label shuffle. | medium | medium |
| Q06 | manifold_distance | Patch-transition entropy (not chart-fracture mean) is the missing manifold signal in failure slices. | Cluster local PCA patch types and compute edge-level transition entropy/rarity features for fail slices. | External-lung and lung fail slices show positive null-gap where `H132` stayed null-negative. | Patch-label permutation within density bins; endpoint swap. | medium | low |
| Q07 | cross_model_alignment | TF-module persistence images provide a biologically anchored cross-model topology signal that survives strict nulls. | Build TRRUST/GO module-level persistence images in scGPT and Geneformer, align with anchor-constrained CCA/Procrustes, evaluate transfer AUROC. | Positive domain null-gap in `>=2/3` domains and immune no longer strongly negative. | Anchor shuffle, module permutation, random-map baseline. | high | high |
| Q08 | cross_model_alignment | Local chart-wise alignment can succeed even when global cross-model maps fail. | Partition genes into ontology charts, fit local maps per chart, aggregate predictions by chart-gated mixture. | At least one non-immune domain has positive domain null-gap versus random charts. | Chart assignment permutation with chart-size matching. | medium | high |
| Q09 | biological_anchor | Strict-null-surviving edges are enriched for multi-source biological concordance (TRRUST + STRING + GO) after confound control. | Model `strict_positive` as outcome with concordance tier plus degree/length covariates over `H139` rows. | Concordance coefficient positive and stable by domain. | Tier permutation within degree-length bins. | high | low |
| Q10 | biological_anchor | Cell-ontology stratification reveals hidden topology signal that pooled-domain runs dilute. | If ontology labels exist, rerun compact `H139` per ontology subgroup and meta-analyze by domain. | At least one ontology subgroup in external-lung or lung becomes strict-margin positive. | Ontology-label shuffle within domain. | medium | medium |
| Q11 | mechanistic_motif | Robust positives follow a recurrent depth motif (anisotropy rise then stabilization) across layers. | Tokenize edge trajectories over layers `{0,3,7,11}` and test motif enrichment/predictive lift. | Enriched motif odds (`>1`) and positive mean null-gap in at least `2` domain-splits. | Layer-order permutation, token shuffle, label shuffle. | medium | low |
| Q12 | mechanistic_motif | Strict-margin failures are dominated by specific null families, enabling targeted rescue instead of global tuning. | Decompose per-row margins versus each null family and cluster failure signatures by domain/split. | One dominant null family explains most failures (`>70%`) in each failing slice class. | Random dominant-null assignment bootstrap. | high | low |
| Q13 | topology | Multi-`k` geodesic detour-slope features capture mesoscale structure missed by static descriptors. | Compute detour ratios across `k` grid per edge and include slope/curvature terms on top of `H70`. | Positive gain with cross-seed sign consistency in at least `2/3` domains. | `k`-order permutation, endpoint-swap control. | medium | low |
| Q14 | persistent_homology | Local Betti change-points between layers 7 and 11 are mechanistic markers of regulatory edges. | Compute local neighborhood Betti summaries at L7 and L11, use delta/change-point features in edge model. | Positive null-gap in immune and lung target-disjoint slices. | Layer-label swap within seed; neighborhood rewiring control. | medium | medium |

## Top 3 for Immediate Execution
1. **High-probability discovery candidate: `Q01` (adaptive-neighborhood fail-slice rescue on `H139`).**
- Why: it directly targets known strict-margin failures while staying on the only currently promoted branch.
- Keep gate: external-lung dual-axis has `>=2` evaluable seeds and at least `3/6` fail slices become mean strict-margin non-negative.

2. **High-risk/high-reward candidate: `Q07` (TF-module persistence-image cross-model alignment).**
- Why: this is a true representation-object reset with biological anchors, not another descriptor tweak.
- Fast-fail gate: if positive domain null-gap is `0/3` in pilot, retire immediately.

3. **Cheap broad-screen candidate: `Q11` (depth motif token screen).**
- Why: low implementation cost, orthogonal to current additive-feature branch, and broad coverage across domains/layers.
- Keep gate: positive mean delta in at least half of tested domain-splits and at least one strict-null-positive split.
