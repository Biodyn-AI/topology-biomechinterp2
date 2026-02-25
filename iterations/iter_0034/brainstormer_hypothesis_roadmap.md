# Retire / Deprioritize

1. `cross_model_alignment` global depth/order/transfer formulations (`H74/H77/H80/H83/H86`) -> `retire_now`.
Reason: consecutive null-gap failures with `0/3` domain support in recent pilots.
Reopen rule: only with one major structural reset that first passes in-model positive controls.

2. Standalone additive `graph_topology` screens (`H61/H84` style) -> `retire_now`.
Reason: repeated negative utility and null-gap outcomes with low rescue yield.
Reopen rule: only as conditional features on top of a positive backbone (`H87`-like), not as standalone score.

3. Standalone intrinsic-dimension utility-lift objectives (`H54/H60/H63/H66`) -> `retire_now`.
Reason: repeated negative outcomes and no convincing rescue trajectory.
Reopen rule: diagnostic-only role inside mechanistic analysis, not promotion endpoint.

4. Global detour/dropout elasticity utility forms (`H78/H81`) -> `retire_now`.
Reason: one inconclusive then one negative iteration with collapsed null-gap support.
Reopen rule: only if reformulated as local phase-transition descriptors, not global perturb-and-score.

5. `H85` dual-filtration witness refinement -> `rescue_once_with_major_change`.
Reason: near gate but unstable null-gap in specific slices (`external_lung/*`, `lung/target_disjoint`).
Required change: adaptive bifiltration calibration by split/layer plus cycle event typing.

6. H70-line global biology interaction overlays (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Reason: directional utility without robust interaction null-gap.
Required change: stratum-local biological anchoring (cell ontology/module strata) instead of global interaction coefficients.

# New Hypothesis Portfolio

1. `N434` `[topology]`
Hypothesis: adaptive bifiltration thresholds (distance + support-margin) recover robust local witness-cycle signal where fixed-threshold H85 failed.
Test design: fit split/layer-specific bifiltration quantiles on training folds only, evaluate `delta_AUROC` and hotspot null-gap on held-out folds across seeds `42/43/44`, domains, splits, layers `{7,11}`.
Expected signal if true: positive mean null-gap in `>=5/6` domain-splits with recovery in `external_lung/*` and `lung/target_disjoint`.
Null/control: matched-random hotspot sets, within-bin feature shuffle, label permutation, plus quantile-label permutation.
Value: `high`; Cost: `medium`.

2. `N435` `[topology]`
Hypothesis: positives show persistent directed cycle motifs across depth transitions when persistence is computed on signed support-aware complexes.
Test design: build directed/signed local complexes at layers `{0,3,7,11}`, compute zigzag lifetime and motif counts per edge neighborhood, then test incremental utility over H70/H87 baseline.
Expected signal if true: positive `delta_AUROC` and positive null-gap in `>=4/6` domain-splits with strongest effects at layers `7/11`.
Null/control: depth-order permutation, sign-flip permutation, endpoint swap within geodesic bins.
Value: `high`; Cost: `medium-high`.

3. `N436` `[topology]`
Hypothesis: barcode entropy anisotropy (across filtration axes) separates true from false edges better than raw lifetime sums.
Test design: compute per-edge local barcode entropy and anisotropy descriptors from geodesic/support filtrations and run additive logistic screen against H70 baseline.
Expected signal if true: entropy-anisotropy features deliver positive mean uplift in `>=4/6` domain-splits.
Null/control: axis-swap control, descriptor shuffle within bins, label permutation.
Value: `medium`; Cost: `low`.

4. `N437` `[topology + mechanism]`
Hypothesis: cycle deaths caused by shortcut-closure events are enriched near positive edges, while collapse deaths are not.
Test design: classify local filtration critical events into closure/collapse types, compute event-frequency features per edge, and test enrichment plus utility gain.
Expected signal if true: closure-event fraction is significantly higher for positives and improves AUROC in at least two domains.
Null/control: event-label permutation within bins, matched-random neighborhood control.
Value: `medium`; Cost: `medium`.

5. `N438` `[topology_stability]`
Hypothesis: true-edge topology scores are locally stable under small metric perturbations while false-edge scores are fragile.
Test design: apply controlled geodesic perturbations (small kNN rewires and metric jitter), compute per-edge stability radius for H82/H87-related scores, and add stability to baseline.
Expected signal if true: positive edges have higher stability radius and stability features improve weak source-disjoint slices.
Null/control: over-perturbation stress control, perturbation-seed permutation, label permutation.
Value: `medium`; Cost: `medium`.

6. `N439` `[manifold geometry]`
Hypothesis: curvature anisotropy (directional variance) is informative even when scalar curvature averages are not.
Test design: estimate multi-scale directional curvature proxies around each edge and evaluate anisotropy ratios as features on top of H70.
Expected signal if true: anisotropy features produce positive mean uplift with consistent direction across both split regimes.
Null/control: direction randomization, scale-label permutation, feature shuffle.
Value: `medium`; Cost: `low`.

7. `N440` `[manifold geometry]`
Hypothesis: positives lie on low-divergence geodesic corridors under endpoint perturbation.
Test design: perturb edge endpoints within local neighborhoods, measure geodesic detour growth rate and divergence slope, and test predictive value.
Expected signal if true: positive edges show lower divergence and measurable AUROC uplift in at least two domains.
Null/control: endpoint swap within geodesic bins, neighbor-identity shuffle, label permutation.
Value: `medium`; Cost: `medium`.

8. `N441` `[manifold geometry + intrinsic dimension]`
Hypothesis: positives concentrate near local linearity phase boundaries (moderate reconstruction error, low depth-wise ID jump).
Test design: compute local PCA reconstruction error, local ID, and depth-wise ID-change descriptors; run a cheap breadth screen across all domains/splits/layers.
Expected signal if true: phase-boundary descriptors give positive mean uplift in `>=4/6` domain-splits.
Null/control: layer-order permutation, descriptor shuffle within bins, label permutation.
Value: `medium`; Cost: `low`.

9. `N442` `[cross-model structure transfer]`
Hypothesis: cross-model concordance emerges after mapping each model to within-model topological roles rather than aligning raw embeddings.
Test design: derive module-level role graphs per depth (from local persistence motifs), align role-transition matrices across models with GW/OT under domain matching, and score transition concordance.
Expected signal if true: positive null-gap in `>=2/3` domains on role-transition concordance metrics.
Null/control: role-label permutation, depth-order permutation, random role-graph baseline.
Value: `high`; Cost: `high`.

10. `N443` `[cross-model structure transfer]`
Hypothesis: a local-cycle motif dictionary learned in scGPT transfers to Geneformer strata without explicit edge mapping.
Test design: learn motif tokens from scGPT neighborhoods, encode Geneformer strata in motif space, and test motif-based retrieval/enrichment agreement across domains.
Expected signal if true: motif retrieval and enrichment exceed q95 null in at least two domains.
Null/control: motif-token permutation, dictionary shuffle, depth-order permutation.
Value: `high`; Cost: `medium-high`.

11. `N444` `[cross-model + biological anchoring]`
Hypothesis: anchor-restricted alignment on high-confidence TRRUST/GO/STRING modules reveals structure hidden in all-gene alignment.
Test design: restrict to high STRING-confidence module genes, compute depth-wise module signatures in both models, and align with contrastive CCA/Procrustes.
Expected signal if true: anchor-restricted alignment has positive null-gap where global alignment stays null.
Null/control: module-membership shuffle preserving size and confidence bins.
Value: `medium`; Cost: `medium`.

12. `N445` `[biological anchoring]`
Hypothesis: H87/H82 improvements are concentrated in specific cell-ontology strata and diluted by global pooling.
Test design: stratify edges by Cell Ontology gene sets, rerun H87/H82 summaries per stratum, and perform mixed-effects aggregation.
Expected signal if true: at least two strata show strong positive null-gap in slices currently weak at global level.
Null/control: ontology-label permutation preserving stratum sizes.
Value: `high`; Cost: `low-medium`.

13. `N446` `[biological anchoring + topology]`
Hypothesis: STRING-confidence-weighted filtration suppresses spurious cycles and improves topology utility in failing slices.
Test design: modify edge birth times by geodesic distance adjusted with STRING confidence, then recompute local persistence features and utility.
Expected signal if true: null-gap improves in `external_lung/*` and `lung/target_disjoint` compared with unweighted filtration.
Null/control: confidence shuffle within degree/support bins, label permutation.
Value: `medium`; Cost: `low`.

14. `N447` `[algorithmic signatures]`
Hypothesis: positives follow a small set of reproducible descriptor-state trajectories across depth that can be modeled as motif automata.
Test design: discretize per-edge descriptor vectors into states across layers, fit motif-transition features, and test enrichment plus utility gain.
Expected signal if true: a small motif subset is consistently enriched in positives across all three domains.
Null/control: sequence permutation preserving per-layer state marginals.
Value: `medium`; Cost: `medium`.

15. `N448` `[algorithmic signatures + robustness]`
Hypothesis: H87 is driven by a stable sparse descriptor subset across seeds/domains rather than seed-specific overfit.
Test design: run multiseed (`42/43/44`) H87 rerun with larger null budget, record nonzero descriptor sets and coefficient signs, and quantify stability (Jaccard/sign agreement) against utility.
Expected signal if true: stable descriptor core (`Jaccard >= 0.6`) with persistent positive mean null-gap in `>=5/6` domain-splits.
Null/control: descriptor-column shuffle, endpoint swap, label permutation, and coefficient-sign randomization control.
Value: `high`; Cost: `medium`.

# Top 3 for Immediate Execution

1. High-probability discovery candidate: `N448` (multiseed sparse-descriptor consensus).
Reason: `H87` is the strongest observed branch and this directly tests whether it is robust and mechanistically stable rather than single-seed luck.
Keep gate: positive mean delta in `>=5/6` domain-splits, positive mean null-gap in `>=5/6`, and descriptor-core stability `Jaccard >= 0.6`.

2. High-risk/high-reward candidate: `N442` (cross-model topological role-transition alignment).
Reason: cross-model work needs a structural reset away from raw transfer endpoints; role-space alignment is materially different with large upside.
Keep gate: positive null-gap in `>=2/3` domains on role-transition concordance.

3. Cheap broad-screen candidate: `N441` (local linearity phase-boundary screen).
Reason: low implementation cost, broad geometric coverage, and direct test of a fresh manifold+ID mechanism not recently screened.
Keep gate: positive mean delta in `>=4/6` domain-splits and positive mean null-gap in `>=2/6`.
