# Retire / Deprioritize

1. Cross-model utility-promotion endpoints (`H65/H68/H71/H74/H77/H80`) -> `retire_now`.
Reason: repeated null-gap failures across materially changed formulations; latest endpoint is still `0/3` domain mean null-gap positive.
Reopen rule: only if endpoint is changed to cross-model invariance/stability (not transfer utility), with a pre-registered null-gap gate.

2. Global detour/dropout elasticity utility formulations (`H78/H81`) -> `retire_now`.
Reason: repeated null-robustness failure, then outright utility drop in `H81`; perturbation signal is degenerate in many rows.
Reopen rule: only for edge-local witness-path perturbation with explicit path-survival metrics.

3. Standalone intrinsic-dimension utility-lift line (`H60/H63/H66`) -> `retire_now`.
Reason: repeated negative or unstable AUROC deltas; low rescue potential in current objective form.
Reopen rule: only if reframed as mechanism diagnostics conditioned on a known positive geometry branch.

4. Support-interaction overlays on H70 (`H73/H76/H79`) -> `rescue_once_with_major_change`.
Reason: base geometry stays positive, but global interaction objective is not null-robust.
Required change: localized/depth-conditional interaction objective with strict pre-registered keep gate.

5. Curvature-as-direct-score branch (`H75` plus earlier curvature negatives) -> `rescue_once_with_major_change`.
Reason: repeated weak/negative behavior in direct edge-scoring forms.
Required change: use curvature as a conditional modifier or transition marker, not a standalone score.

# New Hypothesis Portfolio

1. `N399` (topology + mechanism): H70-positive edges are supported by locally persistent witness cycles that are absent in matched-random edges.
Test design: build edge-centered witness complexes in radius-stratified neighborhoods, compute local H1 lifetime sum at layers `{7,11}`, and compare `H70-top-quintile` vs matched-random edges across domains/splits/seeds.
Expected signal if true: positive local cycle persistence gap and positive null-gap in most domain-splits.
Null/control: degree/geodesic/coexpression matched edge sets, endpoint-swap controls, label shuffles.
Value: `high`; Cost: `medium`.

2. `N400` (topology stability): Topological vineyards across depth have smoother birth/death trajectories for true regulatory edges than controls.
Test design: compute persistence vineyards from layers `0->3->7->11`, score trajectory smoothness and event continuity, and regress utility gain on these trajectory features.
Expected signal if true: regulatory edges show higher continuity and continuity features improve AUROC over baseline geometry.
Null/control: layer-order permutation, within-layer simplex rewiring, label shuffle.
Value: `high`; Cost: `high`.

3. `N401` (topology variant): Signed directed path-homology around support-concordant subgraphs captures regulatory motifs beyond directed/signed baseline.
Test design: construct directed signed subgraphs around candidate edges; compute path-homology-derived features and add to H70 baseline.
Expected signal if true: positive delta AUROC and positive null-gap in at least `4/6` domain-splits.
Null/control: sign permutation preserving degree and in/out flow, path rewiring.
Value: `medium`; Cost: `high`.

4. `N402` (filtration variant): Quantile-adaptive two-parameter filtration (distance x support-margin quantile) rescues earlier multiparameter failures by controlling sparsity mismatch.
Test design: run rank-quantile bifiltration with equal-mass bins and compare against H70 baseline.
Expected signal if true: non-negative utility deltas with improved null-gap versus prior rank-surface runs.
Null/control: quantile-label shuffle, matched random quantile assignments.
Value: `medium`; Cost: `medium`.

5. `N403` (manifold geometry): Ollivier-Ricci curvature conditioned on geodesic detour state identifies high-value regulatory neighborhoods.
Test design: compute Ollivier-Ricci on kNN graphs, interact curvature with H70 defect score, evaluate incremental AUROC.
Expected signal if true: interaction term significantly positive and null-gap positive in multiple domain-splits.
Null/control: graph rewiring preserving degree and edge-length histogram, curvature-bin label shuffles.
Value: `medium`; Cost: `medium`.

6. `N404` (manifold perturbation rescue): Edge-local witness-path ablation reveals causal dependence of true positives on defect-rich geodesic shortcuts.
Test design: for each evaluated edge, ablate top witness nodes/edges on its geodesic shortcut path, recompute local features, and measure targeted-vs-random degradation.
Expected signal if true: targeted ablation reduces predicted score/utility more than matched random ablation.
Null/control: matched random path-node ablation, endpoint swap, feature shuffle.
Value: `high`; Cost: `medium`.

7. `N405` (local linearity): Tangent-subspace angle instability near candidate edges predicts regulatory positives beyond geodesic distance.
Test design: compute local PCA tangent spaces around edge endpoints; use principal-angle mismatch and curvature proxies as features.
Expected signal if true: positive incremental AUROC with strongest gains in late layers.
Null/control: neighborhood label shuffle, tangent recomputation on randomized neighborhoods.
Value: `medium`; Cost: `low`.

8. `N406` (intrinsic dimension mechanism): ID changepoints across layers (not absolute ID) explain where geometry utility emerges.
Test design: estimate local ID per layer per endpoint, fit changepoint features (`0->3`, `3->7`, `7->11`) and test conditional utility linkage on H70.
Expected signal if true: one transition (likely `3->7` or `7->11`) shows stable positive coupling with utility.
Null/control: layer-order permutation and endpoint-matched random transitions.
Value: `medium`; Cost: `low`.

9. `N407` (cross-model invariance, high-risk): Pathway-level persistence-image trajectories are conserved across scGPT and Geneformer after domain-wise whitening.
Test design: derive pathway persistence images across depth in both models and score trajectory concordance (distance Spearman, CKA trajectory, top-k retrieval).
Expected signal if true: positive concordance that exceeds pathway-label and trajectory-destroy nulls.
Null/control: pathway label permutation, depth-order permutation, signature destroy.
Value: `high`; Cost: `high`.

10. `N408` (cross-model structure transfer): Relative pathway geodesic ordering (rank constraints) transfers better than absolute centroid distances.
Test design: learn pairwise ordering constraints in source model and evaluate order-consistency in target model across domains.
Expected signal if true: order-agreement significantly above null even when absolute map quality is weak.
Null/control: random order constraints, pathway-membership shuffle.
Value: `medium`; Cost: `medium`.

11. `N409` (cross-model mechanistic motifs): Edge-state transition motifs across depth form a shared low-cardinality motif alphabet between models.
Test design: discretize edge trajectories into motif tokens; compare motif frequency and transition matrices cross-model.
Expected signal if true: motif alignment significantly above shuffled-token null and concentrated in biologically annotated modules.
Null/control: token permutation preserving marginal frequencies, depth-order shuffle.
Value: `medium`; Cost: `medium`.

12. `N410` (biological anchoring): H70 top-decile edges are enriched for STRING-supported TRRUST TF-target pairs specifically within relevant cell ontology branches.
Test design: stratify by cell ontology context, compute enrichment and incremental predictive value of ontology-conditioned support.
Expected signal if true: enrichment and utility gains are concentrated in ontology-matched strata, not global background.
Null/control: ontology-label permutation, matched support-degree shuffles.
Value: `high`; Cost: `medium`.

13. `N411` (biological anchoring + topology): Persistent generators map to coherent GO/TF programs more strongly than generators from matched-null filtrations.
Test design: extract top generators from local filtrations; run GO/TRRUST enrichment and compare enrichment stability to null generators.
Expected signal if true: higher enrichment stability and stronger biological coherence for observed generators.
Null/control: null generator sets from filtration-label permutations and matched random complexes.
Value: `medium`; Cost: `medium`.

14. `N412` (cheap algorithmic screen): Shortcut-bridge competition index (triangle-defect vs bridge dependence) is a fast predictor of regulatory edges.
Test design: compute a two-feature index per edge from existing graphs, screen seed42 across all domains/splits/layers.
Expected signal if true: positive mean delta AUROC in at least `4/6` domain-splits with at least one positive null-gap domain-split.
Null/control: endpoint swap, feature shuffle, label shuffle (low permutation budget).
Value: `medium`; Cost: `low`.

# Top 3 for Immediate Execution

1. High-probability discovery candidate: `N399` (local witness-cycle persistence on H70 hotspots).
Why now: it directly leverages the strongest active positive branch and tests a concrete mechanism likely to preserve signal.
Keep gate: mean delta AUROC > `0` and mean null-gap > `0` in at least `4/6` domain-splits.

2. High-risk/high-reward candidate: `N407` (cross-model pathway persistence-trajectory invariance).
Why now: cross-model utility endpoints are exhausted; invariance can still yield publishable structure even if transfer utility fails.
Keep gate: at least `2/3` domains with positive null-gap on primary concordance metric.

3. Cheap broad-screen candidate: `N412` (shortcut-bridge competition index).
Why now: low engineering cost, orthogonal mechanism, quick signal triage for next branching.
Keep gate: positive mean delta AUROC in at least `4/6` domain-splits.
