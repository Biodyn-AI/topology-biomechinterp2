# Retire / Deprioritize

1. `cross_model_alignment` as edge-utility/trajectory concordance in current design -> `retire_now`.
Reason: latest chain (`H71`, `H74`, `H77`, `H80`, `H83`) is consistently non-robust with null-gap failure.
Reopen condition: one `rescue_once_with_major_change` using topology-level module-complex comparison, not edge transfer.

2. Standalone additive `graph_topology` scores (`curvature`/`SBC` style) -> `retire_now`.
Reason: repeated negatives (`H61`, `H84`) with no positive domain-split support.
Reopen condition: only as conditional modifiers inside already-positive H82 neighborhoods.

3. Standalone intrinsic-dimension AUROC-lift branch -> `retire_now`.
Reason: consecutive negatives (`H54`, `H60`, `H63`, `H66`) and weak rescue potential.
Reopen condition: only as mechanism diagnostics conditioned on a positive topology branch.

4. Global biology-overlay interaction objectives on H70/H82 -> `rescue_once_with_major_change`.
Reason: base geometry remains positive but interaction null-gaps repeatedly fail (`H73`, `H76`, `H79`).
Required change: local/pathway-specific conditioning with pre-registered split-level gates.

# New Hypothesis Portfolio

1. `N420` `[topology]`
Hypothesis: true edges are enriched for edge-local cycles that persist jointly across geodesic and support-margin filtrations.
Test design: compute two-parameter local persistence area per edge neighborhood and add to H82/H70 baseline across domains/seeds/splits/layers `{7,11}`.
Expected signal if true: positive `delta_AUROC` in `>=5/6` domain-splits and positive mean null-gap in `>=4/6`.
Null/control: support-margin shuffle within geodesic bins, matched-random hotspot sets, label permutation.
Value: `high`; Cost: `medium`.

2. `N421` `[topology]`
Hypothesis: positives have smoother local-cycle vineyards across layers (`0,3,7,11`) than negatives.
Test design: compute per-edge local-cycle trajectory total variation and monotonicity; test incremental predictive value over H70.
Expected signal if true: lower variation for positives and positive AUROC uplift from trajectory features.
Null/control: layer-order permutation and endpoint-swap controls.
Value: `medium`; Cost: `low`.

3. `N422` `[topology]`
Hypothesis: witness complexes with biologically informed landmarks (TRRUST/STRING high-support genes) are more discriminative than random landmarks.
Test design: build local witness complexes using bio-landmark vs matched-random landmark sets and compare H1-derived edge scores.
Expected signal if true: bio-landmark branch exceeds random landmark q95 in most domain-splits.
Null/control: matched-random landmark sampling preserving degree/support distributions.
Value: `medium`; Cost: `medium`.

4. `N423` `[topology]`
Hypothesis: positives maintain local-cycle generators across source/target-disjoint zigzag transitions, while negatives do not.
Test design: compute split-paired local cycle overlap/persistence continuity per edge and add overlap score to H82.
Expected signal if true: overlap score is higher for positives and improves weak slices (immune/source, external_lung/target).
Null/control: split-label swap and neighborhood randomization.
Value: `high`; Cost: `high`.

5. `N424` `[topology + mechanism]`
Hypothesis: informative edges are associated with cycle deaths caused by shortcut closure rather than disconnected collapse.
Test design: track filtration events in local neighborhoods and classify cycle-death mode; test enrichment of closure-mode near positives.
Expected signal if true: closure-mode frequency is significantly higher in positives and H82 hotspots.
Null/control: endpoint permutation within geodesic bins and shuffled event labels.
Value: `medium`; Cost: `medium`.

6. `N425` `[manifold geometry]`
Hypothesis: curvature heterogeneity (variance, not mean) in edge neighborhoods is predictive of regulatory edges.
Test design: estimate local angle-deficit curvature at multiple neighborhood scales and add heterogeneity score to baseline.
Expected signal if true: heterogeneity yields positive delta where mean-curvature-only scores failed.
Null/control: feature shuffle within degree bins, random-hotspot control.
Value: `medium`; Cost: `low`.

7. `N426` `[manifold geometry]`
Hypothesis: positives sit in multiscale sign-switch regions where local curvature flips sign across `k={8,12,16}`.
Test design: compute per-edge curvature sign-switch count across scales and evaluate incremental utility.
Expected signal if true: switch count is enriched among positives and improves source-disjoint weak slices.
Null/control: scale-label permutation and endpoint swap.
Value: `medium`; Cost: `low`.

8. `N427` `[manifold geometry]`
Hypothesis: positives align with low tangent-drift trajectories across depth (stable local tangent orientation).
Test design: estimate local PCA tangent subspaces at each layer and compute endpoint tangent-drift features.
Expected signal if true: lower drift for positives and positive AUROC increment over geodesic baseline.
Null/control: layer-order permutation and neighborhood bootstrap randomization.
Value: `medium`; Cost: `medium`.

9. `N428` `[cross-model structure]`
Hypothesis: cross-model agreement is detectable at pathway module-complex topology level even when edge transfer fails.
Test design: build module-level persistence images per depth for scGPT and Geneformer strata; compare trajectory concordance.
Expected signal if true: positive null-gap concordance in `>=2/3` domains.
Null/control: module-label permutation, depth-order permutation, signature-destroy controls.
Value: `high`; Cost: `medium-high`.

10. `N429` `[cross-model structure]`
Hypothesis: barcode-distribution OT alignment recovers cross-model depth partial orders better than direct correlation.
Test design: compute layerwise barcode distributions and solve Wasserstein OT between models; score monotone depth alignment.
Expected signal if true: OT alignment score exceeds random transport q95 and is positive in at least two domains.
Null/control: random transport plans preserving marginals and depth-label permutation.
Value: `high`; Cost: `high`.

11. `N430` `[biological anchoring]`
Hypothesis: H82 hotspot cycles are enriched for TRRUST TF-target closure motifs supported by STRING high-confidence interactions.
Test design: extract top-decile local-cycle neighborhoods and test motif enrichment versus matched-random neighborhoods.
Expected signal if true: significant enrichment in at least two domains and strongest in immune/lung.
Null/control: degree/support-matched neighborhood randomization.
Value: `high`; Cost: `low-medium`.

12. `N431` `[biological anchoring]`
Hypothesis: topology gains are concentrated in specific cell-ontology gene strata and explain current weak domain-split slices.
Test design: stratify evaluated edges by cell ontology membership and rerun H82 summary metrics by stratum.
Expected signal if true: weak global slices decompose into strong positive and neutral strata with interpretable biology.
Null/control: ontology-label permutation preserving stratum sizes.
Value: `medium-high`; Cost: `low`.

13. `N432` `[algorithmic motif]`
Hypothesis: positives follow a small set of depth-lifecycle motifs (emerge, persist, close) in local cycle + defect trajectories.
Test design: discretize per-edge trajectories into motif tokens and test motif enrichment/transfer across domains.
Expected signal if true: one or more motifs are consistently enriched in positives in all domains.
Null/control: sequence permutation preserving per-layer token frequencies.
Value: `high`; Cost: `medium`.

14. `N433` `[cheap broad-screen]`
Hypothesis: a sparse combination of already-computable descriptors (local-cycle mean/max, triangle defect, bridge, curvature variance, tangent drift) yields a stronger universal screen than any single descriptor.
Test design: run seed42 breadth screen with lasso/logistic additive model over descriptor set and compare against H70 baseline.
Expected signal if true: positive mean delta in `>=4/6` domain-splits with at least `2/6` positive null-gap domain-splits.
Null/control: descriptor-column permutation within bins and label shuffle.
Value: `medium`; Cost: `low`.

# Top 3 for Immediate Execution

1. High-probability discovery candidate: `N420`.
Why: directly extends the only strong current signal (`H82`) while targeting known weak slices with a stronger local topology definition.
Success gate: positive mean `delta_AUROC` in `>=5/6` domain-splits and positive mean null-gap in `>=4/6`.

2. High-risk/high-reward candidate: `N429`.
Why: cross-model branch needs a structural reset; OT on barcode distributions is a materially different objective with publication upside if positive.
Success gate: positive null-gap in at least `2/3` domains on OT alignment score.

3. Cheap broad-screen candidate: `N433`.
Why: low engineering cost and fast orthogonal triage that can surface new combinations without committing to one mechanism.
Success gate: positive mean delta in `>=4/6` domain-splits; otherwise retire immediately.
