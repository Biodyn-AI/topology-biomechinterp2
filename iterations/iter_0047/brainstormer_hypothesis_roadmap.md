# Brainstormer Hypothesis Roadmap - iter_0047

## Retire / Deprioritize
1. Cross-model direct transfer utility line (`H96/H99/H102/H109/H119/H122/H125`) -> `retire_now`.
Evidence: repeated `0/3` domain null-gap support after multiple major resets.

2. Additive topology-stability utility forms (`H90/H92/H107/H110/H111/H112/H113`) -> `retire_now`.
Evidence: broad directional drift without strict-null survival.

3. Standalone additive intrinsic-dimension line (`H60/H63/H66/H98/H114`) -> `retire_now`.
Evidence: repeated negative/inconclusive outcomes with low rescue yield.

4. Scalar additive PH rescue chain (`H100/H101/H103/H106`) -> `retire_now`.
Evidence: repeated inability to clear null-gap gates under changed filtrations.

5. Cross-model family overall -> `rescue_once_with_major_change`.
Rule: one slot max; objective must be structure-invariance stability, not direct edge-transfer AUROC.

6. `H124` lineage (`H116/H118/H123/H124`) -> `keep_active_high_priority`.
Reason: strongest signal branch, with clear bottleneck in split-specific null calibration.

7. `H126` lineage (`H121/H126`) -> `rescue_once_with_major_change`.
Rule: focus on source-disjoint and external_lung fragility with multi-scale path geometry.

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N630` | topology (multiparameter PH) | True regulatory edges lie on stable ridges in a signed-support x geodesic-radius persistence surface. | Build 2-parameter local filtrations at layers `{7,11}` and extract ridge-mass/landscape features over `H70`. | Positive mean null-gap in `>=3/6` domain-splits, with source-disjoint gains in at least 2 domains. | Axis permutation within degree bins, ridge-surface shuffle, label permutation. | high | high |
| `N631` | topology (sheaf/cohomology) | Positive edges show lower signed-cycle inconsistency when encoded as sheaf cohomology obstructions. | Build local signed sheaf on kNN neighborhoods and compute `H1` obstruction norms as edge features. | Negative obstruction for positives and positive null-gap in `>=3/6` domain-splits. | TF-identity sign shuffle, local cycle orientation shuffle, label permutation. | high | high |
| `N632` | topology stability (vineyard dynamics) | True edges have smoother barcode-vineyard trajectories under controlled local perturbations. | Generate perturbation cones around each edge and model vineyard displacement curvature and total variation. | Positive mean null-gap in `>=2/6` and stronger support in source-disjoint splits. | Perturbation-schedule permutation, edge-local random cone baseline, label permutation. | medium | medium |
| `N633` | topology (directed path homology v3) | Directed support-weighted path homology captures causal direction better than undirected topological summaries. | Build support-asymmetric DAG neighborhoods and compute path-homology descriptors (`Betti1/Betti2`, path entropy). | Positive mean null-gap in `>=3/6`; immune/source should stay non-negative. | Direction reversal within degree bins, DAG edge rewiring preserving degree, label permutation. | medium | medium |
| `N634` | manifold geometry (torsion spectrum) | Multi-scale turning-angle/torsion spectra have conserved directional signatures for true edges. | Extend `H126` with scales `{8,12,16}` and spectral moments; evaluate over `H70` at layers `{7,11}`. | Positive mean null-gap in `>=3/6` domain-splits with at least one source-disjoint and one target-disjoint pass. | Path reversal within length bins, endpoint swap within distance bins, label permutation. | medium | low |
| `N635` | manifold geometry (convexity x entropy) | Geodesic convexity defect combined with detour entropy separates positives from negatives better than either alone. | Compute convexity defect, detour entropy, and their interaction over edge-local shortest-path bundles. | Positive interaction coefficient and positive mean null-gap in `>=3/6`. | Detour-order shuffle, endpoint swap controls, label permutation. | medium | low |
| `N636` | manifold geometry + ID interaction | Directional intrinsic-dimension gradients are useful only as interactions with directional geodesic features. | Estimate TWO-NN ID at endpoints and path interior; use interaction-only terms with `H126` features. | Positive interaction null-gap in `>=2/6`, especially source-disjoint slices. | Endpoint-ID swap within distance bins, path direction reversal, label permutation. | medium | medium |
| `N637` | manifold geometry (chart stitching) | True edges traverse more chart-consistent local tangent maps than negatives. | Fit local PCA charts along geodesics and measure stitching residual/holonomy as edge features. | Lower residual for positives and positive mean null-gap in `>=3/6`. | Chart-basis random rotation within local neighborhoods, path-order shuffle, label permutation. | medium | medium |
| `N638` | cross-model structure transfer | Cross-model perturbation fields share homologous persistence-image geometry even under noisy correspondence. | Build perturbation-response complexes in both models, compute persistence images, align with sliced-Wasserstein and score alignment stability. | Positive domain null-gap in `>=2/3`, with immune non-negative. | Random module remap, perturbation-phase shuffle, label permutation. | high | high |
| `N639` | cross-model alignment (invariant subspace) | Anchor-masked contrastive subspaces recover stable cross-model structure not captured by direct transfer metrics. | Train small contrastive projections using TRRUST/GO anchor positives and hard negatives; evaluate cycle-residual stability by domain. | Positive domain null-gap in `>=2/3` and reduced immune failure versus prior endpoints. | Anchor-label permutation preserving counts, random projection baseline, label permutation. | high | high |
| `N640` | cross-model alignment (chart-wise) | Cell-ontology chart-wise alignment is more null-robust than global alignment. | Partition modules by cell ontology chart, fit per-chart orthogonal maps, aggregate chart-consistency and mismatch features. | At least `2/3` domains positive null-gap with lung/external_lung retained and immune improved. | Chart-label permutation, chart-membership shuffle, global-map baseline. | high | medium |
| `N641` | biological anchoring (H124v2) | H124 becomes robust when TRRUST-sign, STRING confidence, and GO co-membership are combined with adversarial decoy controls. | Extend `H124` with hierarchical prior interactions and adversarial motif decoys matched on TF-degree-community strata. | Positive mean null-gap in `>=6/9` domain-splits and `lung/dual_axis_disjoint` null-gap `>0`. | Existing strict null quartet + adversarial decoy shuffle. | high | medium |
| `N642` | biological anchoring (boundary barriers) | Regulatory edges exhibit characteristic barrier-crossing signatures at GO/cell-ontology boundaries. | Compute boundary crossing count, dwell, and curvature spikes around ontology transitions; test over `H70` and `H124` residuals. | Positive mean null-gap in `>=3/6`, strongest in lung/external_lung. | Boundary-label permutation size-matched, boundary-depth shuffle, label permutation. | medium | medium |
| `N643` | algorithmic signatures (continuous automata) | Continuous-state motif automata with dwell-time and transition entropy recover mechanism structure missed by coarse FSMs. | Build continuous trajectory states from sign/support/community variables and fit dwell/entropy descriptors per edge. | Positive mean null-gap in `>=2/6` and improvement over `H107/H111` baselines. | Occupancy-matched state shuffle, layer-order permutation, label permutation. | medium | medium |
| `N644` | algorithmic signatures (counterfactual motifs) | True edges have distinctive local counterfactual response motifs under minimal graph edits. | Apply edge-local swap/flip interventions and summarize response asymmetry signatures as features. | Larger structured response drops for positives and positive mean null-gap in `>=3/6`. | Degree-preserving non-local swap control, random edit sequence control, label permutation. | high | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N641` (H124v2 hierarchical biological hardening with adversarial decoys).
Why: strongest active branch already exists; this directly targets the known null-gap bottleneck instead of chasing new raw lift.
Keep gate: positive mean null-gap in `>=6/9` domain-splits, `lung/dual_axis_disjoint > 0`, and no domain with all splits null-negative.

2. High-risk/high-reward candidate: `N638` (cross-model perturbation-field persistence alignment).
Why: this is a true objective shift from failed transfer endpoints and can revive cross-model signal with topology-level invariants.
Keep gate: positive domain null-gap in `>=2/3` domains, immune domain null-gap `>=0`.

3. Cheap broad-screen candidate: `N634` (multi-scale torsion spectrum over `H126`).
Why: low implementation overhead on existing path features, high information yield on where source-disjoint failures come from.
Keep gate: positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6` with at least one source-disjoint pass.
