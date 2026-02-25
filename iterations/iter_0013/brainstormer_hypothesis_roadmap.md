# Brainstormer Hypothesis Roadmap — iter_0013

## Retire / Deprioritize
| Direction | Evidence | Decision |
|---|---|---|
| Rewiring-null survival (`H07/H09/H12`) | Three iterations of calibrated negatives (`iter_0006`-`iter_0008`) with zero rescue trend | `retire_now` |
| Rewiring distortion-lower-tail rescue | Repeated non-significance and no sign correction | `retire_now` |
| Plain Hungarian OT unsupervised alignment | `iter_0012` map collapse (mean top-1 `0.0024`, `0/3` significant) | `retire_now` |
| Raw Forman negative-curvature enrichment (`H23`) | `iter_0013` below-chance AUROC in all domains and negative enrichment deltas everywhere | `retire_now` |
| Confidence-tier monotonicity as raw slope (`H19`) | Directional failure in both splits with likely prevalence saturation | `rescue_once_with_major_change` |
| Intrinsic branch framed as universal positive coupling (`H04/H18/H21/H22`) | Four neutral/mixed outcomes with domain/split sign flips | `rescue_once_with_major_change` |
| Bridge-conditioned topology explanation (`H11`) | Previous run not identifiable because bridge strata were split-confounded | `rescue_once_with_major_change` |
| Coarse disagreement-bin trend (`H15` style) | Domain-heterogeneous signs and coarse-bin confounding | `rescue_once_with_major_change` |

## New Hypothesis Portfolio
| ID | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|
| N98 | A geodesic-radius x biological-confidence bifiltration reveals stable topological signal missed by single-parameter PH. | Build bifiltration slices using geodesic distance and prior score (TRRUST/STRING), compute rank-invariant persistence summaries per domain/split/layer. | Persistent features concentrate in high-confidence slices and replicate across seeds in `>=2/3` domains. | Confidence-label permutation with matched marginals + distance-stratified edge shuffle. | high | high |
| N99 | Cross-layer zigzag persistence captures depth-persistent structures that explain split asymmetry. | Construct zigzag complexes over consecutive layers for matched genes and compare interval lifetimes with H22 split differences. | Long intervals align with layers/phases showing strongest split effects. | Layer-order permutation and random layer stitching. | high | high |
| N100 | TF-hub neighborhoods have stronger relative homology structure than degree-matched non-hub neighborhoods. | Compute relative homology (hub neighborhood vs local background) for top TF hubs and matched controls across layers. | TF-hub relative Betti signal is consistently larger and enriches regulatory positives. | Degree-matched hub-label shuffle. | medium | medium |
| N101 | Persistence-image transport distance between splits/models is a sensitive stability diagnostic for promoted branches. | Convert barcodes to persistence images, compute Wasserstein distances across split/model pairs, and regress against H24 transfer quality. | Lower transport cost for pairs with stronger cross-model alignment and better edge transfer. | Gene-correspondence permutation and image-pixel permutation baseline. | medium | medium |
| N102 | Density-equalized Mapper graphs expose loop-like modules linked to regulatory edge enrichment. | Build Mapper with geodesic-centrality and reconstruction-error lenses, enforce density-balanced landmarks, and test loop-node enrichment. | Mapper loop regions show higher positive-edge rates and stronger ontology enrichment than non-loop regions. | Lens-value permutation with identical cover/cluster parameters. | medium | medium |
| N103 | Degree-residual Ollivier-Ricci curvature (not raw Forman) positively associates with true regulatory edges. | Compute weighted Ollivier curvature on kNN graphs, then fit edge-label models with curvature + degree + edge-length covariates. | Curvature coefficient remains positive/significant in `>=2/3` domains after adjustment. | Label permutation + degree-preserving graph rewiring. | high | medium |
| N104 | Positive regulatory edges form geodesically non-convex conduits with measurable convexity deficit. | For positive-edge endpoints, compute geodesic convex-hull occupancy and compare against matched negatives per layer/split/domain. | Positive edges show larger convexity deficit, especially in high-geodesic-gain layers. | Endpoint shuffle matched on degree and expression. | medium | low |
| N105 | Diffusion distance at intermediate times outperforms both Euclidean and shortest-path geodesic distance for edge discrimination. | Sweep diffusion times and compute AUROC by layer/split/domain; compare to existing geodesic and Euclidean baselines. | A non-trivial diffusion-time window yields consistent AUROC lift in multiple domains. | Degree-preserving random-walk matrix permutation and label shuffle. | high | low |
| N106 | Intrinsic-dimension distribution shape (tail/skew/dispersion) explains H22 sign heterogeneity better than mean ID. | Estimate local ID via TWO-NN and participation-ratio variants; model split-sign outcomes with distribution-shape statistics. | Tail/skew features predict immune vs lung/external sign behavior with stable coefficients. | Within-domain layer permutation and estimator-swap robustness control. | medium | low |
| N107 | CCA-seeded Sinkhorn OT can recover correspondence without explicit one-to-one supervision. | Use H24 CCA latent as initialization, run entropic OT annealing, and evaluate map/topology transfer metrics vs baselines. | OT map quality and transfer quality exceed random map and no-seed OT in `>=2/3` domains. | Latent-space random rotation + correspondence permutation nulls. | high | high |
| N108 | Unseeded Gromov-Wasserstein graph alignment can recover cross-model structure from geometry alone. | Align scGPT and Geneformer kNN graphs per domain using GW and score cycle-consistency, top-1 retrieval, and transfer AUROC. | Non-trivial recovery beyond null with at least one domain showing clear uplift over random graph alignment. | Degree-matched graph permutation and reversed-neighborhood controls. | high | high |
| N109 | Dynamic layer alignment (DTW over geometric signatures) beats fixed-layer matching in cross-model transfer. | Build per-layer signatures (PH summaries, ID-shape, curvature residuals, module AUC), run DTW, and evaluate transfer on DTW-paired layers. | DTW-paired layers outperform fixed layer mapping on H24 metrics in `>=2` domains. | Random layer pairing and reversed-order DTW controls. | medium | medium |
| N110 | Geometry/alignment effects are strongest on edges jointly supported by TRRUST, STRING, and GO co-membership. | Fit mixed-effects edge models with geometry features, prior-support indicators, and interaction terms by split/domain. | Positive and significant geometry x prior interactions in both disjoint splits. | Prior-label permutation and degree-stratified bootstrap. | high | medium |
| N111 | Cell-ontology program-restricted subgraphs amplify true signal and reduce cross-domain cancellation. | Build Cell Ontology marker-restricted edge subsets (immune, epithelial, stromal, etc.) and rerun H22/H24-style tests per subset. | Specific ontology programs show stronger and more stable effects than pooled analyses. | Size- and expression-matched random gene-set controls. | medium | medium |
| N112 | Feed-forward and feedback motif classes occupy distinct manifold/topology regions that transfer across models. | Annotate edges by motif class from priors, embed with geometry/topology features, and test motif separability plus cross-model centroid transfer. | Motif-class separability is above null and preserved after alignment. | Motif-label permutation with degree-preserving motif rewiring. | high | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: **N110 (multi-prior biological anchoring with interaction models)**.
- Why now: it leverages the strongest positive branch (`H24`) and resolves the main interpretation gap (biological grounding).
- Immediate gate: geometry x prior interaction terms are positive with permutation-calibrated `p < 0.05` in both splits.

2. High-risk/high-reward candidate: **N108 (unseeded GW graph alignment)**.
- Why now: it is the cleanest test of whether cross-model structure can be recovered without correspondence scaffolding.
- Immediate gate: GW beats random graph alignment on map quality and transfer metrics in at least one domain, with directional consistency in `>=2/3` domains.

3. Cheap broad-screen candidate: **N105 (diffusion-distance sweep)**.
- Why now: low implementation cost, broad coverage, and directly comparable to existing distance baselines.
- Immediate gate: at least one diffusion-time band beats both Euclidean and geodesic AUROC in `>=2/3` domains.
