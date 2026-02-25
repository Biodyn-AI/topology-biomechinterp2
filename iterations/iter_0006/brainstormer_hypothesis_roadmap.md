# Brainstormer Hypothesis Roadmap — iter_0006

## Candidate Next Hypotheses (N23-N33)

### Topology Family

#### N23 — Metric-matched stronger null recovers interpretable dynamic range
- Hypothesis: the rewiring branch is failing mainly because observed topology is Euclidean while null topology is geodesic; matching both to geodesic space will reduce the all-negative collapse.
- Experiment: for each seed/layer/split, compute (a) observed Euclidean H1, (b) observed geodesic H1 on original connected graph, (c) rewired geodesic null; test both comparisons separately.
- Success criterion: rewired-vs-observed geodesic shows non-trivial spread (not all `p=1.0`) and at least `>=2/12` layers significant in either split.
- value: high
- cost: medium

#### N24 — Connectivity-stable kNN design changes rewiring conclusions
- Hypothesis: near-max-k plus heavy bridge usage is the dominant artifact driver for rewiring-null failure.
- Experiment: rerun immune full-layer rewiring with a connectivity-stable design (`k` sweep such as `30/40/50` or mutual-kNN + deterministic augmentation) and report bridge counts per row.
- Success criterion: bridge usage drops substantially and mean rewiring deltas shift upward versus iter_0006 baseline.
- value: high
- cost: medium

#### N25 — Late-layer dual-split robustness replicates cross-domain
- Hypothesis: the immune late-layer cluster (`7, 9, 10, 11`) reflects a broader scGPT depth motif, not a single-domain effect.
- Experiment: run the same full-layer split map on `external-lung` under feature-shuffle (and calibrated stronger null if available).
- Success criterion: at least 3 late layers (top quartile depth) pass dual-split significance in external-lung.
- value: high
- cost: medium

#### N26 — Target-split weakness is partly a sample-size/power effect
- Hypothesis: target-split near-miss layers are underpowered at `n_points=180` rather than structurally null.
- Experiment: rerun feature-shuffle branch for target split at `n_points` grid (`180, 240, 300`) on layers `6, 8, 11`.
- Success criterion: at least one near-miss layer crosses Fisher `p<0.05` with stable positive delta.
- value: medium
- cost: medium

### Manifold Geometry Family

#### N27 — Split-robust layers have higher geodesic distortion from Euclidean geometry
- Hypothesis: robust late layers occupy manifolds where Euclidean chords understate path geometry.
- Experiment: per layer compute geodesic/euclidean distortion metrics (mean ratio, percentile tails) on original connected graphs; correlate with dual-split pass indicators.
- Success criterion: consistent positive association across seeds between distortion and dual-split robustness.
- value: medium
- cost: low

#### N28 — Robust layers are characterized by specific barcode shape, not only total H1 sum
- Hypothesis: long-lived-cycle concentration (barcode inequality) is a better discriminator than total H1 lifetime sum.
- Experiment: add descriptors (max lifetime, top-k lifetime share, barcode entropy, Betti curve area) for observed and null distributions.
- Success criterion: one descriptor separates dual-split-pass vs non-pass layers better than total H1 sum alone.
- value: medium
- cost: medium

### Cross-Model Alignment Family

#### N29 — Residual-level layer ranking aligns between scGPT and Geneformer
- Hypothesis: on matched genes, layer-wise topology effect ranking is concordant across models in immune/external-lung.
- Experiment: surface matched-gene residual embeddings for Geneformer and compute per-layer H1 effect sizes with matched protocol; test Spearman rank alignment with permutation null.
- Success criterion: Spearman `rho > 0.5` and permutation `p<0.05` in at least one domain.
- value: high
- cost: high

#### N30 — Persistence-image similarity is stronger in late layers across models
- Hypothesis: cross-model agreement is localized to deeper layers where split robustness appears in scGPT.
- Experiment: compute persistence images/landscapes for matched layers and compare with cosine/CKA under layer-shuffle null.
- Success criterion: late-layer similarity significantly exceeds early-layer similarity under null.
- value: medium
- cost: high

### Biological Anchoring Family

#### N31 — Dual-split robust immune layers enrich immune TF regulatory programs
- Hypothesis: layers `7, 9, 10, 11` are enriched for coherent TF-driven immune programs rather than generic topology.
- Experiment: rank genes by contribution to persistent cycles per layer/split; run TRRUST/DoRothEA enrichment with FDR control.
- Success criterion: replicated enriched TF programs across source and target splits in at least 2 layers.
- value: high
- cost: medium

#### N32 — Split-robust layers preserve pathway/module overlap across split regimes
- Hypothesis: the same biological modules drive both source and target split signals in robust layers.
- Experiment: for each robust layer, run GO/STRING on top genes per split and compute overlap versus random gene-set null.
- Success criterion: overlap metrics exceed null and immune-related terms are reproducible across splits.
- value: medium
- cost: medium

### Controls / Null Stress-Test Family

#### N33 — Bridge-ablation control isolates null-construction artifact
- Hypothesis: component bridging is inflating rewired geodesic null persistence and causing uniform failure.
- Experiment: stratify by-seed/layer rows into bridge-free vs bridged regimes using a rerun that intentionally produces both strata; compare null means/deltas and p-values.
- Success criterion: bridge-free strata show materially less negative rewiring deltas and non-degenerate p-value distribution.
- value: high
- cost: medium

## Priority Signal
- Highest immediate expected value: `N23`, `N24`, `N31`.
- Highest novelty but heavier lift: `N29`, `N30`.
- Fastest low-cost discriminator: `N27`.
