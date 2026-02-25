# Brainstormer Hypothesis Roadmap — iter_0007

## Candidate Next Hypotheses (N34-N43)

### Topology

#### N34 — External-lung replication of metric-matched rewiring branch
- Hypothesis: rewiring-null failure after metric matching is immune-specific rather than domain-general.
- Experiment: rerun `iter_0007` protocol on external-lung (same seeds/layers/splits, same geodesic-vs-rewired-geodesic comparison).
- Success criterion: if external-lung shows `>=2/24` significant geodesic layer-split tests or substantially less-negative mean deltas than immune, the branch remains open cross-domain.
- value: high
- cost: medium

#### N35 — Bridge-conditioned rewiring effect explains most negativity
- Hypothesis: high-k + component bridging drives strongly negative rewiring deltas.
- Experiment: stratify by-seed rows into bridged vs non-bridged and rerun with constrained k schedules (for example fixed k grid below 30 where possible) to produce both strata intentionally.
- Success criterion: non-bridged rows show materially less-negative deltas and broader p-value spread than bridged rows.
- value: high
- cost: medium

#### N36 — Edge-length-bin-preserving rewiring rescues over-adversarial null behavior
- Hypothesis: degree-preserving rewiring is still too geometry-destructive; adding edge-length quantile constraints yields a better calibrated stronger null.
- Experiment: rewire only within edge-length bins (for example 5 quantiles), then rerun geodesic H1 and distortion tests.
- Success criterion: distortion delta moves toward 0 and at least a small subset of layer-split tests (`>=2/24`) becomes directionally positive or near-significant.
- value: high
- cost: medium

### Manifold Geometry

#### N37 — Local manifold distortion predicts feature-shuffle split robustness
- Hypothesis: layers with higher geodesic/euclidean distortion are the same layers that pass dual-split feature-shuffle robustness.
- Experiment: compute per-layer distortion descriptors (mean ratio, upper-tail ratio, graph efficiency) and regress against dual-split pass indicators from iter_0006.
- Success criterion: consistent positive association across seeds/splits with permutation-calibrated significance.
- value: medium
- cost: low

#### N38 — Barcode-shape descriptors outperform total H1 sum for robust-layer detection
- Hypothesis: late-layer robustness is driven by a few long-lived cycles, not just total H1 lifetime mass.
- Experiment: add barcode descriptors (max lifetime, top-3 lifetime share, entropy, Betti curve AUC) for observed and null runs.
- Success criterion: one descriptor separates robust layers (`7,9,10,11`) from non-robust layers better than H1-sum delta.
- value: medium
- cost: medium

### Cross-Model Alignment

#### N39 — Matched-gene residual topology ranks align between scGPT and Geneformer
- Hypothesis: layer-wise topological effect ranking is shared across models once matched residual embeddings are used.
- Experiment: compute per-layer feature-shuffle H1 deltas for Geneformer on matched genes and test rank alignment with scGPT using permutation null.
- Success criterion: Spearman `rho > 0.5` with permutation `p < 0.05` in at least one domain.
- value: high
- cost: high

#### N40 — Cross-model agreement is depth-localized to late layers
- Hypothesis: if alignment exists, it is strongest in deeper layers corresponding to scGPT robust layers.
- Experiment: compare early-vs-late layer persistence-image similarity across models with layer-shuffle null.
- Success criterion: late-layer similarity exceeds early-layer similarity under null calibration.
- value: medium
- cost: high

### Biological Anchoring

#### N41 — Immune robust layers enrich coherent TF regulatory programs
- Hypothesis: layers `7,9,10,11` capture regulatory programs rather than generic manifold artifacts.
- Experiment: derive cycle-contributing genes per layer/split and run TRRUST/DoRothEA enrichment with FDR control.
- Success criterion: at least two robust layers show reproducible immune TF enrichment across both splits.
- value: high
- cost: medium

#### N42 — Cross-split pathway overlap is high in robust layers and low in non-robust controls
- Hypothesis: robust topology corresponds to stable biological modules across source/target disjoint splits.
- Experiment: GO/Reactome enrichment on top cycle genes per split; quantify overlap against random gene-set null.
- Success criterion: robust layers show significantly higher cross-split overlap than matched non-robust layers.
- value: medium
- cost: medium

### Controls / Null Stress Tests

#### N43 — Negative-control manifold destruction should eliminate feature-shuffle signal
- Hypothesis: orthogonal-noise scrambling or gene-ID permutation destroys true topology signal and should collapse positive deltas.
- Experiment: apply controlled manifold-breaking transforms before persistence computation, then rerun feature-shuffle test on robust and non-robust layers.
- Success criterion: transformed data yields near-zero or negative deltas with loss of split-robust passes.
- value: medium
- cost: low

## Priority Signal
- Highest expected value now: `N35`, `N36`, `N41`.
- Fastest discriminator with low engineering lift: `N37`, `N43`.
- High upside but heavy dependency risk: `N39`, `N40`.
