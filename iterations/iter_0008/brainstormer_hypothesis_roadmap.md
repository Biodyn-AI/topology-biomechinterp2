# Brainstormer Hypothesis Roadmap — iter_0008

## Candidate Next Hypotheses (N44-N53)

### Topology

#### N44 — Bridge-identifiable causal test with fixed-k factorial design
- family: `topology`
- Hypothesis: the apparent bridge effect is mainly split/k confounding; within-split bridge contrasts will shrink toward zero once k is controlled.
- Experiment: for each split, force two k schedules (low and high) that produce both bridged and non-bridged rows; compute within-split bridged-minus-nonbridged H1 delta.
- Success criterion: bridge gap changes sign or collapses in magnitude after within-split control.
- value: high
- cost: medium

#### N45 — Layer-wise k phase transition in rewiring survival
- family: `topology`
- Hypothesis: rewiring negativity has a k-threshold (phase transition), not a smooth monotonic trend.
- Experiment: run k sweep (for example 12, 18, 24, 30, 36, 40) on layers `7,9,10,11`; estimate breakpoint in H1 delta vs k.
- Success criterion: detect a reproducible threshold where deltas sharply worsen.
- value: medium
- cost: medium

### Manifold Geometry

#### N46 — Distortion-tail concentration predicts rewiring failure better than mean distortion
- family: `manifold_geometry`
- Hypothesis: failure is driven by upper-tail geodesic distortion, which is masked by mean distortion summaries.
- Experiment: compute q90/q95 distortion deltas per row and regress against H1 deltas with split and layer fixed effects.
- Success criterion: tail distortion terms explain significantly more variance than mean distortion.
- value: medium
- cost: low

#### N47 — Persistence-shape rescue in late layers despite negative lifetime-sum totals
- family: `manifold_geometry`
- Hypothesis: late layers retain long-cycle structure even when total H1 lifetime delta is negative.
- Experiment: compare max lifetime, top-3 lifetime share, and persistence entropy between observed and rewired nulls.
- Success criterion: at least one shape descriptor is positive/significant in late layers (`7,9,10,11`).
- value: medium
- cost: medium

### Cross-Model Alignment

#### N48 — Matched-gene late-layer topology rank concordance (scGPT vs Geneformer)
- family: `cross_model_alignment`
- Hypothesis: late-layer topology ordering is shared across models when matched genes and equivalent nulls are used.
- Experiment: compute per-layer feature-shuffle (and optionally rewiring) topology effect sizes in Geneformer and test Spearman concordance with scGPT.
- Success criterion: `rho > 0.5` with permutation `p < 0.05`.
- value: high
- cost: high

#### N49 — Null-family sensitivity agreement across models
- family: `cross_model_alignment`
- Hypothesis: model agreement is stronger for null-sensitivity profiles than for raw topology magnitudes.
- Experiment: for each layer, compare effect-size drop from feature-shuffle to rewiring null in both models.
- Success criterion: positive and significant cross-model correlation of null-sensitivity drops.
- value: medium
- cost: high

### Biological Anchoring

#### N50 — Robust-vs-fragile cycle genes encode distinct immune regulatory programs
- family: `biological_anchoring`
- Hypothesis: genes dominating robust feature-shuffle layers differ biologically from genes dominating rewiring-fragile layers.
- Experiment: extract top cycle-contributing genes for robust (`7,9,10,11`) and fragile layers; run TRRUST/DoRothEA/GO enrichment and overlap tests.
- Success criterion: robust layers show coherent immune TF/pathway enrichments that are absent or weaker in fragile controls.
- value: high
- cost: medium

#### N51 — Bridge-sensitive rows enrich for sampling/connectivity artifacts rather than immune programs
- family: `biological_anchoring`
- Hypothesis: bridge-heavy rows are enriched for generic/high-degree artifacts, not specific immune regulatory biology.
- Experiment: compare bridged vs non-bridged row gene sets for housekeeping, ribosomal, mitochondrial, and immune pathway enrichment.
- Success criterion: bridged-only signals are artifact-skewed while non-bridged retain immune specificity.
- value: medium
- cost: medium

### Controls / Null Stress Tests

#### N52 — Exact degree+quantile constrained rewiring (MCMC) vs best-of-candidates approximation
- family: `controls_null_stress`
- Hypothesis: current best-of-16 constrained rewiring is too approximate; exact constrained rewiring may materially alter null calibration.
- Experiment: implement/borrow an exact (or near-exact) Markov-chain swap preserving degree and edge-length quantile bins; rerun on a reduced layer packet first.
- Success criterion: materially lower edge-hist L1 drift and changed H1 conclusions in at least one layer-split.
- value: high
- cost: high

#### N53 — Power/stability stress test for null-draw count
- family: `controls_null_stress`
- Hypothesis: `n_null=5` is underpowered/noisy for tail p-value behavior and may obscure weak rescue effects.
- Experiment: rerun selected layers with null draws `{5, 20, 50}` and estimate confidence intervals for H1 delta and Fisher p.
- Success criterion: conclusions stay stable across draw counts, or a hidden weak effect appears with tighter uncertainty.
- value: medium
- cost: low

## Priority Signal
1. Highest immediate decision value: `N44`, `N50`, `N53`.
2. Highest potential upside but heavier lift: `N48`, `N52`.
3. Good low-cost discriminators when time-boxed: `N46`, `N53`.
