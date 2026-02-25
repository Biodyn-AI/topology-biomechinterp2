# Brainstormer Hypothesis Roadmap: iter_0051

## Retire / Deprioritize
| Direction | Status | Why now |
|---|---|---|
| Cross-model descriptor alignment branch (latest: `H137`) | `retire_now` | Multiple major resets still fail strict-null domain gates; latest run is `0/3` domain null-gap support |
| Additive ontology/module hardening branch in current form (`H124/H127/H130/H135/H138`) | `rescue_once_with_major_change` | Directionality remains high but strict-null survival is repeatedly absent on hard slices |
| Standalone ID additive uplift formulations (`H98`, `H134` style) | `retire_now` | Recurrent directional-only or negative behavior with no null survival |
| Rank-surface persistent surrogate add-ons (`H133` style) | `retire_now` | Decisive recent negatives and low rescue potential without changing filtration objective |
| Small manifold descriptor tweaks that optimize AUROC only (`H132` style) | `rescue_once_with_major_change` | Permit one mechanism-driven reset focused on null survival, then stop |

## New Hypothesis Portfolio
| ID | Family | Hypothesis (one sentence) | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| P01 | manifold_distance | Null-surviving anisotropy signal is real and concentrated in late-depth hard slices, not a seed artifact. | Extend `H136` to seeds `{42,43,44}`, splits `{source,target,dual_axis}`, layers `{7,11}` with same feature set plus hard-slice reporting. | `>=3/9` domain-splits with positive mean null-gap and at least one hard slice non-negative. | Endpoint-swap, tangent-rotation, label permutation, plus anisotropy-column shuffle. | high | medium |
| P02 | persistent_homology | A distance x support-margin bifiltration captures robust topology differences missed by one-axis summaries. | Build coarse Betti surfaces (e.g., 8x8 grid) per row and test additive utility vs `H70` at layers `{7,11}`. | Positive mean null-gap in `>=3/6` domain-splits and stronger effect on hard slices. | Margin shuffle, distance-quantile shuffle, label permutation. | high | high |
| P03 | topology_stability | Split-transition zigzag persistence contains regulatory signal even when static persistence fails. | Construct source->target->dual-axis zigzag complexes and score long-bar mass + birth-depth entropy against `H70`. | Positive null-gap in at least 2 domains and non-negative immune aggregate. | Split-order permutation, degree-bin node relabel, label shuffle. | medium | high |
| P04 | persistent_homology | Edge-level persistent entropy slope across depth is a stable mechanistic signature of positives. | Compute entropy trajectory over layers `{0,3,7,11}` and add slope/curvature features to `H70` (seed42 broad screen). | Positive mean delta in `>=4/6` domain-splits and at least `1/6` positive mean null-gap. | Layer-order permutation, entropy-sign randomization, label shuffle. | medium | low |
| P05 | graph_topology | Agreement between curvature contraction and anisotropy contraction marks true regulatory edges. | Compute Ollivier/Forman curvature features and interactions with `H136` anisotropy gap at layers `{7,11}`. | Positive interaction coefficient with null-gap support in `>=2/6` domain-splits. | Curvature shuffle within degree bins, label shuffle. | medium | medium |
| P06 | manifold_distance | Positive edges lie on low-detour geodesic routes through motif-community hubs. | Define hub cores from signed motif-community graph and test detour-to-hub features over `H70`. | Negative detour coefficient and domain-split positive lift in at least half of splits. | Hub-label permutation, endpoint-swap path control, label shuffle. | medium | low |
| P07 | manifold_distance | Regulatory positives are enriched at boundaries between local linear patches with controlled shear. | Fit local PCA patches per node and add patch-transition/shear features. | Better separation than `H134` with at least `2/6` positive mean null-gap domain-splits. | Patch-assignment shuffle within density strata, label shuffle. | medium | medium |
| P08 | intrinsic_dimensionality | Decreasing intrinsic-dimension flux along edge geodesics is a true mechanism rather than a noisy additive feature. | Estimate per-layer ID and compute pathwise flux-cost features with monotone constraints on top of `H70`. | Negative flux-cost coefficient and non-zero null survival (`>=2/6` splits). | Layer-order permutation, matched-length path randomization, label shuffle. | medium | medium |
| P09 | cross_model_alignment | TF-anchored persistence images align across scGPT/Geneformer when the shared biological topology is real. | Build TRRUST/GO module persistence images in both models and evaluate OT alignment + held-out transfer utility. | Positive domain null-gap in `>=2/3` and immune non-negative. | Anchor shuffle, module permutation, depth scramble. | high | high |
| P10 | cross_model_alignment | Perturbation-coupled anchors reveal a shared subspace missed by correspondence-free descriptors. | Use perturbation-informed anchor matrices for CKA/Procrustes fit and test downstream transfer on held-out domain. | Substantial fit gap over random anchors and non-negative transfer in at least one non-immune domain. | Anchor-label permutation, random-teacher control, label shuffle. | high | high |
| P11 | module_structure | Causal chart assignments (not descriptive GO charts) are required for sheaf-obstruction features to survive nulls. | Rebuild sheaf charts from perturbation-response clusters or signed causal simulations and rerun hard slices first. | Hard slices (`immune/source`, `lung/dual_axis`) become non-negative in null-gap summary. | Chart relabel, response-cluster shuffle, label permutation. | high | high |
| P12 | biological_anchor | Multi-source support concordance (TRRUST+STRING+GO) should predict persistence survival after degree control. | Fit stratified survival-style model of persistence lifetime vs support tier across all domains/splits. | Positive support-tier coefficient with `>=3/6` positive null-gap domain-splits. | Tier permutation within degree bins, label shuffle. | medium | medium |
| P13 | biological_anchor | Conditioning by cell ontology neighborhoods amplifies topology signal hidden in pooled-domain analysis. | If cell-type labels are available, run per-ontology subgraph screens for `H136`-style features and meta-analyze. | At least one ontology shows stronger null-gap than pooled domain and replicates across seeds. | Ontology-label shuffle within domain. | medium | medium |
| P14 | mechanistic_motif | Positives are enriched for recurrent depth motifs (expand->contract->stabilize) in geometric trajectories. | Tokenize per-edge depth trajectories (anisotropy gap, support margin, detour) and test motif automaton enrichment. | Enriched motif counts in positives across at least two domains with null support. | Token-order permutation, motif-dictionary shuffle, label shuffle. | medium | low |

## Top 3 for Immediate Execution
1. **High-probability discovery candidate: `P01` (H136 robustness expansion).**
- Why: only branch with current strict-null survival; direct path to confirm or kill quickly.
- Keep gate: `>=3/9` positive mean null-gap domain-splits and at least one hard slice non-negative.

2. **High-risk/high-reward candidate: `P09` (TF-anchored cross-model persistence-image OT).**
- Why: materially different representational object; could reopen cross-model family with a falsifiable, biology-anchored endpoint.
- Fast-fail gate: if `0/3` domain null-gap positive in pilot, retire immediately.

3. **Cheap broad-screen candidate: `P14` (motif automaton recurrence).**
- Why: low implementation cost, broad geometric coverage, and orthogonal to recent additive feature failures.
- Keep gate: positive mean delta in `>=4/6` domain-splits and at least one positive mean null-gap split.
