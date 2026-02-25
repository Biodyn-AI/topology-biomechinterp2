# Brainstormer Hypothesis Roadmap - iter_0046

## Retire / Deprioritize
1. Unanchored cross-model transport/gating endpoints (`H96/H99/H102/H109/H119/H122`) -> `retire_now`.
Evidence: repeated null-gap failures, including `H122` at `0/6` positive mean null-gap domain-splits.

2. Additive PH scalar rescue lineage (`H100/H101/H103/H106/H110/H113`) -> `retire_now`.
Evidence: repeated near-zero/negative robustness under multiple null designs.

3. Standalone additive intrinsic-dimension lineage (`H98/H114`) -> `retire_now`.
Evidence: no robust promotion signal after multiple attempts.

4. Coarse grammar/state discretization lineage (`H104/H107/H111/H112`) -> `rescue_once_with_major_change`.
Required major change: biologically anchored state definitions + occupancy/dwell-time matched nulls.

5. Pooled curvature-drift endpoint (`H120`) -> `retire_now`.
Evidence: superseded by directional asymmetry branch (`H121`) with cleaner interpretability.

6. Directional asymmetry branch (`H121`) -> `rescue_once_with_major_change`.
Required major change: source-disjoint-focused redesign with explicit layer-conditioned asymmetry terms and stronger null resolution.

7. Signed motif-community branch (`H118/H123`) -> `prioritize_hardening`.
Reason: strongest active signal with strict-null robustness; remaining gap is coverage (`lung/dual_axis_disjoint`).

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N615` | topology (bifiltration PH) | True regulatory edges occupy stable high-persistence ridges in a signed-support x geodesic-radius bifiltration. | Build 2-parameter local complexes per edge at layers `{7,11}`, extract Hilbert-surface summaries and ridge-mass features, and test additive lift over `H70` and over `H123` residuals. | Positive mean null-gap in `>=4/9` domain-splits with better dual-axis behavior than one-parameter PH features. | Axis permutation within degree bins, ridge-surface shuffle, label permutation. | high | high |
| `N616` | topology (cohomology localization) | Long-lived cocycles localize near TRRUST/GO-supported motif modules for true edges more than negatives. | Compute representative cocycles for dominant classes, quantify cocycle-support overlap with motif modules, and add overlap interactions to the classifier. | Positive overlap interaction and positive mean null-gap in `>=3/6` domain-splits. | Module-membership shuffle (size/degree matched), cocycle-support shuffle, label permutation. | medium | high |
| `N617` | topology (Mapper graph geometry) | Mapper branch-loop imbalance of local edge neighborhoods distinguishes causal-looking edges from spurious ones. | Build Mapper graphs using lens pair `(geodesic eccentricity, motif support)`, extract loop/branch statistics per edge-neighborhood, and evaluate over `H70`. | Positive mean null-gap in `>=3/6` with strongest gains in source-disjoint splits. | Lens-value permutation within degree bins, cover-overlap randomization, label permutation. | medium | medium |
| `N618` | topology stability (targeted dropout) | Positive edges have slower topological degradation under support-stratified node dropout than negatives. | Apply stratified dropout schedules to local graphs, fit persistence-mass decay slopes/curvature, and test slope features over baseline. | Flatter degradation for positives and positive mean null-gap in `>=3/6`. | Degree-matched random dropout, dropout-order permutation, label permutation. | medium | medium |
| `N619` | manifold geometry (multi-scale curvature) | Regulatory edges show cross-scale curvature anisotropy signatures that are stable across domains. | Estimate directed Ollivier/Forman curvature at radii `{10,20,30}`, derive anisotropy and volatility descriptors, and evaluate across splits. | Positive mean null-gap in `>=3/6` and consistent anisotropy direction in at least two domains. | Neighborhood shuffle within degree bins, radius-order permutation, label permutation. | medium | medium |
| `N620` | manifold geometry (geodesic torsion) | Source->target geodesics for true edges have distinct turning-angle/torsion spectra compared with negatives. | Reuse shortest-path infrastructure to compute directional turning-angle spectra and torsion proxies at layers `{7,11}`, then screen over `H70`. | Positive mean null-gap in `>=2/6` with at least one source-disjoint split clearly positive. | Path-reversal control within length bins, endpoint swap within distance bins, label permutation. | medium | low |
| `N621` | manifold geometry (local linearity breakpoint) | True edges cross smaller endpoint rank-jump discontinuities in local tangent dimension than negatives. | Estimate local PCA rank around source/target neighborhoods over expanding radii and use rank-jump mismatch features in the edge model. | Lower rank-jump mismatch for positives and positive mean null-gap in `>=3/6`. | Endpoint-swap control, radius-order permutation, label permutation. | medium | low |
| `N622` | cross-model alignment (anchor-constrained cycle consistency) | Cross-model transfer becomes robust if correspondence is learned with TRRUST/GO anchor constraints and cycle consistency. | Learn scGPT<->Geneformer bi-directional maps with anchor penalties and cycle-consistency loss, then evaluate transfer lift in held-out splits/domains. | Positive domain null-gap in `>=2/3` domains with immune non-negative. | Anchor-label permutation preserving sizes, random-map baseline, label permutation. | high | high |
| `N623` | cross-model alignment (perturbation-cone concordance) | True edges preserve perturbation-response ordering across models within local influence cones better than negatives. | Compute edge-local perturbation cones in each model, score rank concordance features, and test transfer utility over `H70`. | Positive domain null-gap in `>=2/3` with reduced immune failure versus prior transport metrics. | Cone-membership shuffle degree-matched, response-rank permutation, label permutation. | high | medium |
| `N624` | cross-model alignment (chart-wise transfer) | Partitioning by cell ontology and aligning local charts yields more stable cross-model transfer than global alignment. | Build ontology-partitioned local Procrustes/OT maps, aggregate chart residuals into edge features, and evaluate on held-out domains/splits. | Immune domain null-gap improves to non-negative while keeping lung/external_lung positive. | Ontology-label permutation, chart assignment shuffle, unconstrained global baseline. | high | medium |
| `N625` | biological anchoring (TRRUST+STRING hardening) | The `H123` signal strengthens further when signed motif-community terms are conditioned on STRING confidence and coverage is forced for missing splits. | Extend `H123` features with STRING-confidence interaction terms; run seeds `{42,43,44}` with forced `lung/dual_axis_disjoint` inclusion and full null budgets. | Positive mean null-gap in `>=8/9` domain-splits with explicit `lung/dual_axis_disjoint` row positive. | Existing strict null trio + STRING-bin permutation within degree strata. | high | medium |
| `N626` | biological anchoring (GO/cell-ontology barriers) | True regulatory edges show characteristic geodesic barrier signatures at GO/cell-ontology boundaries that negatives lack. | Define boundary crossing depth, curvature spike, and dwell features around GO/cell ontology transitions and test additive/interactions over `H70`. | Positive mean null-gap in `>=3/6`, especially in external_lung and lung. | Boundary-membership permutation (size matched), boundary-depth shuffle, label permutation. | medium | medium |
| `N627` | algorithmic signatures (motif automata) | Signed feed-forward/feedback motif automata with dwell-time and transition-entropy statistics encode true edge mechanisms. | Build state sequences per edge from sign-consistency, support, and community status across layers; extract dwell and entropy features and test over `H70`. | Positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6`. | Occupancy-matched state shuffle, layer-order permutation, label permutation. | high | low |
| `N628` | algorithmic signatures (resonance motif) | True edges exhibit low-frequency, phase-stable motif-support resonance across depth that negatives do not. | Compute depth-wise spectral signatures of motif-support trajectories around each edge and add frequency/phase-stability features. | Positive mean null-gap in `>=3/6` with strongest gain in source-disjoint slices. | Phase randomization preserving power spectrum, depth-order shuffle, label permutation. | medium | low |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N625` (TRRUST+STRING hardening of `H123` with forced coverage restoration).
Why: `H123` already has strict-null robustness (`8/8` observed domain-splits) and the only blocker is missing `lung/dual_axis_disjoint` coverage.
Keep gate: positive direction in `>=8/9` domain-splits, positive mean null-gap in `>=8/9`, and explicit pass for `lung/dual_axis_disjoint`.

2. High-risk/high-reward candidate: `N622` (anchor-constrained cycle-consistent cross-model alignment).
Why: this is a genuine objective shift from failed transport/gating endpoints and directly injects biological anchors before alignment.
Keep gate: positive domain null-gap in `>=2/3` domains and immune null-gap `>=0`.

3. Cheap broad-screen candidate: `N620` (geodesic torsion/turning-angle asymmetry).
Why: low implementation cost via existing path pipeline and directly targets `H121` source-disjoint fragility.
Keep gate: positive mean null-gap in `>=2/6` domain-splits with at least one source-disjoint domain-split positive.
