# Brainstormer Hypothesis Roadmap - iter_0048

## Retire / Deprioritize
1. Direct cross-model transfer utility endpoints (`H109/H119/H122/H125`) -> `retire_now`.
Reason: four consecutive negative outcomes in recent loop with domain-level null-gap failure.

2. Additive graph-topology surrogate endpoints (`H95/H97/H128`) -> `retire_now`.
Reason: repeated weak directional signal and strict-null failure under rewiring/feature shuffles.

3. Coarse finite-state grammar endpoints (`H104/H107/H111/H112`) -> `retire_now`.
Reason: rescue attempts consumed; still no robust null-gap survival.

4. Standalone additive intrinsic-dimension endpoints (`H98/H114`) -> `retire_now`.
Reason: repeated near-zero utility; low expected rescue yield.

5. Scalar additive PH rescue endpoints (`H103/H106/H113`) -> `retire_now`.
Reason: repeated negative/inverted effects across filtration variants.

6. Torsion-only manifold continuation (`H126/H129` as currently parameterized) -> `rescue_once_with_major_change`.
Constraint: no more scale-only tuning; next attempt must change representation/objective.

7. Signed motif-community lineage (`H116/H118/H123/H124/H127`) -> `keep_active_high_priority`.
Reason: strongest directional branch with identifiable hard slices (`lung/dual_axis_disjoint`, `immune/source_disjoint`).

## New Hypothesis Portfolio
| ID | Category | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N645` | Topology (bifiltration PH) | True regulatory edges occupy stable ridges in a signed-support x geodesic-radius bifiltration. | Build 2-parameter filtrations on edge neighborhoods (layers `7,11`), extract ridge mass/landscape image features, add over `H70`. | Positive mean null-gap in `>=3/6` domain-splits, including at least one target-disjoint split. | Filtration-axis permutation within degree bins, sign shuffle within TF strata, label permutation. | high | high |
| `N646` | Topology (Mapper dynamics) | Positives follow lower-entropy Mapper component transitions across depth than negatives. | Construct Mapper graphs at layers `0,3,7,11`; compute transition entropy, branch persistence, and component-stability features per edge. | Positive null-gap in `>=2/6` and stronger effect in external-lung. | Lens permutation, component relabel shuffle, label permutation. | medium | medium |
| `N647` | Topology (directed path homology v4) | Directed sign-weighted path homology captures causal asymmetry missed by undirected surrogates. | Build directed local complexes around source-target neighborhoods; compute path-homology Betti/entropy descriptors; add over `H70`. | Positive mean null-gap in `>=3/6`, with at least one source-disjoint pass. | Direction reversal, degree-preserving orientation rewiring, label permutation. | high | medium |
| `N648` | Topology stability (stability selection) | Null-robust topology signal improves when only bootstrap-stable topological descriptors are retained. | Bootstrap edge samples, run stability selection on topological features, then fit selected-only additive model vs `H70`. | Hard-slice null-gap increase over `H127`; positive null-gap in `>=4/9` for module-linked variants. | Bootstrap-index shuffle, selection on label-shuffled data, label permutation. | high | medium |
| `N649` | Manifold geometry (discrete Ricci asymmetry) | Positives show systematic source-target asymmetry in discrete Ricci flow along geodesics. | Compute Ollivier/Forman curvature summaries along directed paths; model asymmetry moments as features over `H70`. | Positive null-gap in `>=2/6`, especially source-disjoint. | Path reversal, transport-plan shuffle within degree bins, label permutation. | medium | medium |
| `N650` | Manifold geometry (chart-fracture index) | Negatives require more local chart-fracture events along geodesic traversal than positives. | Fit local PCA charts along each path and count high-angle chart breaks; use fracture count/density features over `H70`. | Positive mean delta in `>=4/6` with null-gap in `>=2/6`. | Chart-basis random rotation, path-order shuffle, label permutation. | medium | low |
| `N651` | Manifold geometry + ID interaction | Intrinsic-dimension gradients are predictive only when coupled to torsion sign changes. | Compute TWO-NN ID profile along paths and interaction terms with torsion sign-flip features; interaction-only model over `H70`. | Positive null-gap in `>=2/6`; recovery on immune/source-disjoint. | ID profile permutation along path, torsion-sign randomization, label permutation. | medium | medium |
| `N652` | Cross-model topology transfer | Cross-model random-walk persistence signatures align even when edge-level transfer fails. | Build anchor-constrained random-walk complexes in both models, compare persistence images with sliced Wasserstein similarity. | Positive domain null-gap in `>=2/3` and immune non-negative. | Anchor remap preserving degree, walk-order shuffle, label permutation. | high | high |
| `N653` | Cross-model chart/sheaf alignment | Cross-model local chart alignment with sheaf consistency recovers robust shared structure. | Partition by cell-ontology chart, fit per-chart orthogonal maps, compute sheaf cycle inconsistency and transfer utility diagnostics. | Positive domain null-gap in `>=2/3` with improved immune behavior over `H125`. | Chart-label permutation, cycle-order shuffle, random-map baseline, label permutation. | high | high |
| `N654` | Cross-model anomaly concordance (cheap) | Transfer succeeds only where both models agree on topological anomaly ranks. | Compute per-model anomaly ranks (PH/geometry residuals), use concordance interaction features for edge classification. | Positive domain null-gap in `>=1/3` in pilot; if none, retire quickly. | Rank permutation within domain, model-swap baseline, label permutation. | medium | low |
| `N655` | Biological anchoring (ontology barriers) | True edges cross low-energy cell-ontology boundaries in a structured way that fixes hard slices. | Add boundary crossing count/dwell/barrier energy features to `H127` residuals at layer `11` across all disjoint splits. | `lung/dual_axis_disjoint` null-gap becomes `>0`; `>=4/9` domain-splits null-positive. | Ontology-label permutation preserving sizes, boundary-depth shuffle, label permutation. | high | medium |
| `N656` | Biological anchoring (continuous GO-STRING semantics) | Continuous GO semantic similarity interacting with STRING confidence is the missing robustness term in `H127`. | Replace binary GO co-membership with semantic-distance features and interaction terms with sign/community/STRING in the H127 stack. | Positive mean null-gap in `>=6/9`; flip `lung/dual_axis_disjoint` and `immune/source_disjoint` to non-negative. | GO-graph rewiring within depth bins, STRING-bin permutation, TF-sign shuffle, label permutation. | high | medium |
| `N657` | Biological + mechanistic sequence motifs | Persistent signed TF-target motif trajectories across layers encode real regulatory programs. | Build motif-transition sequences over layers `0,3,7,11`; use n-gram/HMM likelihood features over `H127` residuals. | Positive null-gap in `>=3/6` source/target splits and improvement over prior FSM line. | Frequency-matched motif-token shuffle, layer-order permutation, label permutation. | medium | medium |
| `N658` | Algorithmic signatures (counterfactual automata) | Positives require larger minimal edit cost to transform into null-like trajectory automata than negatives. | Fit compact automaton on positive trajectories; compute counterfactual edit distance features and add over baseline. | Positive null-gap in `>=2/6` with strongest effect in external-lung. | State-label permutation, transition rewiring preserving out-degree, label permutation. | medium | medium |
| `N659` | Algorithmic signatures (description length, cheap) | Positives have lower description length for multi-layer geometric descriptor streams than negatives. | Encode layer-wise descriptor tokens with compression models and use compression ratio/cross-entropy deltas as features. | Positive mean delta in `>=4/6`, null-gap in `>=2/6` in pilot. | Token shuffle within layer bins, random codebook baseline, label permutation. | medium | low |
| `N660` | Topology + biological anchors (witness complexes) | Anchor-based witness complexes around high-confidence TFs produce robust edge-local topological evidence. | Build witness complexes using TRRUST A/B TF landmarks at layer `11`; extract Betti/lifetime descriptors per edge neighborhood. | Positive null-gap in `>=4/9` when merged into `H127` hard-slice packet. | Landmark swap within degree strata, witness assignment shuffle, label permutation. | high | medium |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N656` (continuous GO-STRING semantic hardening of `H127`).
Why: strongest live lineage, direct attack on known hard slices, moderate implementation cost.
Keep gate: positive mean null-gap `>=6/9`, `lung/dual_axis_disjoint > 0`, `immune/source_disjoint >= 0`.

2. High-risk/high-reward candidate: `N653` (cross-model chart/sheaf consistency alignment).
Why: genuine objective shift from repeated failed transfer AUROC endpoints.
Keep gate: positive domain null-gap in `>=2/3` and immune domain null-gap non-negative.

3. Cheap broad-screen candidate: `N650` (local chart-fracture manifold diagnostic).
Why: low-cost geometry screen with clear mechanistic readout and immediate pruning value.
Keep gate: positive mean delta `>=4/6` and positive mean null-gap `>=2/6`.
