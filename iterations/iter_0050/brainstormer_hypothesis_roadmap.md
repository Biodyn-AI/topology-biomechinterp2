# Brainstormer Hypothesis Roadmap - iter_0050

## Retire / Deprioritize
1. Rank-surface persistence surrogates (`H67`, `H133`) -> `retire_now`.
Reason: repeated full failure under direction and null-gaps.

2. Standalone intrinsic-dimension additive endpoints (`H89`, `H98`, `H114`, `H134`) -> `retire_now`.
Reason: repeated directional-only behavior with no strict-null survival.

3. H130-style semantic hardening endpoint (`H130`, `H135`) -> `retire_now`.
Reason: hard-slice null failures persist after targeted reruns and larger null budgets.

4. Cross-model map-learning transfer endpoints (`H119`, `H122`, `H125`, `H131`) -> `retire_now`.
Reason: repeated objective resets with low rescue potential.

5. Path-additive manifold rescues (`H129`, `H132`) -> `rescue_once_with_major_change`.
Constraint: next attempt must be interaction-based or topology-coupled, not additive path descriptors.

6. Module-structure branch overall (`H123` lineage) -> `rescue_once_with_major_change`.
Constraint: restart from H123-strength core and target hard slices with new mechanism + adversarial nulls.

## New Hypothesis Portfolio
| ID | Area | One-sentence hypothesis | Concrete test design | Expected signal if true | Null / control | Value | Cost |
|---|---|---|---|---|---|---|---|
| `N675` | Topology (witness PH) | Hard-slice positives lie on stable local 1-cycles in TF-landmark witness complexes. | Build TF-anchored witness complexes at layer `11` on `immune/source` and `lung/dual_axis`, extract long-bar mass and cocycle concentration, add over H123 baseline. | Hard slices become non-negative null-gap and at least `3/4` hard-slice groups pass. | Landmark identity shuffle in TF-degree bins, witness assignment permutation, label permutation. | high | medium |
| `N676` | Topology (bifiltration) | A geodesic-distance x GO-semantic-residual bifiltration contains discriminative ridge persistence missed by current single-axis features. | Compute two-parameter persistence surfaces at layers `7,11`, derive ridge area/curvature stats, evaluate over H70/H123. | Positive mean null-gap in `>=3/6` domain-splits. | Axis permutation within strata, GO-depth-preserving semantic shuffle, label permutation. | high | high |
| `N677` | Topology (persistent cohomology) | Positive edges show consistent cocycle orientation across depth transitions while negatives decorrelate. | Compute representative H1 cocycles on directed kNN complexes across `0->3->7->11`; score orientation coherence features. | Positive orientation-coherence null-gap in `>=4/6` domain-splits. | Direction randomization preserving degree, cycle-basis permutation, label permutation. | medium | medium |
| `N678` | Topology stability (zigzag turnover) | True edges have lower barcode turnover across source->union<-target zigzag transitions than negatives. | Run weighted zigzag on source/target complexes; extract turnover rate, birth-death drift, and long-bar retention features. | Positive mean delta and positive mean null-gap in `>=2/6` domain-splits. | Transition-order shuffle, target-set shuffle in size bins, label permutation. | medium | high |
| `N679` | Topology (Mapper motifs) | Positive edges concentrate in Mapper regions with stable loop-rich topology under diffusion lenses. | Build Mapper graph with diffusion and support lenses; use node loop entropy and co-membership as edge features. | Positive mean delta in `>=4/6` and null-gap in `>=2/6` domain-splits. | Lens-value shuffle in degree bins, cluster relabeling, label permutation. | medium | medium |
| `N680` | Manifold geometry (sectional anisotropy) | Positives show stronger source-target sectional-curvature anisotropy than negatives. | Estimate local tangent 2-plane curvature at endpoints for layers `7,11`; add anisotropy ratios over H70. | Positive null-gap in `>=2/6` splits, strongest in source-disjoint. | Endpoint swap in distance bins, tangent-basis random rotation, label permutation. | medium | low |
| `N681` | Manifold geometry (geodesic deviation) | Positive edges are more stable to path perturbations, with lower geodesic deviation growth. | Perturb intermediate nodes on directed geodesic paths; compute deviation growth exponent and dispersion features. | Positive mean delta in `>=5/6`, null-gap in `>=2/6`. | Path-order shuffle, perturbation schedule permutation, label permutation. | medium | medium |
| `N682` | Manifold geometry (subspace transport) | True edges preserve local linear subspaces better across layers than negatives. | Compute principal-angle transport costs for local subspaces across `0,3,7,11`; use max-jump and cumulative transport features. | Positive mean null-gap in `>=2/6` domain-splits. | Layer-order permutation, basis shuffle, label permutation. | medium | medium |
| `N683` | Intrinsic dimension (interaction-only) | ID features become useful only as interactions with motif/community evidence, explaining prior standalone failures. | Fit interaction-only terms: `ID-gradient x sign-consistency x same-community` on top of H123, with no standalone ID main effects. | Hard-slice null-gaps improve by `>=+0.01` vs H135 and `>=2/4` hard-slice groups are positive. | Interaction-term permutation in strata, ID-profile shuffle, label permutation. | high | low |
| `N684` | Cross-model alignment (correspondence-free) | Cross-model persistence-kernel agreement predicts edge utility without gene-level maps. | Build persistence images for scGPT/Geneformer per domain/split/layer; compute kernel alignment and use as transfer priors. | Positive domain null-gap in `>=2/3`, immune non-negative. | Cross-model pairing shuffle, kernel-spectrum permutation, label permutation. | high | high |
| `N685` | Cross-model structure transport | OT between model-level persistence landscapes captures conserved regulatory geometry in at least one non-immune domain. | Compute Sinkhorn OT distance/plan sparsity between model landscape histograms per domain/split/layer as features. | At least `1/3` domains with positive null-gap and one target-disjoint positive split. | Domain swap, landscape-bin shuffle, random transport baseline, label permutation. | high | high |
| `N686` | Biological anchoring (cell ontology sheaf) | Hard-slice false positives are enriched for cell-ontology sheaf obstructions, and modeling these restores null robustness. | Build ontology charts, compute sheaf obstruction energy and compatibility with signed motif/community signals over H123. | `lung/dual_axis` and `immune/source` null-gaps become `>=0`; overall `>=4/9` positive domain-splits. | Chart relabel preserving chart size, section shuffle preserving degree, label permutation. | high | medium |
| `N687` | Biological anchoring (signed motif closure) | Signed feedback and FFL closure weighted by STRING confidence captures mechanistic constraints missing in H130-style semantics. | Compute signed motif closure and imbalance features in edge neighborhoods, integrate with H123 core on layer `11`. | Positive mean null-gap in `>=3/9` domain-splits and `lung/dual_axis > 0`. | Degree-matched motif rewiring, sign shuffle preserving TF sign rates, label permutation. | high | medium |
| `N688` | Algorithmic signatures (compressibility) | True edges have lower description length for multi-layer geometric-topological token streams than negatives. | Tokenize per-edge trajectories (support, triangle-defect, motif, curvature) over layers and compute MDL/compression-gap features. | Positive mean delta in `>=4/6` and null-gap in `>=2/6` domain-splits. | Token-order shuffle, random codebook baseline, label permutation. | medium | low |

## Top 3 for Immediate Execution
1. High-probability discovery candidate: `N686` (cell-ontology sheaf hard-slice rescue on H123 backbone).
Why: directly targets the persistent hard-slice failure mechanism with a materially different biological/topological signal, while staying on the only branch with prior strict-null success.
Keep gate: `lung/dual_axis_disjoint >= 0`, `immune/source_disjoint >= 0`, and `>=4/9` positive mean null-gap domain-splits.

2. High-risk/high-reward candidate: `N684` (correspondence-free cross-model persistence-kernel alignment).
Why: tests cross-model structure transfer without repeating failed map-learning objectives.
Keep gate: positive domain null-gap in `>=2/3` and immune domain non-negative.

3. Cheap broad-screen candidate: `N680` (sectional-curvature anisotropy screen).
Why: low implementation cost, genuinely new geometry family relative to recent additive torsion/ID path endpoints, high pruning value.
Keep gate: positive mean delta in `>=4/6` and positive mean null-gap in `>=2/6` domain-splits.
