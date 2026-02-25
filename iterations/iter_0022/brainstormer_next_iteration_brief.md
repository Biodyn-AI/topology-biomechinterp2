# Brainstormer Next Iteration Brief - iter_0022

## Objective
Exploit `H50` to decision grade, open one genuinely new directed-topology axis, and run one cheap intrinsic-geometry screen that broadens mechanism coverage.

## Required Execution Packet (3 slots)

### Slot A - High-probability discovery (`N226` -> proposed `H52`)
- Hypothesis: directed/signed topology is multiseed robust and remains positive under stricter directed nulls.
- Scope: domains `immune/lung/external_lung`, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `0,3,7,11`.
- Required outputs:
  - `iterations/iter_0023/h52_directed_signed_multiseed_by_seed_layer_split.csv`
  - `iterations/iter_0023/h52_directed_signed_multiseed_domain_summary.csv`
  - `iterations/iter_0023/h52_directed_signed_multiseed_null_summary.csv`
- Null/control: degree-preserving orientation-constrained edge swaps, sign-shuffle, and split-label placebo.

### Slot B - High-risk/high-reward (`N230` -> proposed `H53`)
- Hypothesis: directed path-homology signatures capture mechanistic structure that improves utility and transfer.
- Scope: seed42 pilot first on all 3 domains, both splits, layers `7,11`; expand to multiseed only if pilot clears gate.
- Required outputs:
  - `iterations/iter_0023/h53_directed_path_homology_by_domain_layer_split.csv`
  - `iterations/iter_0023/h53_directed_path_homology_domain_summary.csv`
  - `iterations/iter_0023/h53_directed_path_homology_null_summary.csv`
- Null/control: directed degree-preserving rewiring, sign inversion/shuffle, and random-map transfer control.

### Slot C - Cheap broad-screen (`N233` -> proposed `H54`)
- Hypothesis: local linearity rupture index across depth is a robust edge-level mechanism signal.
- Scope: `3 domains x 3 seeds x 2 splits x layers {0,3,7,11}` using existing embedding pipeline.
- Required outputs:
  - `iterations/iter_0023/h54_linearity_rupture_by_seed_layer_split.csv`
  - `iterations/iter_0023/h54_linearity_rupture_domain_summary.csv`
  - `iterations/iter_0023/h54_linearity_rupture_null_summary.csv`
- Null/control: layer-order shuffle, endpoint swap within degree bins, and label permutation.

## Promotion / Continuation Gates
1. `H52` promote gate: `delta_AUROC>0` in `>=75%` rows, Fisher-significant in `>=4/6` domain-splits, and no domain-split with negative mean delta.
2. `H53` continue gate: significant utility or transfer lift in `>=2` domains under directed nulls.
3. `H54` keep gate: positive calibration or AUROC gain in `>=4/6` domain-splits, with at least one Fisher-significant domain.

## Stop / Retire Rules
- Do not allocate execution slots to rewiring-survival, GW-first correspondence recovery, raw curvature/thinness direct-score reruns, tier-monotonic concentration reruns, or motif-overlap-as-endpoint reruns.
- If `H53` fails pilot gate, stop immediately and replace next-loop high-risk slot with `N235` (cross-model persistence-image transfer).
- If `H52` fails multiseed robustness, do not spend another loop on directed/signed variants without a major null redesign.

## Minimal Recovery Plan (only if next executor gate unexpectedly fails)
1. Run reduced `H52` on seed42, all domains, layers `7,11`, to produce one valid comparative topology artifact quickly.
2. Run reduced `H54` on seed42, all domains, layers `11` only, to guarantee a second independent machine result.
3. Emit required report triad and one summary JSON/CSV before any expansion.
