# Brainstormer Next Iteration Brief - iter_0021

## Objective
Exploit the new `H47` signal to decision grade while opening one new topology family and one cheap cross-model screen that fixes the known sparsity failure in `H48`.

## Required Execution Packet (3 slots)

### Slot A - High-probability discovery (`N212` -> proposed `H49`)
- Hypothesis: bifiltration gain is multi-seed stable and utility-linked.
- Scope: domains `immune/lung/external_lung`, seeds `42/43/44`, splits `source_disjoint/target_disjoint`, layers `0,3,7,11`.
- Required outputs:
  - `iterations/iter_0022/h49_bifiltration_multiseed_by_seed_layer_split.csv`
  - `iterations/iter_0022/h49_bifiltration_multiseed_domain_summary.csv`
  - `iterations/iter_0022/h49_bifiltration_multiseed_null_summary.csv`
- Null/control: support shuffle within distance strata, distance-only ablation, plus layer-order permutation placebo.

### Slot B - High-risk/high-reward (`N213` -> proposed `H50`)
- Hypothesis: directed/signed persistence captures regulatory asymmetry missed by undirected topology.
- Scope: seed42 pilot first (`3` domains, both disjoint splits, layers `7,11`), expand only if directional.
- Required outputs:
  - `iterations/iter_0022/h50_directed_signed_topology_by_domain_layer_split.csv`
  - `iterations/iter_0022/h50_directed_signed_topology_domain_summary.csv`
  - `iterations/iter_0022/h50_directed_signed_topology_null_summary.csv`
- Null/control: orientation-preserving degree null and sign-shuffle null.

### Slot C - Cheap broad-screen (`N224` -> proposed `H51`)
- Hypothesis: expanded motif fingerprints become cross-domain once sparse-regime failure is addressed.
- Scope: motif panel `{FFL, bifan, feedback triad, feedforward chain, multi-input}` with `k={100,200,400}`, layers `7,11`, and module-collapsed variant.
- Required outputs:
  - `iterations/iter_0022/h51_cross_model_motif_fingerprint_by_domain_layer_k.csv`
  - `iterations/iter_0022/h51_cross_model_motif_fingerprint_summary.csv`
  - `iterations/iter_0022/h51_cross_model_motif_fingerprint_null_summary.csv`
- Null/control: degree-preserving rewiring and module-membership shuffle.

## Promotion / Continuation Gates
1. `H49` promote gate: positive `delta_AUROC(bif-distance)` in `>=70%` rows, Fisher-significant in `>=4/6` domain-splits, and positive utility coupling in `>=2/3` domains.
2. `H50` continue gate: directional improvement over undirected baseline in `>=2` domain-splits with `p<0.05` in at least one domain.
3. `H51` keep gate: enrichment significant in `>=2` domains and non-degenerate null variance in `>=80%` rows.

## Stop / Retire Rules
- Do not allocate slots to rewiring-survival null lineage, GW-first correspondence recovery, raw curvature/thinness direct-score reruns, or `H46`-style weighted-threshold zigzag reruns.
- If `H50` misses directional gate in pilot, stop immediately and reallocate next loop to `N220` (cross-model persistence-image alignment).

## Minimal Recovery Plan (only if next executor gate unexpectedly fails)
1. Run a minimal `H51` packet (`immune`, layer `11`, `k=200`) to guarantee machine non-narrative output quickly.
2. Run reduced `H49` (`seed42`, all domains, layers `0,11`) to restore a valid comparative experiment with clear null controls.
3. Emit required report triad (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, at least one summary CSV) before expanding scope.
