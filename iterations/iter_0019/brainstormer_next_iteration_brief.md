# Brainstormer Next Iteration Brief - iter_0019 -> iter_0020

## Gate Status
- Current gate passed: `passed_min_research_gate=true`.
- Run a normal 3-slot experiment packet.

## Recommended 3-Hypothesis Packet

### 1) High-probability discovery slot
- Candidate: `N195` (STRING + ontology extension of `H40`).
- Goal: convert `H40` from promising to robust by adding missing biological priors and heterogeneity modeling.
- Minimal protocol:
1. Add true STRING edge-score prior to the existing continuous support score.
2. Fit mixed-effects interaction models with cell-ontology random effects across domains/splits/layers `{0,3,7,11}` and seeds `{42,43,44}`.
3. Calibrate with support permutations within `degree x coexpression x geodesic x ontology` strata.
- Required artifacts:
  - `h43_support_interaction_ontology_by_seed_layer_split.csv`
  - `h43_support_interaction_ontology_domain_summary.csv`
  - `h43_support_interaction_ontology_null_summary.csv`
- Promotion gate: interaction and top-decile uplift positive in `>=2/3` domains, plus lung no longer negative in both splits.

### 2) High-risk/high-reward slot
- Candidate: `N185` (true split-zigzag persistence).
- Goal: test split-invariant cycle structure with a real zigzag method instead of the `H41` proxy.
- Minimal protocol:
1. Install/use zigzag-capable persistence tooling and build source/target split complexes on shared landmarks.
2. Compute zigzag summaries across split transitions for layers `{0,3,7,11}`.
3. Compare edge utility against geodesic baseline with split-swap and layer-order permutation controls.
- Required artifacts:
  - `h44_true_zigzag_by_seed_layer_split.csv`
  - `h44_true_zigzag_domain_summary.csv`
  - `h44_true_zigzag_null_summary.csv`
- Promotion gate: at least one domain-split Fisher-significant positive zigzag delta and no global sign flip versus proxy direction.

### 3) Cheap broad-screen slot
- Candidate: `N191` (robust OOS intrinsic-dimension/local-linearity screen).
- Goal: rapidly decide if the ID mechanism survives robust out-of-sample scoring after `H42` instability.
- Minimal protocol:
1. Reuse H38 feature set and split structure.
2. Evaluate leave-layer-out and leave-seed-out with MAE and winsorized/trimmed `R^2` (not raw-only `R^2`).
3. Test robust OOS gains against permutation and block-bootstrap nulls.
- Required artifacts:
  - `h45_id_oos_robust_by_seed_split.csv`
  - `h45_id_oos_robust_domain_summary.csv`
  - `h45_id_oos_robust_null_summary.csv`
- Promotion gate: robust OOS gain significant in `>=2` domain-split groups with reduced holdout variance versus `H42`.

## Execution Order
1. `N195` first (highest near-term discovery probability).
2. `N191` second (cheap kill/rescue discriminator).
3. `N185` third (high-upside topology expansion).

## Scope Guardrails
1. Do not reopen rewiring-survival experiments.
2. Do not reopen GW-first correspondence recovery.
3. Do not rerun coarse discrete tier-gap variants.

## Minimal Recovery Plan (only if a future gate fails)
1. Run seed42-only `h43` with reduced null draws (`<=80`).
2. Run seed42-only `h45` with reduced null draws (`<=80`).
3. Emit mandatory executor artifacts and one machine summary JSON, then return to full 3-slot execution.
