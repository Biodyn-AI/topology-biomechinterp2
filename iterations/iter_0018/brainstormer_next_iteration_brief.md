# Brainstormer Next Iteration Brief - iter_0018 -> iter_0019

## Gate Status
- Current gate passed: `passed_min_research_gate=true`.
- Run a normal 3-hypothesis packet (no gate-recovery-only mode required).

## Recommended 3-Hypothesis Packet

### 1) High-probability discovery slot
- Candidate: `N179` (continuous biological support interaction).
- Goal: replace failed coarse tiers with a continuous multi-prior support model.
- Minimal protocol:
1. Build per-edge support score from TRRUST, DoRothEA confidence, STRING edge score, and GO overlap.
2. Fit baseline vs interaction models across domains/splits/layers `{0,3,7,11}` and seeds `{42,43,44}`.
3. Calibrate with support-score permutation within degree x coexpression x geodesic strata.
- Required artifacts:
  - `h40_support_interaction_by_seed_layer_split.csv`
  - `h40_support_interaction_domain_summary.csv`
  - `h40_support_interaction_null_summary.csv`
- Promotion gate: positive significant interaction with stronger top-decile support uplift in `>=2/3` domains.

### 2) High-risk/high-reward slot
- Candidate: `N171` (split-zigzag persistence).
- Goal: test split-invariant cycle structure with a materially new PH construction.
- Minimal protocol:
1. Build source/target split complexes on shared landmarks and compute zigzag persistence summaries.
2. Derive edge-local zigzag persistence features and compare against current PH/geodesic baselines.
3. Use split swap + layer-order permutation controls.
- Required artifacts:
  - `h41_zigzag_persistence_by_seed_layer_split.csv`
  - `h41_zigzag_persistence_domain_summary.csv`
  - `h41_zigzag_persistence_null_summary.csv`
- Promotion gate: positive zigzag-derived incremental utility with at least one significant domain-split aggregate.

### 3) Cheap broad-screen slot
- Candidate: `N174` (out-of-sample ID moments).
- Goal: determine quickly whether `H38` is robust or mostly in-sample fit.
- Minimal protocol:
1. Reuse ID moment features from H38.
2. Evaluate leave-one-layer-out and leave-one-seed-out prediction of layer AUROC trajectories.
3. Compare out-of-sample `delta R^2` against permutation null.
- Required artifacts:
  - `h42_id_oos_by_seed_split.csv`
  - `h42_id_oos_domain_summary.csv`
  - `h42_id_oos_null_summary.csv`
- Promotion gate: significant out-of-sample gain in at least two domain-splits.

## Execution Order
1. `N179` first (highest near-term discovery probability, low cost).
2. `N174` second (fast discriminator for mechanism validity).
3. `N171` third (highest novelty and upside).

## Scope Guardrails
1. Do not spend slots on rewiring-null survival reruns.
2. Do not reopen GW-primary edge correspondence methods.
3. Do not rerun coarse tier-gap variants of `H37`.

## Minimal Recovery Plan (only if a future gate fails)
1. Seed42-only `N179` on all domains/splits and layers `{0,3,7,11}`.
2. Seed42-only `N174` with reduced permutations (`<=100`).
3. Emit required executor files and one machine summary artifact, then return to full packet next loop.
