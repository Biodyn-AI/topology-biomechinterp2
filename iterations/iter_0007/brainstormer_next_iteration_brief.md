# Brainstormer Next Iteration Brief — iter_0007 -> iter_0008

## Gate Status
- Current gate passed: `passed_min_research_gate=true`.
- No immediate recovery action is required for iter_0007 deliverable quality.

## Prioritized Plan

### 1) Primary Experiment
- Name: **Connectivity-aware stronger-null calibration** (`N35 + N36`)
- Objective: test whether rewiring negativity is mainly a graph-construction artifact (bridge/high-k, geometry-destructive swaps) rather than true absence of topology.
- Minimal protocol:
  1. Keep immune full-layer, source/target-disjoint setup from iter_0007.
  2. Add edge-length-bin-preserving degree rewiring (quantile bins) as alternative null.
  3. Emit row-level strata: bridged vs non-bridged, and k buckets.
  4. Report geodesic H1 deltas/p-values and distortion deltas per stratum and per layer-split.
- Decision rule:
  - If both constrained-null and bridge-conditioned analyses stay uniformly negative, close rewiring-survival branch for immune.
  - If non-bridged or constrained-null strata recover spread/selective support, keep branch open and recalibrate before global rejection.

### 2) Backup Experiment
- Name: **Biological anchoring of robust immune late layers** (`N41`)
- Objective: keep scientific momentum on the positive branch even if rewiring calibration remains negative.
- Minimal protocol:
  1. Use immune robust layers `7,9,10,11` from feature-shuffle pass matrix.
  2. Extract cycle-contributing genes per split.
  3. Run TRRUST/DoRothEA and GO enrichment with FDR correction.
- Decision rule:
  - Promote biological branch if at least two layers replicate immune-regulatory enrichment across both splits.

### 3) Stretch Experiment
- Name: **External-lung metric-matched rewiring replication** (`N34`)
- Objective: test cross-domain generality of the immune rewiring-negative result.
- Minimal protocol:
  1. Reuse iter_0007 script path with external-lung inputs.
  2. Keep same output schema (by-seed, layer summary, pass matrix, domain summary).
- Decision rule:
  - If external-lung remains uniformly negative, confidence increases that rewiring branch is broadly non-supportive under this family.
  - If external-lung shows selective support, branch remains domain-conditional and should not be globally closed.

## Contingency Recovery Plan (only if a future gate fails)
1. Run a minimal salvage packet on immune layers `7,9,10,11` only.
2. Execute one split (`target_disjoint`) with:
- feature-shuffle null (`10` draws),
- one constrained rewiring null (`4` draws).
3. Write at minimum:
- one machine CSV with per-layer deltas/p-values,
- `executor_hypothesis_screen.json`,
- short `executor_iteration_report.md` with command trace.
4. This preserves forward screening progress in one short cycle even under tooling/runtime failure.
