# Brainstormer Next Iteration Brief — iter_0006 -> iter_0007

## Gate Status
- Current gate (`executor_research_validation.json`) passed: `passed_min_research_gate=true`.

## Prioritized Plan

### 1) Primary Experiment (highest value)
- Name: **Metric-matched stronger-null calibration** (`N23`)
- Objective: resolve whether H07 is truly negative or calibration-mismatched.
- Minimal protocol:
  - Keep same immune full-layer split setup (3 seeds, 12 layers, source/target disjoint, 180 genes, PCA14).
  - Add observed geodesic PH on the original connected graph.
  - Compare both:
    - Euclidean observed vs feature-shuffle (continuity baseline).
    - Geodesic observed vs rewired geodesic null (calibrated stronger-null test).
- Required outputs:
  - by-seed CSV including `observed_metric_family` and calibrated null stats.
  - layer/domain summaries and pass matrix mirroring iter_0006 schema.
- Decision rule:
  - If calibrated branch still gives near-uniform `p=1.0` and all-negative deltas, keep rewiring hypothesis negative.
  - If calibrated branch recovers spread and selective significance, keep stronger-null branch active.

### 2) Backup Experiment (fast, still advances science)
- Name: **Biological anchoring of robust immune layers** (`N31`)
- Objective: convert current topological signal into biological interpretation.
- Protocol:
  - Use current robust layers `7, 9, 10, 11` from feature-shuffle pass matrix.
  - Extract top contribution genes per layer/split.
  - Run TRRUST/DoRothEA + GO enrichment with FDR correction.
- Required outputs:
  - layer/split enrichment tables, overlap summary across splits, and a compact narrative file.
- Decision rule:
  - Promote biological-anchoring branch if at least 2 layers show reproducible immune-regulatory enrichment across both splits.

### 3) Stretch Experiment (if time remains)
- Name: **External-lung full-layer split replication** (`N25`)
- Objective: test whether late-layer dual-split structure is domain-general.
- Protocol:
  - Reuse iter_0006 screening code path with `external-lung` inputs.
  - Run feature-shuffle first; include calibrated stronger null only if primary is complete.
- Required outputs:
  - external-lung analogs of layer summary, pass matrix, and domain summary.
- Decision rule:
  - If late layers again dominate dual-split passes, prioritize cross-domain depth-mechanism hypothesis.

## If A Future Gate Fails (Recovery Template)
- Execute a **minimal salvage packet** in the same iteration:
  1. Restrict to immune layers `7, 9, 10, 11` and one split (`target_disjoint`) to reduce runtime.
  2. Run feature-shuffle null only (10 draws) plus one calibrated stronger-null branch (4 draws).
  3. Emit at least one machine-readable CSV + `executor_hypothesis_screen.json` with explicit pass/fail.
- This guarantees forward screening progress even under runtime/tooling failure.
