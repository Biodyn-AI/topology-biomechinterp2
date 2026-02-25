# Brainstormer Next Iteration Brief — iter_0010 -> iter_0011

## Gate Status
- `iter_0010` research gate: `passed_min_research_gate=true`.
- No immediate gate-recovery run is required.

## Execution Priorities

### 1) Primary (high-probability discovery)
- Experiment: **N63 — biological anchoring of H13/H14 top layers**.
- Objective: translate geometric/topological positives into interpretable regulatory programs.
- Minimal protocol:
1. Use top H13 layers (`4,8,9`) and top H14 layers (`0,1,4`) plus weak-layer controls.
2. Extract top contributing genes per layer for both source- and target-disjoint analyses.
3. Run TRRUST + DoRothEA enrichment and report split-replication overlap.
- Decision rule:
1. Promote if key TF programs replicate across both splits and are stronger than weak-layer controls.
2. Deprioritize if enrichment is inconsistent or only appears in one split.

### 2) Secondary (high-risk/high-reward)
- Experiment: **N60 — cross-model geodesic transfer**.
- Objective: test whether manifold signal is model-invariant between scGPT and Geneformer.
- Minimal protocol:
1. Build shared-gene alignment per layer (start with Procrustes; optional OT follow-up).
2. Transfer geodesic neighborhoods and evaluate edge AUROC lift (geodesic vs Euclidean) in transferred space.
3. Report neighborhood overlap and transfer degradation relative to in-model baseline.
- Decision rule:
1. Promote if transferred geodesic lift remains positive and neighborhood overlap is materially above random maps.
2. Retire if transfer collapses to baseline after aligned controls.

### 3) Tertiary (cheap broad-screen)
- Experiment: **N58 — stretch-decile monotonicity**.
- Objective: quickly test whether local nonlinearity explains H13 gains.
- Minimal protocol:
1. Compute geodesic/euclidean stretch ratio per candidate edge.
2. Bin by deciles and model geodesic lift trend with split-aware controls.
3. Run permutation test for slope significance.
- Decision rule:
1. Keep mechanistic branch if slope is positive and significant in both disjoint splits.
2. Drop this explanation if trend is flat/unstable.

## Scope Guardrails for iter_0011
1. Do not reopen rewiring-survival broad sweeps; that branch is retired unless a single major-design rescue is explicitly selected.
2. Keep analyses decision-grade: one clear null/control per test and explicit promotion/retirement gate.
3. Prefer artifacts that are directly pluggable into paper narrative (tables/CSV with per-layer, per-split effect + p-values).

## Minimal Recovery Plan (only if a future executor gate fails)
1. Run one compact continuity packet on `N58` only (single domain, both disjoint splits, permutation null).
2. Emit three mandatory artifacts: machine CSV, `executor_hypothesis_screen.json`, and short `executor_iteration_report.md` with command trace.
3. Use this to re-establish valid experiment flow before resuming high-cost branches.
