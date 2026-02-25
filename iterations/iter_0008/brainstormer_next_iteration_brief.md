# Brainstormer Next Iteration Brief — iter_0008 -> iter_0009

## Gate Status
- Current iteration gate: `passed_min_research_gate=true`.
- No gate-recovery action is required for iter_0008 itself.

## Prioritized Plan

### 1) Primary Experiment
- Name: **Bridge-identifiable immune rerun with fixed-k factorial control** (`N44`)
- Objective: resolve `H11` confounding and produce a decision-grade bridge effect estimate.
- Minimal protocol:
1. Reuse immune full-layer pipeline with same seeds/splits.
2. For each split, force two k schedules (low/high) that yield both bridged and non-bridged rows.
3. Keep both null families (`degree_preserving_geodesic_rewire`, `quantile_constrained_geodesic_rewire`).
4. Report within-split bridge gaps plus split-adjusted pooled estimate.
- Decision rule:
1. If bridge gap remains absent/opposite after within-split control, close bridge-explanation branch.
2. If bridge gap appears only in high-k strata, keep branch open but narrowly scoped.

### 2) Backup Experiment
- Name: **External-lung replication of constrained-null packet**
- Objective: test whether iter_0008 immune-negative rewiring behavior is domain-general.
- Minimal protocol:
1. Run the same constrained-null schema on external-lung.
2. Emit the same artifact set (`by_seed`, `layer_summary`, `pass_matrix`, `domain_summary`, bridge and paired-shift summaries).
- Decision rule:
1. If external-lung is also uniformly negative, treat rewiring-survival as broadly non-supportive.
2. If external-lung shows selective survival, downgrade to domain-conditional rather than globally negative.

### 3) Stretch Experiment
- Name: **Biological anchoring of robust-vs-fragile layer gene programs** (`N50`)
- Objective: maintain scientific yield independent of rewiring branch outcome.
- Minimal protocol:
1. Use robust feature-shuffle layers (`7,9,10,11`) and matched fragile controls.
2. Extract cycle-contributing genes per split/layer.
3. Run TRRUST/DoRothEA/GO enrichment and cross-split overlap tests.
- Decision rule:
1. Promote if robust layers show reproducible immune TF/pathway coherence across both splits.

## Fast Recovery Plan (only if a future gate fails)
1. Time-box to a mini packet on layers `7,9,10,11` and one split (`target_disjoint`).
2. Run one weak null (feature-shuffle, `10` draws) plus one stronger null (constrained rewiring, `4-6` draws).
3. Require three outputs only:
- one machine CSV with per-layer deltas/p-values,
- `executor_hypothesis_screen.json`,
- short `executor_iteration_report.md` with command trace.
4. Use this as a continuity run to preserve screening velocity while unblocking infra or data issues.
