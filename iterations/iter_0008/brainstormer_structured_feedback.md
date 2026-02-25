# Brainstormer Structured Feedback — iter_0008

## Inputs Inspected
- `iterations/iter_0008/executor_iteration_report.md`
- `iterations/iter_0008/executor_hypothesis_screen.json`
- `iterations/iter_0008/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0008/`:
  - `h1_immune_constrained_rewire_by_seed_layer.csv`
  - `h1_immune_constrained_rewire_layer_summary.csv`
  - `h1_immune_constrained_rewire_pass_matrix.csv`
  - `h1_immune_constrained_rewire_domain_summary.csv`
  - `h1_immune_constrained_rewire_bridge_k_strata_summary.csv`
  - `h1_immune_constrained_rewire_bridge_gap_summary.csv`
  - `h1_immune_constrained_rewire_paired_shift_by_seed_layer.csv`
  - `h1_immune_constrained_rewire_paired_shift_summary.csv`
  - `iter0008_screen_summary.json`
  - `run_iter0008_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`
  - `tracking/prompt.md`

## Research Gate
- `passed_min_research_gate`: `true`
- Artifact completeness/format checks passed; this is a valid screening iteration.

## Hypothesis Triage

### Promising
1. **Bridge-identifiable rerun is high-leverage and unresolved.**
- Direct evidence: bridge status is strongly confounded with split (`source: 36/36 bridged`, `target: 2/36 bridged`), so current `N35` comparison is not identifiable.
- Why promising: one targeted rerun can convert this from confounded to decision-grade with minimal scope expansion.

2. **Late layers remain closest to rewiring survival even though still negative.**
- Direct evidence: least-negative deltas are concentrated at deeper layers (for example layer `11`: source `-6.81` unconstrained, `-5.35` constrained; target `-17.94` unconstrained, `-16.78` constrained).
- Why promising: if any rescue exists, it is more likely depth-localized than global.

3. **Cumulative feature-shuffle topology branch remains the strongest positive anchor.**
- Direct evidence from cumulative log: prior immune dual-split robust layers (`7,9,10,11`) persist across iterations even while rewiring branch stays negative.
- Why promising: biological anchoring on these layers can still yield publishable insight even if rewiring branch is closed.

### Neutral
1. **H11 (bridge-conditioned explanation) is currently inconclusive, not supported.**
- Direct evidence: pooled bridge-minus-nonbridge H1 gap is opposite expected direction (`+16.147` unconstrained; `+17.496` constrained), but attribution is invalid under split confounding.
- Interpretation: neutral/inconclusive pending identifiable design.

2. **Quantile-constrained rewiring calibration is directional but weak and split-dependent.**
- Direct evidence: source mean H1 shift (constrained - unconstrained) is `+0.556`, while target is `-1.132`; source edge-hist L1 improves slightly (`0.3249 -> 0.3129`) and target does not (`0.1461 -> 0.1466`).
- Interpretation: calibration effect exists but is too small/unstable to change branch state.

3. **Cross-domain generality of constrained-null behavior is unresolved.**
- Direct evidence: iter_0008 constrained-null packet was run in immune only.
- Interpretation: keep as neutral until one external-lung replication is done.

### Negative
1. **H12 (constrained rewiring rescue) is negative in immune.**
- Direct evidence: H1 significant tests `0/24` in both null families; dual-split passes `0/12`; all mean layer deltas remain negative.
- Interpretation: no rescue under current constrained-null design.

2. **Distortion-lower-tail rescue branch is negative.**
- Direct evidence: distortion lower-tail significance `0/24` in both families; minimum Fisher `p=0.0964`.
- Interpretation: no evidence that constrained rewiring restores topology via reduced distortion.

3. **Rewiring-survival branch is repeatedly non-supportive across recent iterations.**
- Direct evidence from cumulative log: iter_0006, iter_0007, iter_0008 all show zero significant rewiring-survival passes in immune under progressively calibrated designs.
- Interpretation: branch should be closed soon unless one final high-information control contradicts this trend.

## Decision Guidance
1. Run exactly one bridge-identifiable rerun in immune, not another broad unconstrained sweep.
2. Run one external-lung constrained-null replication in parallel or immediately after.
3. If both remain uniformly negative, close rewiring-survival as a primary branch and prioritize biological anchoring of robust feature-shuffle layers.
