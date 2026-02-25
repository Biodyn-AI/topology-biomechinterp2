# Executor Iteration Report — iter_0026

## Scope
Executed a breadth-oriented 3-hypothesis packet with one new family and two major-change rescues:
- `H61` (`graph_topology`, new family): curvature-assortativity surrogate screen.
- `H62` (`cross_model_alignment`, major-change rescue): biologically anchored contrastive alignment with null-gap objective.
- `H63` (`intrinsic_dimensionality`, major-change rescue): layer-transition ID-gradient screen with geodesic baseline.

## Environment
- Python environment: `subproject40-topology`
- No package installation required this iteration.

## Command Trace
```bash
# syntax check
conda run -n subproject40-topology \
  python -m py_compile iterations/iter_0026/run_iter0026_screen.py

# execute H61/H62/H63 packet
conda run -n subproject40-topology \
  python iterations/iter_0026/run_iter0026_screen.py
```

Primary runner:
- `iterations/iter_0026/run_iter0026_screen.py`

Primary machine-readable outputs:
- `iterations/iter_0026/iter0026_screen_summary.json`
- `iterations/iter_0026/h61_graph_curvature_by_seed_layer_split.csv`
- `iterations/iter_0026/h61_graph_curvature_domain_summary.csv`
- `iterations/iter_0026/h61_graph_curvature_null_summary.csv`
- `iterations/iter_0026/h62_anchor_alignment_by_domain_layer_split.csv`
- `iterations/iter_0026/h62_anchor_alignment_domain_summary.csv`
- `iterations/iter_0026/h62_anchor_alignment_null_summary.csv`
- `iterations/iter_0026/h63_transition_id_gradient_by_seed_transition_split.csv`
- `iterations/iter_0026/h63_transition_id_gradient_domain_summary.csv`
- `iterations/iter_0026/h63_transition_id_gradient_null_summary.csv`

Execution runtime:
- End-to-end screen command wall time: `~16.7s`.

## Results

### H61 — Graph Curvature-Assortativity Surrogate (`graph_topology`, new family)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Score: distance baseline plus Forman-like node-curvature gap, local clustering mean, and assortativity residual gap.
- Nulls: curvature shuffle within degree bins, topology-feature shuffle within degree bins, label permutation (`24` each).

Direct evidence:
- Mean `delta_AUROC(topology - distance) = -0.00719`.
- Positive mean domain-splits: `2/6` (`immune/target`, `lung/target`).
- Fisher-significant domain-splits: `1/6` (`lung/target_disjoint`, `p=0.00147`).
- Worst slices were source-disjoint (`immune/source = -0.03087`, `external_lung/source = -0.02427`).

Interpretation:
- This curvature surrogate is not a robust global gain over distance baseline; effect is split-conditional and mostly negative.

### H62 — Anchored Contrastive Cross-Model Alignment (`cross_model_alignment`, major-change rescue)
Design:
- Coverage: seed42 pilot over `3 domains x 2 disjoint splits x 2 layers = 12` rows.
- Method: source-domain anchor-restricted Procrustes mapping (anchors from DoRothEA+OmniPath+STRING+GO support), target-domain edge transfer scoring, null-gap objective.
- Nulls: random-map alignment, signature-destroy permutation, anchor-label shuffle (`24` each).

Direct evidence:
- Mean `delta_AUROC(transfer - baseline) = +0.04757`.
- Mean `null_gap_q95 = -0.12923` (fails null-gap keep gate globally).
- Immune is the only domain with strong support (`mean delta = +0.09047`, domain Fisher `p=0.00116`); lung and external-lung remain non-robust with strongly negative null gaps.
- Domain-split Fisher-significant rows: `2/6` (immune source/target only).

Interpretation:
- Directional transfer gain exists, but it does not clear null-gap robustness outside immune. Treat as inconclusive rescue, not promotable.

### H63 — Layer-Transition ID-Gradient Screen (`intrinsic_dimensionality`, major-change rescue)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 3 transitions (0->3,3->7,7->11) = 54` rows.
- Score: geodesic baseline plus transition ID-gradient consistency terms.
- Nulls: layer-order swap per gene, endpoint swap within geodesic bins, label permutation (`24` each).

Direct evidence:
- Mean `delta_AUROC(transition-ID - geodesic) = -0.02061`.
- Positive mean domain-splits: `1/6` (`external_lung/target = +0.01132`).
- Transition means are all negative: `0->3=-0.01843`, `3->7=-0.02018`, `7->11=-0.02324`.

Interpretation:
- Transition-based ID gradients did not rescue intrinsic-dimensionality utility over geodesic baseline.

## Decision Summary
- `H61`: **negative** (non-robust and mostly negative incremental value).
- `H62`: **inconclusive** (positive direction, but null-gap failure outside immune).
- `H63`: **negative** (all transition aggregates negative).

## Blockers and Fallbacks
- No hard blocker (data/runtime/environment all available).
- Runtime remained bounded, so no fallback packet was needed.

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with marker `ITERATION UPDATE: iter_0026`.
- Compile command (run from `paper/`):
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.
