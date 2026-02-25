# Executor Iteration Report — iter_0027

## Scope
Executed a breadth-oriented 3-hypothesis packet aligned to the brainstormer brief (`N315`, `N326`, `N324`):
- `H64` (`persistent_homology`, new method): support-margin two-axis filtration surrogate.
- `H65` (`cross_model_alignment`, major-change rescue): cross-model topology codebook transport.
- `H66` (`intrinsic_dimensionality`, new method): interaction-only ID screening.

## Environment
- Python environment: `subproject40-topology`
- No package installation required this iteration.

## Command Trace
```bash
# syntax check
conda run -n subproject40-topology \
  python -m py_compile iterations/iter_0027/run_iter0027_screen.py

# execute H64/H65/H66 screen packet
conda run -n subproject40-topology \
  python iterations/iter_0027/run_iter0027_screen.py
```

Primary runner:
- `iterations/iter_0027/run_iter0027_screen.py`

Primary machine-readable outputs:
- `iterations/iter_0027/iter0027_screen_summary.json`
- `iterations/iter_0027/h64_support_margin_two_axis_by_seed_layer_split.csv`
- `iterations/iter_0027/h64_support_margin_two_axis_domain_summary.csv`
- `iterations/iter_0027/h64_support_margin_two_axis_null_summary.csv`
- `iterations/iter_0027/h65_codebook_transport_by_domain_layer_split.csv`
- `iterations/iter_0027/h65_codebook_transport_domain_summary.csv`
- `iterations/iter_0027/h65_codebook_transport_null_summary.csv`
- `iterations/iter_0027/h66_id_interaction_by_seed_transition_split.csv`
- `iterations/iter_0027/h66_id_interaction_domain_summary.csv`
- `iterations/iter_0027/h66_id_interaction_null_summary.csv`

Execution runtime:
- End-to-end screen command wall time: `~8s`.

## Results

### H64 — Support-Margin Two-Axis Filtration (`persistent_homology`, new method)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 2 layers (7,11) = 36` rows.
- Score: directed baseline (`-distance + support + margin`) versus two-axis filtration connectivity summaries.
- Controls: margin shuffle within degree bins, support shuffle within degree bins, label permutation (`12` each).

Direct evidence:
- Mean `delta_AUROC(two-axis - baseline) = -0.03184`.
- Failure slices stayed negative:
  - `lung/source_disjoint = -0.02633`
  - `external_lung/source_disjoint = -0.02928`
- Positive mean domain-splits: `0/6` (keep gate failed).
- Two-axis vs one-axis ablation delta was near-zero (`mean +0.00034`), indicating no practical rescue.

Interpretation:
- This two-axis filtration surrogate did not recover the known source-disjoint failures and did not improve utility over baseline.

### H65 — Cross-Model Topology Codebook Transport (`cross_model_alignment`, major-change rescue)
Design:
- Coverage: seed42 pilot with held-out-domain transfer across `3 domains x 2 disjoint splits x 2 layers = 12` rows.
- Method: learned GF->scGPT token transport on two source domains and evaluated target-domain transfer.
- Primary endpoint: `null_gap_q95`; controls were random codebook mapping, token-frequency shuffle, and signature-destroy permutation (`24` each).

Direct evidence:
- Mean `null_gap_q95 = +0.13671`.
- Immune gate passed on null-gap (`immune mean null_gap_q95 = +0.16157`).
- Non-immune null-gap condition passed (`4/4` non-immune domain-splits had positive mean null-gap).
- However, transfer utility remained uniformly negative:
  - Mean `delta_AUROC(transfer - baseline) = -0.10204`
  - Positive mean domain-splits: `0/6`

Interpretation:
- The redesign improved null robustness relative to its own nulls, but it still degrades edge discrimination versus baseline across all domain-splits; this is not a promotable positive.

### H66 — ID Interaction-Only Screen (`intrinsic_dimensionality`, new method)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 3 transitions (0->3,3->7,7->11) = 54` rows.
- Score: directed-geodesic baseline plus ID interaction terms with support/margin only (no direct ID main-effect score).
- Controls: interaction-partner shuffle, layer-order permutation per gene, label permutation (`12` each).

Direct evidence:
- Mean `delta_AUROC(interaction - baseline) = -0.13176`.
- Positive mean domain-splits: `0/6`.
- Fisher-significant domain-splits: `0/6`.

Interpretation:
- Interaction-only ID features remained decisively negative and did not rescue the retired ID branch.

## Decision Summary
- `H64`: **negative** (fails both failure-slice rescue and aggregate utility criteria).
- `H65`: **negative** for practical utility (null-gap improved, but transfer utility is uniformly negative).
- `H66`: **negative** (large global negative effect with no domain-split significance).

## Branch Retirement Actions
- Retired `H64` endpoint form (two-axis filtration AUROC-lift variant).
- Retired `H65` cross-model branch for this loop after major-change rescue failed utility criterion.
- Retired `H66` ID-interaction branch for this loop.

## Blockers and Fallbacks
- No hard blocker (data, runtime, and environment available).
- No fallback packet required.

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` with marker `ITERATION UPDATE: iter_0027`.
- Compile command (run from `paper/`):
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
```
- Compiled PDF: `paper/autoloop_research_paper.pdf`.
