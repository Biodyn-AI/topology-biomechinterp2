# Executor Iteration Report — iter_0022

## Scope
Executed a 3-hypothesis screening packet with one carry-over refinement and two materially new variants:
- `H49` (`persistent_homology`, refinement): multiseed bifiltration robustness + utility coupling check.
- `H50` (`topology_stability`, new method): directed/signed topology pilot vs distance-only baseline.
- `H51` (`cross_model_alignment`, rescue/new method): expanded cross-model motif fingerprint with anti-sparsity module collapse.

## Environment
- Python environment: `subproject40-topology`
- No new package installation this iteration.

## Command Trace
```bash
# compile runner
conda run --no-capture-output -n subproject40-topology \
  python -m py_compile iterations/iter_0022/run_iter0022_screen.py

# execute screening packet (final deterministic run)
conda run --no-capture-output -n subproject40-topology \
  python iterations/iter_0022/run_iter0022_screen.py

# compile paper update
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

Primary script:
- `iterations/iter_0022/run_iter0022_screen.py`

Primary machine summary:
- `iterations/iter_0022/iter0022_screen_summary.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0022`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H49 — Multiseed Bifiltration Robustness + Utility Coupling (`persistent_homology`, refinement)
Design:
- Coverage: `3 domains x 3 seeds x 2 disjoint splits x 4 layers = 72` rows.
- Method: distance x support cycle-rank score against distance-only ablation.
- Nulls: distance-stratified support-shuffle (`20` permutations/row) + layer-order placebo for utility coupling (`200` permutations/domain).

Key outcomes:
- Mean `delta_AUROC(bifiltration - distance-only) = +0.00599`.
- Positive delta in `69/72` rows.
- Row-level `p<0.05` in `48/72` rows.
- Domain-split Fisher significance in `6/6` groups.
- Utility coupling did not replicate broadly:
  - positive utility correlation in `1/3` domains,
  - layer-placebo significant in `0/3` domains.

Interpretation:
- Edge-discrimination robustness of bifiltration is strong and reproducible across seeds/splits.
- The utility-coupling claim remains unproven in this formulation.

Artifacts:
- `iterations/iter_0022/h49_bifiltration_multiseed_by_seed_layer_split.csv`
- `iterations/iter_0022/h49_bifiltration_multiseed_domain_summary.csv`
- `iterations/iter_0022/h49_bifiltration_multiseed_null_summary.csv`

---

### H50 — Directed/Signed Topology Pilot (`topology_stability`, new method)
Design:
- Coverage: seed42 pilot on `3 domains x 2 disjoint splits x 2 layers = 12` rows.
- Method: directed/signed topology scores from support-direction margins, compared with distance-only cycle baseline.
- Nulls: node-relabel degree-orientation null + sign-shuffle null (`24` permutations each/row).

Key outcomes:
- Mean `delta_AUROC(directed-signed - distance-only) = +0.01585`.
- Positive delta in `11/12` rows.
- `p_degree<0.05` in `7/12` rows.
- `p_sign<0.05` in `8/12` rows.
- Domain-split Fisher significance in `6/6` groups.

Interpretation:
- This materially new directed/signed formulation is the strongest new positive this iteration and clears the continuation gate.

Artifacts:
- `iterations/iter_0022/h50_directed_signed_topology_by_domain_layer_split.csv`
- `iterations/iter_0022/h50_directed_signed_topology_domain_summary.csv`
- `iterations/iter_0022/h50_directed_signed_topology_null_summary.csv`

---

### H51 — Expanded Cross-model Motif Fingerprint (`cross_model_alignment`, rescue/new method)
Design:
- Coverage: domains `immune/lung/external_lung`, layers `{7,11}`, `k={100,200,400}`, variants `{gene,module}` (`36` rows total).
- Motif panel: `{FFL, bifan, feedback triad, feedforward chain, multi-input}`.
- Nulls: degree-preserving rewiring (`36` permutations/row) and module-membership shuffle (module variant only).

Key outcomes:
- Degree-null enrichment improved materially:
  - degree-null `p<0.05` in `16/36` rows,
  - domain-variant Fisher significant in `5/6` summary rows,
  - at least one significant variant in `3/3` domains.
- Module-shuffle robustness failed:
  - module-shuffle `p<0.05` in `0/18` module rows.

Interpretation:
- Anti-sparsity changes rescued broad degree-null signal but not module-level biological-attribution robustness.
- This formulation is mixed and should be retired unless the objective is materially changed.

Artifacts:
- `iterations/iter_0022/h51_cross_model_motif_fingerprint_by_domain_layer_k.csv`
- `iterations/iter_0022/h51_cross_model_motif_fingerprint_summary.csv`
- `iterations/iter_0022/h51_cross_model_motif_fingerprint_null_summary.csv`

## Decision Summary
- `H49`: **inconclusive** (robust discrimination signal, insufficient utility coupling).
- `H50`: **promising** (consistent positive directed/signed lift under two null families).
- `H51`: **neutral/mixed** (degree-null gains broadened, module-shuffle control failed).

## Blockers and Fallbacks
- No hard data/runtime blockers.
- Method caveat: `H51` gene-level rows still include some degenerate degree-null variance; inference relied on permutation p-values and aggregate Fisher tests, with module-shuffle controls used as the strict attribution check.
