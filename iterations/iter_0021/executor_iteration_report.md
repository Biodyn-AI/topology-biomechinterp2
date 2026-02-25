# Executor Iteration Report — iter_0021

## Scope
Executed a 3-hypothesis screening packet with breadth and one carry-over refinement:
- `H46` (`topology_stability`, refinement): support-weighted zigzag excess vs unweighted zigzag excess.
- `H47` (`persistent_homology`, new method): bifiltration-like cycle-rank score (`distance x support`) vs distance-only ablation.
- `H48` (`cross_model_alignment`, rescue with materially changed method): cross-model top-k motif-overlap enrichment under degree-preserving nulls.

## Environment
- Python environment: `subproject40-topology`
- No new package installation this iteration.

## Command Trace
```bash
# compile script
conda run --no-capture-output -n subproject40-topology \
  python -m py_compile iterations/iter_0021/run_iter0021_screen.py

# run the iteration packet
conda run --no-capture-output -n subproject40-topology \
  python iterations/iter_0021/run_iter0021_screen.py

# postprocess H48 to handle zero-variance null rows (z-score safeguard)
conda run --no-capture-output -n subproject40-topology python - <<'PY'
# recompute h48 delta/z summary from existing by-row + null CSVs,
# set z=NaN when null std is 0 and observed != null mean,
# update iter0021_screen_summary.json h48 block
PY

# compile paper update
latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex
# run in workdir=paper/
```

Primary script:
- `iterations/iter_0021/run_iter0021_screen.py`

Primary machine summary:
- `iterations/iter_0021/iter0021_screen_summary.json`

## Paper Update + Compile Trace
- Updated source: `paper/autoloop_research_paper.tex` (added section marker `ITERATION UPDATE: iter_0021`).
- Compiled PDF: `paper/autoloop_research_paper.pdf`.

## Results

### H46 — Support-weighted Zigzag Excess Coupling (`topology_stability`, refinement)
Design:
- Coverage: `3 domains x 3 seeds x 4 layers = 36` paired source→union←target rows.
- Topology metric: true zigzag `H1` total lifetime (unweighted and support-threshold-weighted variants).
- Utility anchor: mean diffusion incremental `ΔAUROC` from `iter_0016` (`h31`), aggregated over source/target disjoint splits.
- Null: target-set permutations (`36` permutations/row).

Key outcomes:
- Weighted-vs-unweighted objective failed: `0/36` rows where weighted excess > unweighted excess.
- Domain-level weighted utility-coupling improvement over unweighted: `1/3` domains.
- Leave-domain-out linear `R²`: weighted `-0.5186`, unweighted `-1.0049` (both poor; weighted less poor).
- Mean excess vs null remained positive for both variants:
  - weighted `+103.19`
  - unweighted `+211.40`
- `p_weighted_upper < 0.05` in `36/36` rows, but this did not translate into improved utility linkage.

Interpretation:
- This refinement gives decisive negative evidence for the specific claim that support-weighting improves zigzag utility coupling. The branch should not be promoted in current form.

Artifacts:
- `iterations/iter_0021/h46_weighted_zigzag_by_seed_layer_split.csv`
- `iterations/iter_0021/h46_weighted_zigzag_domain_summary.csv`
- `iterations/iter_0021/h46_weighted_zigzag_null_summary.csv`

---

### H47 — Bifiltration-like Cycle-Rank Screen (`persistent_homology`, new method)
Design:
- Coverage: seed42 pilot across all domains/splits/layers (`3 domains x 2 splits x 4 layers = 24` rows).
- Method: kNN graph cycle-rank (`β1 = m - n + c`) summarized over a `distance x support` grid; edge scores compared against distance-only ablation.
- Null: support shuffle within distance strata (`36` permutations/row).

Key outcomes:
- Mean `ΔAUROC(bifiltration - distance-only) = +0.00566`.
- Positive delta in `24/24` rows.
- Row-level `p_delta_auc_upper < 0.05` in `13/24` rows.
- Domain-split means were positive in `6/6`; Fisher-combined significance in `6/6`.
- Delta range: min `+0.00166`, max `+0.00975`.

Interpretation:
- This new method produced robust directional gains over distance-only in all domain/split groups and clears the continuation gate for multi-seed expansion.

Artifacts:
- `iterations/iter_0021/h47_bifiltration_by_domain_layer_split.csv`
- `iterations/iter_0021/h47_bifiltration_domain_summary.csv`
- `iterations/iter_0021/h47_bifiltration_null_summary.csv`

---

### H48 — Cross-model Top-k Motif Overlap (`cross_model_alignment`, rescue/new method)
Design:
- Coverage: domains `immune/lung/external_lung`, layers `{7,11}`, `k={50,100,200}` (`18` rows).
- Scores:
  - scGPT: layerwise centered-cosine in PCA space.
  - Geneformer: centered-cosine over token embedding table.
- Motifs: directed FFL + bifan overlap between scGPT and Geneformer top-k edge sets.
- Null: degree-preserving target permutation in each model (`90` permutations/row).

Key outcomes:
- Significant overlap enrichment localized to immune late-layer settings:
  - positive delta + significant `p` in `4/18` rows (all immune rows with `k>=100`).
- Domain summary:
  - immune: mean delta overlap `+1.0`, Fisher `p=3.14e-4`
  - lung: `0.0`, Fisher `p=1.0`
  - external_lung: `0.0`, Fisher `p=1.0`
- `domain_fisher_sig = 1/3`.

Interpretation:
- Evidence is mixed and domain-concentrated. This rescue method is not yet broadly robust, but immune-specific signal is non-random and worth one targeted follow-up.

Artifacts:
- `iterations/iter_0021/h48_cross_model_motif_overlap_by_domain_layer.csv`
- `iterations/iter_0021/h48_cross_model_motif_overlap_summary.csv`
- `iterations/iter_0021/h48_cross_model_motif_overlap_null_summary.csv`

## Decision Summary
- `H46`: **negative** (weighted zigzag did not improve utility coupling).
- `H47`: **promising** (consistent positive incremental signal vs distance-only, null-calibrated).
- `H48`: **inconclusive** (immune-only enrichment; no cross-domain robustness).

## Blockers and Fallbacks
- No hard runtime/data blockers.
- Method caveat in `H48`: many rows had zero-variance null overlap (sparse motif regime), so z-scores are undefined for non-tie rows. Fallback used exact upper-tail permutation p-values and raw overlap deltas as primary evidence.
