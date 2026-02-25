Iteration `iter_0014` is complete with concrete experiment execution and all required artifacts written.

- Ran bounded 3-hypothesis screen in `subproject40-topology`:
  1. `H25` diffusion-distance sweep (promising): mean AUROC uplift vs best Euclidean/geodesic baseline `+0.0173`; all 3 domains Fisher-significant.
  2. `H26` geometry x biological-prior interaction (neutral/mixed): interaction term not significant in `0/6` rows; small calibration gains in `2/6` rows (`combined Fisher p=0.0140` for AUROC deltas).
  3. `H27` unseeded GW alignment (inconclusive/mixed): correspondence recovery failed (mean top-1 `0.00119`, combined `p=0.990`), but coarse geometry alignment was strong (distance Spearman combined `p=2.33e-05`).

- Required iteration files created:
  - `iterations/iter_0014/executor_iteration_report.md`
  - `iterations/iter_0014/executor_next_steps.md`
  - `iterations/iter_0014/executor_hypothesis_screen.json`

- Machine-readable artifacts generated this iteration include:
  - `iterations/iter_0014/h25_diffusion_distance_by_seed_layer_split.csv`
  - `iterations/iter_0014/h25_diffusion_distance_domain_summary.csv`
  - `iterations/iter_0014/h25_diffusion_distance_null_summary.csv`
  - `iterations/iter_0014/h26_bio_anchor_edge_table.csv`
  - `iterations/iter_0014/h26_bio_anchor_model_summary.csv`
  - `iterations/iter_0014/h26_bio_anchor_permutation_null.csv`
  - `iterations/iter_0014/h27_gw_alignment_domain_summary.csv`
  - `iterations/iter_0014/h27_gw_alignment_null_summary.csv`
  - `iterations/iter_0014/h27_gw_alignment_map_quality.csv`
  - `iterations/iter_0014/iter0014_screen_summary.json`

- Cumulative updates completed:
  - `reports/autoloop_master_log.md` updated with `iter_0014`.
  - `paper/autoloop_research_paper.tex` updated with marker `% ITERATION UPDATE: iter_0014`.
  - PDF compiled at `paper/autoloop_research_paper.pdf` via:
    - `latexmk -pdf -interaction=nonstopmode autoloop_research_paper.tex`
  - `tracking/prompt.md` updated with `iter_0014` progress/decisions/blockers/next step.

- Dependency installed (logged): `conda run -n subproject40-topology pip install POT`.

