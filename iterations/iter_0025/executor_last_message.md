Completed one full executor iteration (`iter_0025`) with concrete experiments, artifacts, log updates, and paper compile.

Key outcomes:
- `H58` (topology weighting rescue): negative for rescue objective.  
  Mean `delta_AUROC(weighted-distance)=+0.01137`, but `weighted-unweighted=-0.00052`; source-disjoint failures remained negative (`lung=-0.00193`, `external_lung=-0.00433`).
- `H59` (cross-model topology-signature transfer): inconclusive.  
  Mean transfer delta `+0.02404`, positive `10/12`, but null-robust support failed (`0/12` rows with `p_best<0.05`, domain Fisher `0/3`).
- `H60` (ID-jump broad screen): negative.  
  Mean `delta_AUROC=-0.00435`, positive `31/72`, positive mean domain-splits `3/6`, Fisher-significant domain-splits `1/6` (negative-direction slice).

Mandatory iteration files written:
- `iterations/iter_0025/executor_iteration_report.md`
- `iterations/iter_0025/executor_next_steps.md`
- `iterations/iter_0025/executor_hypothesis_screen.json`

Machine-readable artifacts generated this iteration include:
- `iterations/iter_0025/iter0025_screen_summary.json`
- `iterations/iter_0025/iter0025_metric_digest.json`
- `iterations/iter_0025/h58_weighted_directed_signed_by_seed_layer_split.csv`
- `iterations/iter_0025/h59_cross_model_topology_signature_transfer_by_domain_layer.csv`
- `iterations/iter_0025/h60_id_jump_by_seed_layer_split.csv`
- plus their domain/null summary CSVs in `iterations/iter_0025/`.

Cumulative assets updated:
- `reports/autoloop_master_log.md`
- `paper/autoloop_research_paper.tex` (includes exact marker `ITERATION UPDATE: iter_0025`)
- `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`)

Tracking updated:
- `tracking/prompt.md`