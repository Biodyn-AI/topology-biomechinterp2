Implemented and executed exactly one iteration (`iter_0012`) and stopped.

Key outcomes:
- `H19` (confidence-stratified module anchoring) was negative: mean tier-slope AUROC was `-0.0771` (source) and `-0.0627` (target), with `0/12` positive-slope layers in both splits.
- `H20` (cross-model Procrustes/OT transfer) was promising/mixed: map-aware transfer was significant in `3/3` domains (mean transfer AUROC `0.5650`), while unsupervised OT recovery failed (mean top-1 `0.0024`, `0/3` significant).
- `H21` (local reconstruction mechanism) was mixed: source split positive (mean AUROC `0.5331`), target split negative (`0.4780`), and target coupling to geodesic lift was inverse/significant (`rho=-0.4079`, `p=0.0190`).

Required iteration artifacts written:
- `iterations/iter_0012/executor_iteration_report.md`
- `iterations/iter_0012/executor_next_steps.md`
- `iterations/iter_0012/executor_hypothesis_screen.json`
- Machine outputs include:
  - `iterations/iter_0012/iter0012_screen_summary.json`
  - `iterations/iter_0012/h19_confidence_community_by_seed_layer_split_bin.csv`
  - `iterations/iter_0012/h20_cross_model_transfer_by_domain_layer.csv`
  - `iterations/iter_0012/h21_local_reconstruction_edge_features.csv`

Cumulative updates completed:
- `reports/autoloop_master_log.md` updated with `iter_0012`.
- `paper/autoloop_research_paper.tex` updated with section marker `ITERATION UPDATE: iter_0012`.
- `paper/autoloop_research_paper.pdf` recompiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error autoloop_research_paper.tex`.
- `tracking/prompt.md` updated.

Runtime blockers resolved this iteration:
- Installed missing env deps:
  - `conda run -n subproject40-topology pip install transformers`
  - `conda run -n subproject40-topology pip install torch`