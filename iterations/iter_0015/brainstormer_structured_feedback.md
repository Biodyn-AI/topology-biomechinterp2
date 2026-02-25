# Brainstormer Structured Feedback — iter_0015

## Inputs Inspected
- `iterations/iter_0015/executor_iteration_report.md`
- `iterations/iter_0015/executor_hypothesis_screen.json`
- `iterations/iter_0015/executor_research_validation.json`
- New machine artifacts in `iterations/iter_0015/`:
  - `h28_diffusion_coexp_by_seed_layer_split.csv`
  - `h28_diffusion_coexp_domain_summary.csv`
  - `h28_diffusion_coexp_null_summary.csv`
  - `h29_seeded_gw_domain_summary.csv`
  - `h29_seeded_gw_null_summary.csv`
  - `h29_seeded_gw_map_quality.csv`
  - `h30_hyperbolicity_by_seed_layer_split.csv`
  - `h30_hyperbolicity_domain_summary.csv`
  - `h30_hyperbolicity_null_summary.csv`
  - `iter0015_screen_summary.json`
  - `run_iter0015_screen.py`
- Cumulative context:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex`

## Research Gate
- `passed_min_research_gate=true`.
- No gate-recovery patch is required for this iteration.

## Iteration Signal Assessment
1. `H28` kept the diffusion branch alive but removed the prior promotion confidence.
- Overall mean uplift stayed positive (`+0.00774`), but matched-null support dropped sharply (`3/72` significant rows; `0/3` domains Fisher-significant).
- Split structure is informative: lung source-disjoint remains strongest (`mean delta +0.0193`), while external-lung and immune source-disjoint are slightly negative.
- Interpretation: raw diffusion advantage is partly coexpression-captured in lung/external-lung and must be tested as incremental signal after covariate adjustment.

2. `H29` is decisive negative evidence against GW as the primary correspondence engine.
- Seeded GW top-1 recovery remained near chance (`0.00833`) and transfer AUROC remained non-significant (`0.5008`).
- CCA seed itself remained strong (`top-1 ~0.745`), so GW post-processing is harming useful structure rather than rescuing it.

3. `H30` produced decisive negative evidence for the current thinness formulation.
- Mean thinness AUROC stayed below chance (`0.4657`) and below geodesic baseline by `-0.085`.
- Only `1/24` rows was significant; signal did not replicate across domains/splits.

## Stale Direction Triage

### `retire_now`
| Direction | Evidence | Action |
|---|---|---|
| GW-primary correspondence recovery (`H27` + `H29`) | Two materially different GW variants fail recovery/transfer while CCA seed is already strong | `retire_now` |
| Current triangle-thinness hyperbolicity edge score (`H30` form) | Below chance and below baseline in nearly all strata | `retire_now` |
| Rewiring-null survival lineage (`H07/H09/H12`) | Long negative streak with no rescue trend | `retire_now` |
| Raw Forman negative-curvature enrichment (`H23` form) | Opposite-direction and below-chance results across domains | `retire_now` |

### `rescue_once_with_major_change`
| Direction | Why not fully retired | Required major change | Action |
|---|---|---|---|
| Diffusion branch as raw AUROC delta (`H28` framing) | Direction remains positive but confounded under matched null | Switch to incremental value tests conditional on coexpression + degree + baseline distances | `rescue_once_with_major_change` |
| Geometry x prior interaction (`H26` framing) | Some calibration gain exists but no robust interaction sign | Use consensus priors (TRRUST+GO+STRING) plus ontology strata and prevalence controls | `rescue_once_with_major_change` |
| Intrinsic mechanism as universal sign claim (`H04/H18/H21/H22`) | Repeated domain/split sign flips suggest regime dependence | Model heterogeneity explicitly (domain x split x depth with ID-shape/anisotropy features) | `rescue_once_with_major_change` |

## Navigation Guidance for Next Loop
1. Keep diffusion as the main exploitation branch, but move the endpoint from raw uplift to adjusted incremental utility.
2. Replace GW-based mapping with CCA/spectral and cycle-consistency families for cross-model work.
3. For topology expansion, avoid another thinness polish and pivot to filtration/time-evolution PH methods.
4. Tie biological interpretation to consensus priors and cell-ontology strata so negative global averages do not hide program-specific signal.

## Minimal Recovery Plan (Only if a Future Gate Fails)
1. Run one diffusion incremental-value test on immune + lung only (both disjoint splits).
2. Emit mandatory files (`executor_iteration_report.md`, `executor_hypothesis_screen.json`, one machine summary CSV).
3. Resume full 3-hypothesis packet after validation gate is restored.
