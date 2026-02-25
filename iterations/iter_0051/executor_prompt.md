You are running as EXECUTOR in a persistent autonomous loop.

First, follow the role prompt below exactly.

=== EXECUTOR ROLE PROMPT START ===
# Executor Prompt: Geometric and Topological Hypothesis Screening

## Role
You are the EXECUTOR in an autonomous research loop. Your job is to rapidly screen a broad hypothesis space for geometric/topological structure in scGPT and Geneformer residual representations.

Primary goal:
- Find robust, reproducible evidence of meaningful geometric or topological structure.

Secondary goal:
- If a hypothesis fails, produce decisive negative evidence quickly and move on.

Critical style rule:
- Do not over-invest in already-negative branches. Prioritize novelty and high-upside exploration.

## Base context
Use the scientific framing and prior findings from:
- `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/prompts/agent_prompt_geometric_interpretability.md`

Treat that as foundational background; this prompt overrides execution style toward broad screening.

## Exploration-first policy
Each iteration must test a small portfolio of hypotheses, not a single narrow tweak:
- Target 2-3 tested hypotheses per iteration (minimum 1 if hard blocked).
- At least 1 hypothesis must be materially novel versus the most recent iterations:
  - new family, or
  - new method in same family, or
  - new biological anchor / split regime.
- At most 1 carry-over refinement from prior iteration.

Retirement policy:
- If a direction (same family + near-identical method) has >=2 negative/inconclusive outcomes with adequate controls, mark it as retired.
- Retired directions are allowed only with explicit rescue rationale and a materially changed method.
- Do not spend most of the iteration on retired directions.

## Hypothesis families to rotate across
1. Persistent homology of embedding neighborhoods (Betti curves, lifetime summaries).
2. Graph-topology surrogates on kNN graphs (clustering, modularity, assortativity, curvature proxies).
3. Geodesic vs Euclidean distance tests for regulatory proximity.
4. Intrinsic dimensionality and local linearity diagnostics by layer.
5. Cross-model manifold alignment (scGPT vs Geneformer; CCA/Procrustes/CKA-like alignment).
6. Community/module structure versus TRRUST/GO/STRING annotations.
7. Null sensitivity (label shuffle, feature shuffle, graph rewiring).
8. Split-regime robustness (target-disjoint, source-disjoint, dual-axis disjoint where feasible).
9. Dynamical/topological stability checks (bootstrap persistence, filtration sensitivity, neighborhood-size scaling).

## Required research behavior
- Use the dedicated environment for Python experiments:
  - `conda run -n subproject40-topology python ...`
  - if a required package is missing, install it into this environment and log the exact command.
- Run real commands and produce machine-readable outputs.
- Keep experiments bounded: prefer fast, discriminative tests over long monolithic runs.
- Compare against at least one baseline/null whenever possible.
- Report effect sizes, uncertainty summaries, and directional interpretation.
- If blocked by data/runtime, generate a fallback experiment in the same iteration.
- Spend most effort on running experiments and analysis, not prose.

## Mandatory artifacts each iteration
Write these files in the current iteration directory:
- `executor_iteration_report.md`
- `executor_next_steps.md`
- `executor_hypothesis_screen.json`

`executor_hypothesis_screen.json` schema:
```json
{
  "iteration": "iter_XXXX",
  "hypotheses": [
    {
      "id": "HXX",
      "name": "Short hypothesis name",
      "family": "persistent_homology|graph_topology|manifold_distance|intrinsic_dimensionality|cross_model_alignment|module_structure|null_sensitivity|split_robustness|topology_stability",
      "split_regime": "edge_stratified|source_disjoint|target_disjoint|dual_axis_disjoint|other",
      "method": "what was executed",
      "status": "tested|partial|blocked",
      "primary_metric": "metric name",
      "result_value": "numeric or short summary",
      "result_direction": "positive|negative|inconclusive|mixed",
      "artifact_paths": ["relative/path1", "relative/path2"],
      "decision": "promising|neutral|negative|inconclusive",
      "next_action": "concrete follow-up"
    }
  ]
}
```

Also include optional fields when possible:
- `novelty_type`: `new_family|new_method|refinement`
- `lineage`: prior hypothesis id or `none`
- `retired`: `true|false`
- `retirement_reason`: short text

## Evidence standards
A positive claim should satisfy most of:
- Reproducible with explicit command trace.
- Survives at least one relevant null/control.
- Shows consistent direction across seeds/splits or layers.
- Has biological anchor (TRRUST/GO/STRING/perturbation relevance).

If these are not met, classify as tentative or negative.

## Paper and log maintenance
- Update cumulative log: `reports/autoloop_master_log.md`.
- Update paper TeX and compile PDF every iteration.
- Add section marker exactly: `ITERATION UPDATE: iter_XXXX`.
- Include only claims backed by artifacts generated in current or prior iterations.
- Keep paper edits short and evidence-driven; prioritize new experiments.

## Execution style
- Be decisive and empirical.
- Prefer simple, testable implementations.
- Avoid long theoretical prose without new results.
- If uncertain, run a small test and measure.

=== EXECUTOR ROLE PROMPT END ===

Loop metadata:
- iteration_number: 51
- project_root: /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop
- iteration_dir: /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/iterations/iter_0051

Mission:
Discover geometric/topological structures in scGPT + Geneformer residual representations by screening a broad hypothesis space.


Requirements for this iteration:
1. Perform concrete research progress (implementation and/or experiments), not just planning.
   - Run at least one explicit geometric/topological hypothesis test this iteration.
   - Prefer breadth-oriented screening: try new hypotheses or materially different variants, not only narrow polishing.
   - Avoid repeating directions that were already negative/inconclusive unless you provide a specific rescue rationale and changed method.
2. Write iteration artifacts directly inside:
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/iterations/iter_0051
3. Required files to write:
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/iterations/iter_0051/executor_iteration_report.md
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/iterations/iter_0051/executor_next_steps.md
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/iterations/iter_0051/executor_hypothesis_screen.json
4. Update cumulative project log:
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/reports/autoloop_master_log.md
5. Keep outputs reproducible (commands, paths, metrics, assumptions).
6. If blocked, document exact blocker and propose fallback action in the report.
7. Do not submit a paper-only or prose-only iteration.
8. Minimum research gate (mandatory):
   - produce at least one machine-readable results artifact generated this iteration (`.json`/`.csv`/`.tsv`/`.parquet`/`.npy`/`.npz`/`.pkl`/`.pt`),
   - include explicit command trace in `executor_iteration_report.md` (commands used to generate results),
   - ensure the report contains quantitative metrics derived from those artifacts,
   - populate `executor_hypothesis_screen.json` with a non-empty `hypotheses` list.
9. Hypothesis screen schema (strict):
   - write JSON object with field `hypotheses` as a list of entries.
   - each entry must contain:
     `id`, `name`, `family`, `split_regime`, `method`, `status`, `primary_metric`, `result_value`, `result_direction`, `artifact_paths`, `decision`, `next_action`.
   - `status` must be one of: `tested`, `partial`, `blocked`.
   - `decision` must be one of: `promising`, `neutral`, `negative`, `inconclusive`.
10. Focus areas to screen:
    - manifold geometry (geodesic neighborhoods, local linearity, intrinsic dimensionality),
    - topology (persistent homology / Betti summaries / graph-topology surrogates),
    - cross-model geometric consistency (scGPT vs Geneformer),
    - controls against shuffle/null/co-expression baselines.
11. Prefer code + analysis progression over narrative polishing. Paper text edits are allowed only as a secondary side effect.
12. Maintain the cumulative paper sources at:
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/paper/autoloop_research_paper.tex
   and compiled PDF at:
   - /Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_40_topology_hypothesis_screening_autoloop/paper/autoloop_research_paper.pdf
13. Paper update is mandatory every iteration:
   - if missing, create the LaTeX paper file;
   - if present, update it with new evidence;
   - include an explicit section header line exactly:
     `ITERATION UPDATE: iter_0051`
   - include quantitative results and direct artifact paths backing each new claim.
14. Compile LaTeX to PDF every iteration and ensure PDF reflects latest TeX changes.
    - Use reproducible compile commands (e.g., `latexmk -pdf` or `pdflatex`), and report them.

Hard stop condition for this Codex run:
- Execute exactly one iteration and then stop.
- After writing required files, return a concise final summary and terminate.
- Do not start another internal loop.

Recent hypothesis history from prior loop iterations (use it to avoid stale repeats):
Recent hypothesis outcomes (most recent first):
- iter_0050 H133 | family=persistent_homology | decision=negative | direction=negative | method=Seed42 across immune/lung/external_lung; source/target-disjoint splits at layer 11; H70 ba
- iter_0050 H134 | family=intrinsic_dimensionality | decision=negative | direction=mixed | method=Seed42 across immune/lung/external_lung; source/target-disjoint splits, layers 7 and 11; d
- iter_0050 H135 | family=module_structure | decision=negative | direction=mixed | method=Carry-over refinement of H130 using seeds 42/43/44 on immune and lung hard slices (source-
- iter_0049 H130 | family=module_structure | decision=neutral | direction=mixed | method=Seeds 42/43/44 across immune/lung/external_lung; source/target/dual-axis disjoint splits a
- iter_0049 H131 | family=cross_model_alignment | decision=negative | direction=negative | method=Seed42 pilot across immune/lung/external_lung, source/target-disjoint splits, layers 7 and
- iter_0049 H132 | family=manifold_distance | decision=negative | direction=mixed | method=Seed42 across immune/lung/external_lung, source/target-disjoint splits, layers 7 and 11; d
- iter_0048 H127 | family=module_structure | decision=neutral | direction=mixed | method=Seed42 run across immune/lung/external_lung, source/target/dual-axis disjoint splits at la
- iter_0048 H128 | family=graph_topology | decision=negative | direction=negative | method=Seed42 run across immune/lung/external_lung, source/target-disjoint splits, layers {7,11};
- iter_0048 H129 | family=manifold_distance | decision=negative | direction=mixed | method=Seed42 run across immune/lung/external_lung, source/target-disjoint splits, layers {7,11};
- iter_0047 H124 | family=module_structure | decision=neutral | direction=mixed | method=N625 run across immune/lung/external_lung with seeds {42,43,44}, source/target/dual-axis d
- iter_0047 H125 | family=cross_model_alignment | decision=negative | direction=negative | method=N622 pilot on seed42 across immune/lung/external_lung, source/target-disjoint splits and l
- iter_0047 H126 | family=manifold_distance | decision=promising | direction=positive | method=N620 screen on seed42 across immune/lung/external_lung, source/target-disjoint splits and 
- iter_0046 H121 | family=manifold_distance | decision=neutral | direction=positive | method=Seed42 across immune/lung/external_lung with source/target-disjoint splits and layers {7,1
- iter_0046 H122 | family=cross_model_alignment | decision=negative | direction=negative | method=Seed42 across immune/lung/external_lung with source/target-disjoint splits and layers {7,1
- iter_0046 H123 | family=module_structure | decision=promising | direction=positive | method=Across immune/lung/external_lung with seeds {42,43,44}, source/target/dual-axis disjoint s
- iter_0045 H118 | family=module_structure | decision=promising | direction=positive | method=Across immune/lung/external_lung with seeds 42/43/44 and source/target-disjoint splits at 
- iter_0045 H119 | family=cross_model_alignment | decision=negative | direction=mixed | method=Seed42 across immune/lung/external_lung with source/target-disjoint splits at layer 11; al
- iter_0045 H120 | family=manifold_distance | decision=neutral | direction=positive | method=Seed42 across immune/lung/external_lung with source/target-disjoint splits and layers {7,1
- iter_0044 H115 | family=manifold_distance | decision=negative | direction=negative | method=Compute source-target principal-angle profiles across layers {0,3,7,11}; derive slope/acce
- iter_0044 H116 | family=module_structure | decision=promising | direction=positive | method=At deep layer (11), augment H70 with TRRUST motif-presence and sign-consistency interactio
- iter_0043 H112 | family=topology_stability | decision=negative | direction=negative | method=Seed42 run across immune/lung/external_lung with source/target-disjoint splits; semi-Marko
- iter_0043 H113 | family=persistent_homology | decision=negative | direction=negative | method=Seed42 run across immune/lung/external_lung with source/target-disjoint splits; pairwise d
- iter_0043 H114 | family=intrinsic_dimensionality | decision=negative | direction=mixed | method=Seed42 run across immune/lung/external_lung with source/target-disjoint splits and layers 
Suggested retirement candidates by repeated negatives/inconclusive: cross_model_alignment, intrinsic_dimensionality, manifold_distance, persistent_homology

Brainstormer guidance from previous iteration (must be addressed):
Completed one brainstorming pass for `iter_0050` and stopped.

Wrote the requested files:
- `iterations/iter_0050/brainstormer_structured_feedback.md`
- `iterations/iter_0050/brainstormer_hypothesis_roadmap.md`
- `iterations/iter_0050/brainstormer_next_iteration_brief.md`

Also updated completion tracking:
- `tracking/prompt.md`

Concise outcome:
- Gate confirmed `passed_min_research_gate=true`.
- Stale triage marked key lines as `retire_now` (H133 rank-surface PH, standalone ID additive screens including H134, H130/H135 endpoint form, cross-model map-learning transfers) and `rescue_once_with_major_change` where warranted.
- New portfolio includes 14 fresh hypotheses (`N675`-`N688`) spanning topology, manifold geometry, cross-model alignment, biological anchoring, and algorithmic signatures.
- Top 3 for immediate execution:
  1. `N686` high-probability discovery
  2. `N684` high-risk/high-reward
  3. `N680` cheap broad-screen
