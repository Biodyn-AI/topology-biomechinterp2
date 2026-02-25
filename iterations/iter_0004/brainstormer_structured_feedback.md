# Brainstormer Structured Feedback — iter_0004

## Gate Status
- `passed_min_research_gate`: `true` (from `iterations/iter_0004/executor_research_validation.json`).

## Hypothesis Triage

### Promising
- `H03` (`persistent_homology`) is strong and replicated in new domains.
- Evidence: immune `12/12` and external-lung `12/12` layers with Fisher `p < 0.05`; mean layer deltas `12.074` and `12.482`; top-layer Fisher `p = 0.0056` in both domains.
- Cumulative read: topology branch is now positive in lung (iter_0003) plus immune/external-lung (iter_0004), so failure mode is no longer “domain-specific artifact in one tissue”.

### Neutral
- `H04` (`intrinsic_dimensionality`) remains mixed.
- Evidence: external-lung shows coherent coupling (`participation_ratio`: rho `+0.508`, `p=0.0229`; `linearity_top5`: rho `-0.510`, `p=0.0178`), but immune is non-significant (`p >= 0.147`).
- Interpretation: plausible manifold branch, but not yet cross-domain robust.

### Negative / Cautionary
- `mle_intrinsic_dim` is effectively negative in this screen: null-like (`fisher_p=0.936` external-lung, `0.353` immune) and acknowledged unstable in executor notes.
- `H02` from iter_0003 (`cross_model_alignment`) is still inconclusive (high raw agreement but weak null-calibrated significance).
- Topology claim is still vulnerable to null-family criticism because all positives so far use feature-shuffle null only.

## Machine-Artifacts Read
- `iterations/iter_0004/scgpt_cross_domain_h1_domain_summary.csv`
- `iterations/iter_0004/scgpt_cross_domain_h1_layer_summary.csv`
- `iterations/iter_0004/intrinsic_dimensionality_h1_correlation_summary.csv`
- `iterations/iter_0004/iter0004_screen_summary.json`
- Cumulative context: `reports/autoloop_master_log.md`, `paper/autoloop_research_paper.tex`

## Brainstormer Recommendation
- Treat topology as a robustness/causality branch now (stronger nulls + split robustness), not another replication branch.
- Keep manifold coupling active but swap out unstable local-ID estimators and pool all three domains in one meta-analysis.
- Re-open cross-model alignment only with residual-level matched-gene tensors; avoid spending another iteration on 3-feature summary vectors.
