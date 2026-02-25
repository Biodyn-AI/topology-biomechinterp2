# Executor Next Steps: iter_0043

1. Retire tested formulations `H112/N565`, `H113/N552`, and `H114/N559` for promotion endpoints (all three failed null-gap robustness; `H112/H113` also failed directionality on primary objective).
2. Shift next execution packet away from retired families toward a non-retired family with stronger upside (recommended: `split_robustness` or `module_structure`) and keep only one rescue-once slot.
3. If motif dynamics are revisited, require a materially different endpoint than sequence AUROC deltas (e.g., perturbation-conditioned transfer objective with explicit external biological anchor), otherwise skip.
4. Reduce future runtime risk by pre-registering bounded null budgets and caching for topology-heavy branches before first run.
