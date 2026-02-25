# Executor Next Steps — iter_0019

1. Promote `H40` with stricter priors and controls:
   - Add true STRING edge-score prior (if surfaced) and rerun the same interaction packet.
   - Require out-of-domain robustness gate (at least 2/3 domains with Fisher-significant interaction and positive top-decile uplift).

2. Keep `H41` as a tooling-gated branch:
   - Install a zigzag-capable TDA package and replace proxy with explicit split-zigzag persistence.
   - Reuse the same split-swap and layer-order controls for comparability.

3. Deprioritize `H42` mechanism claim unless OOS stability is recovered:
   - Use robust OOS metrics (trimmed/winsorized `R^2` or MAE-based scoring) to prevent single-holdout blowups.
   - If instability persists, retire H38-style moment mechanisms for this autoloop phase.
