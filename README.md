# Autonomous Topology Hypothesis Screening for Gene-Expression Transformer Embeddings

This repository contains the full codebase, data outputs, and documentation for an autonomous hypothesis screening campaign that tested 141 geometric and topological hypotheses on scGPT and Geneformer gene-embedding representations across 53 iterations.

## Overview

**Research question:** Do single-cell gene-expression transformers (scGPT, Geneformer) encode non-trivial geometric or topological structure in their embedding spaces that reflects known biological regulation?

**Approach:** An autonomous executor-brainstormer loop, driven by OpenAI Codex 5.3 ("extra high" reasoning setting), systematically generated, executed, and evaluated hypotheses spanning persistent homology, manifold geometry, cross-model alignment, community structure, and regulatory motifs. Each hypothesis was tested against multiple null models and evaluated using disjoint gene-pool splits to prevent information leakage.

**Key findings:**
- **~19% positive rate** (27/141 hypotheses), **~15% inconclusive**, **~45% decisively negative**, with the remaining 30 being cross-cutting methodological variants
- Under a strict max-null audit, fewer than 15 hypotheses (~10%) survive
- The strongest finding, **signed motif-community hardening (H123)**, achieves positive null-gap in every domain-split group tested --- the only hypothesis to do so
- Two independently trained models (scGPT and Geneformer) converge on similar geometric organization
- Persistent homology reveals non-trivial H1 (loop) structure that significantly exceeds null expectations
- Geodesic manifold distances modestly but consistently outperform Euclidean for regulatory edge discrimination

## Repository Structure

```
.
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── generate_figures.py         # Reproduces all paper figures from CSV data
├── .gitignore
│
├── loop/                       # Autonomous loop infrastructure
│   ├── run_codex_topology_autoloop.py   # Main orchestrator (executor-brainstormer loop)
│   ├── config.json             # Loop configuration (relative paths)
│   ├── start_codex_autoloop.sh # Launch script (double-fork daemon)
│   ├── stop_codex_autoloop.sh  # Graceful stop script
│   └── status_codex_autoloop.sh # Status check script
│
├── prompts/                    # Agent prompt templates
│   ├── executor_prompt_topology_hypothesis_screening.md
│   └── brainstormer_prompt_template.md
│
├── planning/                   # Research design
│   └── research_plan.md        # Initial hypothesis slate and iteration protocol
│
├── reports/                    # Cumulative logs
│   └── autoloop_master_log.md  # Full iteration-by-iteration event log (113 KB)
│
├── runtime/                    # Loop state snapshot (from final iteration)
│   ├── loop_status.json        # Final loop status
│   ├── loop_events.jsonl       # All loop events (one JSON per line)
│   ├── latest_brainstormer_feedback.md
│   └── STOP                    # Stop sentinel (loop completed)
│
├── figures/                    # Pre-generated paper figures (PNG)
│   ├── fig_persistent_homology.png
│   ├── fig_cross_model_cca.png
│   ├── fig_distance_hierarchy.png
│   ├── fig_motif_community.png
│   ├── fig_hypothesis_outcomes.png
│   ├── codex_h123_vs_h139_null_gap.png
│   ├── codex_h140_scaling_gain.png
│   └── codex_h141_strict_margin_audit.png
│
└── iterations/                 # All 53 iteration directories
    ├── iter_0001/              # Each contains:
    │   ├── run_iter0001_screen.py           # Auto-generated experiment script
    │   ├── executor_hypothesis_screen.json  # Structured hypothesis outcomes
    │   ├── executor_iteration_report.md     # Narrative report
    │   ├── *.csv                            # Quantitative results
    │   └── ...
    ├── iter_0002/
    ├── ...
    └── iter_0053/
```

## Reproducing Figures

All paper figures can be regenerated from the iteration CSV data:

```bash
pip install pandas matplotlib numpy
python generate_figures.py
```

This reads from `iterations/` and writes to `figures/`. Use `--iter` and `--out` flags to customize paths.

### Critical CSV files for paper figures

| Figure | Hypothesis | CSV file | Iteration |
|--------|-----------|----------|-----------|
| Fig. 1 (persistent homology) | H01/H03 | `scgpt_lung_h1_persistence_layer_summary.csv` | iter_0003 |
| Fig. 1 (cross-domain) | H03 | `scgpt_cross_domain_h1_layer_summary.csv` | iter_0004 |
| Fig. 2 (cross-model CCA) | H24 | `h24_cross_model_cca_domain_summary.csv` | iter_0013 |
| Fig. 3 (distance hierarchy) | H13 | `h13_manifold_distance_layer_summary.csv` | iter_0010 |
| Fig. 4 (motif-community) | H123 | `h123_signed_motif_module_hardening_by_seed_domain_split.csv` | iter_0046 |

## Autonomous Loop Architecture

The loop (`loop/run_codex_topology_autoloop.py`) alternates between two LLM-driven phases:

1. **Executor phase:** Receives a hypothesis brief, writes and executes Python experiment scripts, produces structured JSON outcomes and CSV data, and updates the running paper draft.

2. **Brainstormer phase:** Reviews executor outputs, retires negative directions, generates a portfolio of 12-16 new hypothesis ideas, and selects the top 3 for the next iteration (one high-probability, one high-risk/high-reward, one cheap broad-screen).

Both agents are instantiated via OpenAI Codex 5.3 with "extra high" reasoning. The loop runs as a daemonized process with heartbeat monitoring and graceful stop support.

### Hypothesis families

The campaign rotated across nine hypothesis families:

| Family | Description |
|--------|-------------|
| Persistent homology | Betti curves, H1 lifetime summaries of embedding neighborhoods |
| Manifold distance | Geodesic vs. Euclidean distance tests for regulatory proximity |
| Cross-model alignment | scGPT vs. Geneformer manifold comparison (CCA, Procrustes, CKA) |
| Community structure | kNN community detection vs. TRRUST/GO/STRING annotations |
| Directed topology | Signed/directed features from regulatory edge directionality |
| Intrinsic dimensionality | Local linearity and dimensionality diagnostics by layer |
| Sparse descriptors | Stability-selected sparse blending of geometric features |
| Motif-community | Regulatory motif presence combined with geometric community |
| Other geometric | Curvature proxies, anisotropy, convexity, and miscellaneous |

## Null Model Hierarchy

Every hypothesis was tested against at least one null model:

- **Feature-shuffle null:** Randomly permute gene embeddings within each layer (20-24 replicates)
- **Label-permutation null:** Randomly reassign regulatory labels while preserving graph statistics (120-200 replicates)
- **Degree-preserving graph rewiring:** Rewire regulatory edges while preserving node degree distribution
- **Coexpression-matched null:** Stratified by Pearson correlation (5 bins) and degree (5 bins) to control for expression similarity confounds

The **strict max-null audit** (H141) computes the observed signal minus the maximum 95th percentile across all null families, providing the most conservative estimate.

## Evaluation Design

- **Source-disjoint splits:** Held-out transcription factors (TFs) not seen during feature computation
- **Target-disjoint splits:** Held-out target genes
- **Dual-axis disjoint splits:** Both TFs and targets held out simultaneously
- **Primary metric:** AUROC for regulatory edge discrimination
- **Null-gap:** Observed metric minus the 95th percentile of the null distribution

## Data Dependencies

The experiments in this repository were run against:

- **scGPT** (Cui et al., 2024): 12-layer transformer, embeddings extracted from Tabula Sapiens atlas tissues (lung, immune, external-lung)
- **Geneformer V2-316M** (Theodoris et al., 2023): 18-layer, 18-head transformer
- **Tabula Sapiens** (The Tabula Sapiens Consortium, 2022): Single-cell atlas providing tissue-specific expression profiles
- **TRRUST v2** (Han et al., 2018): Transcriptional regulatory network (human)
- **STRING** (Szklarczyk et al., 2019): Protein-protein interaction network
- **Gene Ontology** (Ashburner et al., 2000): Functional annotations

Pre-computed embeddings are not included in this repository due to size. Each iteration script (`run_iter*_screen.py`) documents the embedding extraction procedure used.

## Claims-to-Evidence Mapping

The following table maps key paper claims to their supporting evidence in this repository:

| Claim | Evidence | Location |
|-------|----------|----------|
| 141 hypotheses tested across 53 iterations | Per-iteration JSON files | `iterations/iter_*/executor_hypothesis_screen.json` |
| Cross-model alignment (scGPT-Geneformer) exceeds null | CCA domain summary with null comparisons | `iterations/iter_0013/h24_cross_model_cca_domain_summary.csv` |
| H1 persistent homology exceeds null in 11/12 layers | Layer-by-layer persistence delta and p-values | `iterations/iter_0003/` and `iter_0004/` |
| Geodesic > Euclidean for regulatory edges (ΔAUROC ≈ 0.01) | Layer summary with split-regime stratification | `iterations/iter_0010/h13_manifold_distance_layer_summary.csv` |
| H123 (signed motif-community) achieves 100% null-gap coverage | Per-seed, per-domain, per-split results | `iterations/iter_0046/h123_signed_motif_module_hardening_by_seed_domain_split.csv` |
| Strict max-null audit: fewer than 15 hypotheses survive | Max-null fragility audit results | `iterations/iter_0050/` through `iter_0053/` |
| ~45% of hypotheses are decisively negative | Cumulative hypothesis screen JSONs | `iterations/iter_*/executor_hypothesis_screen.json` |
| Immune domain most robust under strict audit | Domain-stratified strict margin results | `iterations/iter_0050/` through `iter_0053/` |

## Hypothesis Index

A complete index of all 141 hypotheses with their outcomes can be reconstructed from the structured JSON files:

```bash
# Extract all hypothesis outcomes across all iterations
python -c "
import json, glob, os
for f in sorted(glob.glob('iterations/iter_*/executor_hypothesis_screen.json')):
    with open(f) as fh:
        data = json.load(fh)
    for h in data.get('hypotheses', []):
        print(f\"{h.get('id','?'):>6s}  {h.get('decision','?'):>12s}  {h.get('name','?')}\")
"
```

## Running the Loop (for reference)

The loop was run to completion (53 iterations) and is not designed to be re-run from scratch without the original data dependencies. The code is provided for transparency and reproducibility of the methodology:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (edit loop/config.json with your paths)
# Requires: OpenAI Codex CLI, scGPT/Geneformer models, Tabula Sapiens data

# Start the loop
bash loop/start_codex_autoloop.sh

# Check status
bash loop/status_codex_autoloop.sh

# Stop gracefully
bash loop/stop_codex_autoloop.sh
```

## License

This research code is provided for academic and research purposes. Please cite the accompanying paper if you use this work.

## References

- Cui, H., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*.
- Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. *Nature*.
- The Tabula Sapiens Consortium (2022). The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans. *Science*.
- Han, H., et al. (2018). TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. *Nucleic Acids Research*.
- Szklarczyk, D., et al. (2019). STRING v11: protein-protein association networks with increased coverage. *Nucleic Acids Research*.
- Ashburner, M., et al. (2000). Gene Ontology: tool for the unification of biology. *Nature Genetics*.
