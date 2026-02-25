"""Generate figures for the research paper from experimental CSV data.

Usage:
    python generate_figures.py [--iter ITERATIONS_DIR] [--out FIGURES_DIR]

By default, reads from ./iterations/ and writes to ./figures/.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['figure.dpi'] = 200

COLORS = {"lung": "#2196F3", "immune": "#4CAF50", "external_lung": "#FF9800"}
DOMAIN_LABELS = {"lung": "Lung", "immune": "Immune", "external_lung": "Ext-Lung"}


# ============================================================================
# Figure 1: Persistent homology across layers and domains (H01/H03)
# ============================================================================
def fig_persistent_homology(ITER, OUT):
    # Lung from iter_0003
    lung = pd.read_csv(f"{ITER}/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv")
    lung["domain"] = "lung"
    # Cross-domain from iter_0004
    cross = pd.read_csv(f"{ITER}/iter_0004/scgpt_cross_domain_h1_layer_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: H1 delta by layer for each domain
    ax = axes[0]
    for domain, color in COLORS.items():
        if domain == "lung":
            df = lung
        else:
            df = cross[cross["domain"] == domain]
        if len(df) == 0:
            continue
        df = df.sort_values("layer")
        ax.bar(df["layer"] + (list(COLORS.keys()).index(domain) - 1) * 0.25,
               df["mean_h1_sum_delta"], width=0.25, color=color, alpha=0.85,
               label=DOMAIN_LABELS[domain])
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("H1 persistence delta\n(observed $-$ null)")
    ax.set_title("(a) Topological signal across layers and domains")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(12))

    # Panel B: Significance (Fisher p) as -log10
    ax = axes[1]
    for domain, color in COLORS.items():
        if domain == "lung":
            df = lung
        else:
            df = cross[cross["domain"] == domain]
        if len(df) == 0:
            continue
        df = df.sort_values("layer")
        pvals = df["fisher_p"].clip(lower=1e-10)
        ax.plot(df["layer"], -np.log10(pvals), "o-", color=color, ms=3, lw=1.2,
                label=DOMAIN_LABELS[domain])
    ax.axhline(-np.log10(0.05), color="red", lw=0.8, ls=":", label="$p = 0.05$")
    ax.axhline(-np.log10(0.01), color="darkred", lw=0.8, ls=":", label="$p = 0.01$")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("$-\\log_{10}(p)$")
    ax.set_title("(b) Statistical significance")
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.set_xticks(range(0, 12, 2))

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_persistent_homology.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_persistent_homology.png")


# ============================================================================
# Figure 2: Cross-model CCA alignment (H24)
# ============================================================================
def fig_cross_model(ITER, OUT):
    df = pd.read_csv(f"{ITER}/iter_0013/h24_cross_model_cca_domain_summary.csv")

    metrics = [
        ("mean_canonical_corr", "null_mean_canonical_corr", "Canonical\ncorrelation"),
        ("distance_spearman_cca", "null_mean_distance_spearman", "Distance\nSpearman"),
        ("knn_jaccard_cca", "null_mean_knn_jaccard", "kNN\nJaccard"),
        ("top1_retrieval_cca", "null_mean_top1_retrieval", "Top-1\nretrieval"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(metrics))
    width = 0.25

    for i, domain in enumerate(["immune", "lung", "external_lung"]):
        ddf = df[df["domain"] == domain]
        if len(ddf) == 0:
            continue
        obs_vals = [ddf[m[0]].mean() for m in metrics]
        null_vals = [ddf[m[1]].mean() for m in metrics]
        bars = ax.bar(x + (i - 1) * width, obs_vals, width, color=COLORS[domain],
                      alpha=0.85, label=f"{DOMAIN_LABELS[domain]} (observed)")
        # null as horizontal markers
        for j, nv in enumerate(null_vals):
            ax.plot([x[j] + (i - 1) * width - width/2, x[j] + (i - 1) * width + width/2],
                    [nv, nv], color="black", lw=1.5, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in metrics], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Cross-model alignment: scGPT vs Geneformer (H24)\nBars = observed; dashed = null expectation")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cross_model_cca.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_cross_model_cca.png")


# ============================================================================
# Figure 3: Distance hierarchy (H13 manifold distance)
# ============================================================================
def fig_distance_hierarchy(ITER, OUT):
    df = pd.read_csv(f"{ITER}/iter_0010/h13_manifold_distance_layer_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for idx, split in enumerate(["source_disjoint", "target_disjoint"]):
        ax = axes[idx]
        sdf = df[df["split_regime"] == split].sort_values("layer")
        if len(sdf) == 0:
            continue
        ax.fill_between(sdf["layer"], 0, sdf["mean_delta_auc_geodesic_minus_euclidean"],
                        alpha=0.3, color="#2196F3")
        ax.plot(sdf["layer"], sdf["mean_delta_auc_geodesic_minus_euclidean"],
                "o-", color="#2196F3", ms=4, lw=1.5, label="Geodesic $-$ Euclidean")
        ax.axhline(0, color="gray", lw=0.5, ls="--")

        # Mark significant layers
        sig = sdf[sdf["fisher_p_delta_upper"] < 0.05]
        if len(sig) > 0:
            ax.scatter(sig["layer"], sig["mean_delta_auc_geodesic_minus_euclidean"],
                      marker="*", s=80, color="red", zorder=5, label="$p < 0.05$")

        split_label = "Source-disjoint" if "source" in split else "Target-disjoint"
        ax.set_title(f"({chr(97+idx)}) {split_label} gene pool")
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("$\\Delta$AUROC (geodesic $-$ Euclidean)")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xticks(range(0, 12, 2))

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_distance_hierarchy.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_distance_hierarchy.png")


# ============================================================================
# Figure 4: H123 signed motif-community — null-gap across all rows
# ============================================================================
def fig_motif_community(ITER, OUT):
    df = pd.read_csv(f"{ITER}/iter_0046/h123_signed_motif_module_hardening_by_seed_domain_split.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel A: Delta vs H70 by domain-split
    ax = axes[0]
    df["label"] = df["domain"].map(DOMAIN_LABELS).fillna(df["domain"]) + "\n" + df["split_regime"].str.replace("_disjoint", "").str.replace("_", " ")
    groups = df.groupby(["domain", "split_regime"])
    labels, deltas = [], []
    for (dom, spl), grp in sorted(groups):
        lab = DOMAIN_LABELS.get(dom, dom) + "\n" + spl.replace("_disjoint", "").replace("_", " ")
        labels.append(lab)
        deltas.append(grp["delta_vs_h70"].values)

    positions = range(len(labels))
    bp = ax.boxplot(deltas, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor="#81C784", alpha=0.7),
                    medianprops=dict(color="black", lw=1.5))
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$\\Delta$AUROC vs baseline")
    ax.set_title("(a) Effect size by domain-split")

    # Panel B: Null-gap (all positive = key finding)
    ax = axes[1]
    null_gaps = []
    for (dom, spl), grp in sorted(groups):
        null_gaps.append(grp["null_gap_q95"].values)

    bp2 = ax.boxplot(null_gaps, positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor="#FFB74D", alpha=0.7),
                     medianprops=dict(color="black", lw=1.5))
    ax.axhline(0, color="red", lw=1.0, ls="--", label="Null-gap = 0")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Null-gap (obs $-$ 95th pctl of null)")
    ax.set_title("(b) Null-gap: all rows positive")
    ax.legend(fontsize=8)

    fig.suptitle("Signed motif--community hardening (H123): strongest finding", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_motif_community.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_motif_community.png")


# ============================================================================
# Figure 5: Hypothesis outcome overview (positive/negative/inconclusive)
# ============================================================================
def fig_hypothesis_outcomes(ITER, OUT):
    """Create an overview showing the distribution of outcomes across 141 hypotheses.

    Note: The 9 categories below sum to 111, not 141. The remaining 30 hypotheses
    are cross-cutting methodological variants (topology stability tests,
    null-framework development, split-design validation) that span multiple
    families and are not assigned to a single content category.
    """
    categories = {
        "Persist.\nhomol.": (5, 1, 0),
        "Manifold\ndist.": (5, 4, 8),
        "Cross-\nmodel": (3, 1, 15),
        "Commun.\nstruct.": (4, 2, 5),
        "Directed\ntopol.": (3, 2, 2),
        "Intrinsic\ndim.": (0, 2, 6),
        "Sparse\ndescr.": (4, 1, 3),
        "Motif-\ncommun.": (2, 3, 4),
        "Other\ngeom.": (1, 5, 20),
    }

    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    families = list(categories.keys())
    pos = [c[0] for c in categories.values()]
    inc = [c[1] for c in categories.values()]
    neg = [c[2] for c in categories.values()]

    x = np.arange(len(families))
    w = 0.25

    ax.bar(x - w, pos, w, color="#4CAF50", alpha=0.85, label="Robust positive")
    ax.bar(x, inc, w, color="#FFC107", alpha=0.85, label="Inconclusive / partial")
    ax.bar(x + w, neg, w, color="#F44336", alpha=0.85, label="Decisively negative")

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=8)
    ax.set_ylabel("Number of hypotheses")
    ax.set_title("Outcome distribution across 141 hypotheses by family")
    ax.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
    ax.set_ylim(0, max(neg) + 3)

    # Add total counts
    for i in range(len(families)):
        total = pos[i] + inc[i] + neg[i]
        ax.text(i, max(pos[i], inc[i], neg[i]) + 0.5, f"n={total}",
                ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hypothesis_outcomes.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_hypothesis_outcomes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures from iteration CSV data.")
    parser.add_argument("--iter", default="iterations",
                        help="Path to iterations directory (default: ./iterations)")
    parser.add_argument("--out", default="figures",
                        help="Path to output figures directory (default: ./figures)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Generating figures...")
    fig_persistent_homology(args.iter, args.out)
    fig_cross_model(args.iter, args.out)
    fig_distance_hierarchy(args.iter, args.out)
    fig_motif_community(args.iter, args.out)
    fig_hypothesis_outcomes(args.iter, args.out)
    print("Done.")
