"""
Spotify Music Feature Reduction using PCA
------------------------------------------
Phase 3: Cluster Visualization in PCA Space

This script generates:
1. PC1 vs PC2 scatter plot colored by K-Means cluster
   (sampled at 10,000 tracks for visual clarity and rendering performance)
2. Cluster sizes / distribution bar chart
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = PROJECT_ROOT / "results"

CLUSTERED_DATA_FILE = RESULTS_FOLDER / "kmeans_clustered_data.csv"
CLUSTER_SUMMARY_FILE = RESULTS_FOLDER / "cluster_summary.csv"

# Visualization Outputs
PCA_CLUSTER_PLOT_FILE = RESULTS_FOLDER / "kmeans_pca_clusters.png"
CLUSTER_SIZES_PLOT_FILE = RESULTS_FOLDER / "cluster_sizes.png"

# Visualization settings
PLOT_SAMPLE_SIZE = 10000
RANDOM_STATE = 42

# High-contrast, clean color palette for music clusters
CLUSTER_COLORS = [
    "#1DB954",  # Spotify Green
    "#E63946",  # Vibrant Red
    "#457B9D",  # Steel Blue
    "#F4A261",  # Sandy Orange
    "#9B5DE5",  # Purple
    "#00BBF9",  # Sky Blue
    "#F15BB5",  # Magenta
    "#2A9D8F",  # Teal
    "#E76F51",  # Burnt Coral
    "#D4A373",  # Tan
]


# ============================================================
# 2. LOAD CLUSTERED DATA
# ============================================================

def load_clustered_data():
    """Load the clustered dataset and summary."""
    if not CLUSTERED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clustered data not found at: {CLUSTERED_DATA_FILE}\n"
            "Please run clustering.py first."
        )

    df_clustered = pd.read_csv(CLUSTERED_DATA_FILE)
    df_summary = (
        pd.read_csv(CLUSTER_SUMMARY_FILE) if CLUSTER_SUMMARY_FILE.exists() else None
    )

    print(f"Loaded clustered data: {len(df_clustered):,} tracks")
    return df_clustered, df_summary


# ============================================================
# 3. PC1 vs PC2 CLUSTER SCATTER PLOT
# ============================================================

def plot_pca_clusters(df_clustered):
    """
    Plot PC1 vs PC2 scatter plot colored by cluster.
    Uses a representative sample of 10,000 tracks for readability and rendering efficiency.
    """
    print("\n" + "=" * 60)
    print("GENERATING PC1 vs PC2 CLUSTER SCATTER PLOT")
    print("=" * 60)

    total_points = len(df_clustered)
    unique_clusters = sorted(df_clustered["cluster"].unique())
    num_clusters = len(unique_clusters)

    # Sample for plotting if dataset is large
    if total_points > PLOT_SAMPLE_SIZE:
        print(
            f"Note: Full dataset has {total_points:,} tracks. Sampling {PLOT_SAMPLE_SIZE:,} "
            f"tracks (seed={RANDOM_STATE}) for scatter plot readability and performance."
        )
        plot_df = df_clustered.sample(n=PLOT_SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        plot_df = df_clustered

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Plot each cluster
    for idx, cluster_id in enumerate(unique_clusters):
        subset = plot_df[plot_df["cluster"] == cluster_id]
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            c=color,
            label=f"Cluster {cluster_id} (n={len(df_clustered[df_clustered['cluster'] == cluster_id]):,})",
            alpha=0.45,
            edgecolors="none",
            s=18,
        )

    # Plot cluster centroids in PC1-PC2 space
    centroids_pc1 = df_clustered.groupby("cluster")["PC1"].mean()
    centroids_pc2 = df_clustered.groupby("cluster")["PC2"].mean()

    ax.scatter(
        centroids_pc1,
        centroids_pc2,
        c="black",
        s=120,
        marker="X",
        edgecolors="white",
        linewidths=1.5,
        label="Cluster Centroids",
        zorder=5,
    )

    for cluster_id in unique_clusters:
        ax.annotate(
            f"C{cluster_id}",
            (centroids_pc1[cluster_id], centroids_pc2[cluster_id]),
            fontsize=11,
            fontweight="bold",
            color="black",
            xytext=(6, 6),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7, ec="none"),
            zorder=6,
        )

    ax.set_title("Spotify Tracks: K-Means Clusters in PCA Space", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("PC1 (Loudness, Energy vs. Acousticness — 32.12% var)", fontsize=11, labelpad=8)
    ax.set_ylabel("PC2 (Danceability, Valence — 15.87% var)", fontsize=11, labelpad=8)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        title=f"K-Means Clusters (K={num_clusters})",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(PCA_CLUSTER_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PC1 vs PC2 plot to: {PCA_CLUSTER_PLOT_FILE.name}")


# ============================================================
# 4. CLUSTER SIZES BAR CHART
# ============================================================

def plot_cluster_sizes(df_clustered):
    """Generate a clean bar chart showing track counts per cluster."""
    print("\n" + "=" * 60)
    print("GENERATING CLUSTER SIZE DISTRIBUTION CHART")
    print("=" * 60)

    cluster_counts = df_clustered["cluster"].value_counts().sort_index()
    unique_clusters = cluster_counts.index.tolist()
    counts = cluster_counts.values
    percentages = (counts / len(df_clustered)) * 100

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(unique_clusters))]

    bars = ax.bar(
        [f"Cluster {c}" for c in unique_clusters],
        counts,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        width=0.6,
    )

    # Add count and percentage labels above bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title("Track Distribution Across K-Means Clusters", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cluster ID", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Tracks", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CLUSTER_SIZES_PLOT_FILE, dpi=300)
    plt.close()
    print(f"Saved cluster sizes chart to: {CLUSTER_SIZES_PLOT_FILE.name}")


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("SPOTIFY MUSIC FEATURE REDUCTION - PHASE 3: VISUALIZATIONS")
    print("=" * 60)

    df_clustered, df_summary = load_clustered_data()
    plot_pca_clusters(df_clustered)
    plot_cluster_sizes(df_clustered)

    print("\n" + "=" * 60)
    print("PHASE 3 VISUALIZATIONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
