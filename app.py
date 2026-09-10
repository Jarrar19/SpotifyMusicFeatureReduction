"""
Spotify Music Feature Reduction using PCA & K-Means Clustering
---------------------------------------------------------------
Interactive Streamlit Application for PBL Project Deployment

Authors: Antigravity Pair Programming
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Spotify Music PCA & Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium styling with Spotify accents
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #181818 0%, #282828 100%);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #1DB954;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #B3B3B3;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .cluster-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 12px;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. PATHS & REPRODUCIBLE PALETTES
# ============================================================

APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"

# Data file paths
CLUSTERED_DATA_FILE = RESULTS_DIR / "kmeans_clustered_data.csv"
CLUSTERED_SAMPLE_FILE = RESULTS_DIR / "kmeans_clustered_sample.csv"
CLUSTER_SUMMARY_FILE = RESULTS_DIR / "cluster_summary.csv"
CLUSTER_AUDIO_MEANS_FILE = RESULTS_DIR / "cluster_audio_feature_means.csv"
CLUSTER_GENRES_FILE = RESULTS_DIR / "cluster_genre_distribution.csv"
PCA_VARIANCE_FILE = RESULTS_DIR / "cleaned_pca_explained_variance.csv"
PCA_LOADINGS_FILE = RESULTS_DIR / "cleaned_pca_loadings.csv"
ELBOW_SCORES_FILE = RESULTS_DIR / "kmeans_elbow_scores.csv"
SILHOUETTE_SCORES_FILE = RESULTS_DIR / "kmeans_silhouette_scores.csv"

# Image paths
IMG_ELBOW = RESULTS_DIR / "kmeans_elbow.png"
IMG_SILHOUETTE = RESULTS_DIR / "kmeans_silhouette.png"
IMG_PCA_CLUSTERS = RESULTS_DIR / "kmeans_pca_clusters.png"
IMG_CLUSTER_SIZES = RESULTS_DIR / "cluster_sizes.png"
IMG_EXPLAINED_VAR = RESULTS_DIR / "explained_variance.png"

CLUSTER_COLORS = [
    "#1DB954",  # Spotify Green
    "#E63946",  # Red
    "#457B9D",  # Steel Blue
    "#F4A261",  # Sandy Orange
    "#9B5DE5",  # Purple
    "#00BBF9",  # Sky Blue
    "#F15BB5",  # Magenta
]

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


# ============================================================
# 3. CACHED DATA LOADERS
# ============================================================

@st.cache_data
def load_summary_metrics():
    """Load core metric datasets."""
    df_var = pd.read_csv(PCA_VARIANCE_FILE) if PCA_VARIANCE_FILE.exists() else None
    df_summary = pd.read_csv(CLUSTER_SUMMARY_FILE) if CLUSTER_SUMMARY_FILE.exists() else None
    df_audio_means = pd.read_csv(CLUSTER_AUDIO_MEANS_FILE) if CLUSTER_AUDIO_MEANS_FILE.exists() else None
    df_genres = pd.read_csv(CLUSTER_GENRES_FILE) if CLUSTER_GENRES_FILE.exists() else None
    df_elbow = pd.read_csv(ELBOW_SCORES_FILE) if ELBOW_SCORES_FILE.exists() else None
    df_sil = pd.read_csv(SILHOUETTE_SCORES_FILE) if SILHOUETTE_SCORES_FILE.exists() else None
    df_loadings = pd.read_csv(PCA_LOADINGS_FILE) if PCA_LOADINGS_FILE.exists() else None
    return df_var, df_summary, df_audio_means, df_genres, df_elbow, df_sil, df_loadings


@st.cache_data
def load_clustered_tracks(sample_for_interactive=True):
    """
    Load track dataset for interactive visualization and track explorer.
    Falls back gracefully to sample if full dataset is not available or too large.
    """
    if CLUSTERED_DATA_FILE.exists():
        if sample_for_interactive:
            # Load 5,000-sample slice for swift interactive rendering in browser
            if CLUSTERED_SAMPLE_FILE.exists():
                return pd.read_csv(CLUSTERED_SAMPLE_FILE)
            df_full = pd.read_csv(CLUSTERED_DATA_FILE)
            return df_full.sample(n=min(5000, len(df_full)), random_state=42)
        else:
            return pd.read_csv(CLUSTERED_DATA_FILE)
    elif CLUSTERED_SAMPLE_FILE.exists():
        return pd.read_csv(CLUSTERED_SAMPLE_FILE)
    return None


# Load cached files
df_var, df_summary, df_audio_means, df_genres, df_elbow, df_sil, df_loadings = load_summary_metrics()


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg",
    width=60,
)
st.sidebar.title("Navigation")

navigation_options = [
    "📊 Executive Dashboard",
    "📉 PCA Reduction Analysis",
    "📐 K-Means Evaluation (Elbow & Silhouette)",
    "🎯 Cluster Profiles & Audio Characteristics",
    "🗺️ 2D PCA Cluster Map",
    "🔍 Interactive Track Explorer",
]

selected_page = st.sidebar.radio("Select View:", navigation_options)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Highlights:**
    - **Raw Data:** 114,000 Tracks
    - **Deduplicated:** 89,741 Tracks
    - **Audio Features:** 9 Features
    - **PCA Reduction:** 7 PCs (94.81% Var)
    - **K-Means Model:** K = 7 Clusters
    """
)
st.sidebar.info("💡 Built with Streamlit, Scikit-Learn & Plotly for PBL Deployment.")


# ============================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================

if selected_page == "📊 Executive Dashboard":
    st.title("🎵 Spotify Music Feature Reduction & Clustering")
    st.markdown(
        "Dimensionality reduction using **Principal Component Analysis (PCA)** "
        "and unsupervised segmentation with **K-Means Clustering**."
    )

    # Top KPI Metrics Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Cleaned Tracks</div>
                <div class="metric-value">89,741</div>
                <small style='color: #888;'>From 114k raw (-21.3% dupes)</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Original Features</div>
                <div class="metric-value">9</div>
                <small style='color: #888;'>Standardized Audio Metrics</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">PCA Components</div>
                <div class="metric-value">7 PCs</div>
                <small style='color: #888;'>Target ≥ 90% Variance</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Variance Retained</div>
                <div class="metric-value">94.81%</div>
                <small style='color: #888;'>Minimal Information Loss</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Selected K Clusters</div>
                <div class="metric-value">K = 7</div>
                <small style='color: #888;'>Elbow & Silhouette Backed</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Workflow Diagram & Project Summary
    st.subheader("End-to-End Pipeline Architecture")
    st.markdown(
        """
        ```text
        Spotify Dataset (114,000 Tracks)
                      ↓
        Track Deduplication (89,741 Unique Tracks by track_id)
                      ↓
        Extract 9 Standardized Audio Features (danceability, energy, loudness, tempo, etc.)
                      ↓
        Principal Component Analysis (9 Components fitted -> 7 PCs selected for 94.81% variance)
                      ↓
        K-Means Clustering Evaluation (Elbow Method & Silhouette Scores evaluated across K=2..10)
                      ↓
        Selected K = 7 Clusters (Elbow bend + multi-cluster silhouette peak at 0.2108)
                      ↓
        Profiling & Interactive Exploration (Audio feature means, dominant genres & 2D PCA mapping)
        ```
        """
    )

    st.markdown("---")
    st.subheader("Cluster Distribution Snapshot")
    if df_summary is not None:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_bar = px.bar(
                df_summary,
                x=[f"Cluster {c}" for c in df_summary["cluster"]],
                y="track_count",
                color=[f"Cluster {c}" for c in df_summary["cluster"]],
                color_discrete_sequence=CLUSTER_COLORS,
                labels={"x": "Cluster", "y": "Track Count", "color": "Cluster"},
                title="Number of Tracks per K-Means Cluster",
                text="track_count",
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_bar.update_layout(showlegend=False, template="plotly_dark", height=380)
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            fig_pie = px.pie(
                df_summary,
                values="percentage",
                names=[f"Cluster {c}" for c in df_summary["cluster"]],
                title="Percentage Share of Tracks",
                color_discrete_sequence=CLUSTER_COLORS,
                hole=0.4,
            )
            fig_pie.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================
# PAGE 2: PCA REDUCTION ANALYSIS
# ============================================================

elif selected_page == "📉 PCA Reduction Analysis":
    st.title("📉 Principal Component Analysis (PCA)")
    st.markdown(
        "PCA reduces 9 Spotify audio features into uncorrelated orthogonal components. "
        "The first 7 components retain **94.81%** of total variance, achieving significant dimension reduction with minimal information loss."
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        if df_var is not None:
            # Plot Cumulative Explained Variance
            var_col = "Cumulative Variance (%)" if "Cumulative Variance (%)" in df_var.columns else df_var.columns[2]
            ind_col = "Explained Variance (%)" if "Explained Variance (%)" in df_var.columns else df_var.columns[1]
            pc_col = df_var.columns[0]

            fig_var = go.Figure()
            fig_var.add_trace(
                go.Bar(
                    x=df_var[pc_col],
                    y=df_var[ind_col],
                    name="Individual Variance (%)",
                    marker_color="#4A90E2",
                )
            )
            fig_var.add_trace(
                go.Scatter(
                    x=df_var[pc_col],
                    y=df_var[var_col],
                    name="Cumulative Variance (%)",
                    line=dict(color="#1DB954", width=3),
                    mode="lines+markers",
                )
            )
            fig_var.add_hline(
                y=90,
                line_dash="dash",
                line_color="red",
                annotation_text="90% Threshold",
                annotation_position="bottom right",
            )
            fig_var.update_layout(
                title="Explained & Cumulative Variance by Principal Component",
                xaxis_title="Principal Component",
                yaxis_title="Variance (%)",
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=420,
            )
            st.plotly_chart(fig_var, use_container_width=True)

    with col2:
        st.subheader("Variance Breakdown")
        if df_var is not None:
            st.dataframe(df_var, use_container_width=True, hide_index=True)
            st.info(
                "💡 **Key Observation:** The 7-PC model captures **94.81%** of the information. "
                "The 2D plane (PC1 + PC2) captures **47.99%**, making it suitable for intuitive 2D projection."
            )

    st.markdown("---")
    st.subheader("PCA Loading Matrix (Feature Contributions)")
    st.markdown("Loadings represent the weights and directional influence of each original audio feature on each PC.")
    if df_loadings is not None:
        st.dataframe(df_loadings, use_container_width=True)
    elif IMG_EXPLAINED_VAR.exists():
        st.image(str(IMG_EXPLAINED_VAR), caption="Explained Variance Curve", use_column_width=True)


# ============================================================
# PAGE 3: K-MEANS EVALUATION (ELBOW & SILHOUETTE)
# ============================================================

elif selected_page == "📐 K-Means Evaluation (Elbow & Silhouette)":
    st.title("📐 K-Means Cluster Optimization (K = 2 to 10)")
    st.markdown(
        r"To select the optimal number of clusters without arbitrary guesswork, we evaluated $K \in [2, 10]$ "
        r"using both the **Elbow Method (WCSS Inertia)** on all 89,741 tracks and **Silhouette Analysis** "
        r"on a representative 10,000-track sample."
    )

    if df_elbow is not None and df_sil is not None:
        eval_df = pd.merge(df_elbow, df_sil, on="k")
        eval_df["pct_drop"] = eval_df["inertia"].pct_change().abs() * 100

        col1, col2 = st.columns(2)
        with col1:
            fig_elb = px.line(
                eval_df,
                x="k",
                y="inertia",
                markers=True,
                title="Elbow Method: Within-Cluster Sum of Squares (WCSS)",
                labels={"k": "Number of Clusters (K)", "inertia": "Inertia / WCSS"},
            )
            fig_elb.update_traces(line_color="#1DB954", marker=dict(size=9, color="#1DB954"))
            fig_elb.add_vline(x=7, line_dash="dash", line_color="orange", annotation_text="Elbow Inflection (K=7)")
            fig_elb.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_elb, use_container_width=True)

        with col2:
            fig_sil = px.line(
                eval_df,
                x="k",
                y="silhouette_score",
                markers=True,
                title="Silhouette Score vs. K (10,000 Sample Tracks)",
                labels={"k": "Number of Clusters (K)", "silhouette_score": "Silhouette Score"},
            )
            fig_sil.update_traces(line_color="#4A90E2", marker=dict(size=9, color="#4A90E2"))
            fig_sil.add_vline(x=7, line_dash="dash", line_color="orange", annotation_text="Multi-Cluster Peak (K=7)")
            fig_sil.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_sil, use_container_width=True)

        st.markdown("### Numerical Evaluation Table")
        display_eval = eval_df.rename(
            columns={
                "k": "K",
                "inertia": "Inertia (WCSS)",
                "pct_drop": "% Inertia Drop",
                "silhouette_score": "Silhouette Score",
            }
        )
        st.dataframe(
            display_eval.style.format(
                {"Inertia (WCSS)": "{:,.2f}", "% Inertia Drop": "{:.2f}%", "Silhouette Score": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            r"""
            > [!IMPORTANT]
            > **Cluster Selection Rationale (K = 7):**
            > **$K = 7$ was selected as the most useful multi-cluster solution.**
            > - **Global Silhouette ($K = 2$)**: $K = 2$ achieved the highest silhouette score (**0.2720**), but represents a coarse two-group partition (high-energy produced tracks vs. quiet acoustic tracks).
            > - **Multi-Cluster Peak ($K \ge 3$)**: Among granular candidate configurations ($K \ge 3$), **$K = 7$ provided the strongest silhouette score (0.2108)**.
            > - **Elbow & Diminishing Returns**: Inertia drops steadily by ~10% per step through $K=7$ (WCSS: 591,133 $\to$ 337,715). At $K=8$, the reduction drops abruptly to **6.06%**, confirming that $K=7$ is consistent with the elbow/diminishing-return region.
            > - **Musical Nuance**: $K=7$ uncovers meaningful music groupings, separating spoken-word/comedy (C0), heavy rock/metal (C1), upbeat pop/Latin (C2), ambient/classical (C3), live Brazilian rhythm (C4), acoustic ballads (C5), and electronic/techno (C6).
            """
        )


# ============================================================
# PAGE 4: CLUSTER PROFILES & INTERPRETATION
# ============================================================

elif selected_page == "🎯 Cluster Profiles & Audio Characteristics":
    st.title("🎯 Cluster Profiles & Musical Interpretation")
    st.markdown(
        "By examining the average values of the **original 9 audio features** and the **dominant genres** "
        "associated with each cluster, we can interpret what each musical cluster represents in practice."
    )

    if df_audio_means is not None and df_genres is not None:
        # Cluster Selection
        cluster_list = sorted(df_audio_means["cluster"].unique())
        selected_c = st.selectbox(
            "Select Cluster to Profile:",
            cluster_list,
            format_func=lambda x: f"Cluster {x} ({df_audio_means.loc[df_audio_means['cluster']==x, 'percentage'].values[0]:.1f}% of tracks)",
        )

        row_c = df_audio_means[df_audio_means["cluster"] == selected_c].iloc[0]
        genres_c = df_genres[df_genres["cluster"] == selected_c]

        # Top details
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Tracks", f"{int(row_c['track_count']):,}")
        with c2:
            st.metric("Dataset Share", f"{row_c['percentage']:.2f}%")
        with c3:
            st.metric("Avg Tempo", f"{row_c['tempo']:.1f} BPM")
        with c4:
            st.metric("Avg Loudness", f"{row_c['loudness']:.1f} dB")

        col_radar, col_genres = st.columns([1.2, 1])

        with col_radar:
            # Radar chart of 0-1 bounded audio features
            radar_feats = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence"]
            cluster_vals = [row_c[f] for f in radar_feats]
            overall_vals = [df_audio_means[f].mean() for f in radar_feats]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=cluster_vals + [cluster_vals[0]],
                    theta=radar_feats + [radar_feats[0]],
                    fill="toself",
                    name=f"Cluster {selected_c}",
                    line_color=CLUSTER_COLORS[selected_c % len(CLUSTER_COLORS)],
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=overall_vals + [overall_vals[0]],
                    theta=radar_feats + [radar_feats[0]],
                    fill="none",
                    name="All Tracks Average",
                    line=dict(color="#888888", dash="dot"),
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                template="plotly_dark",
                title=f"Audio Feature Radar: Cluster {selected_c}",
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_genres:
            st.subheader(f"Top Genres in Cluster {selected_c}")
            fig_genre = px.bar(
                genres_c,
                x="genre_percentage_in_cluster",
                y="genre",
                orientation="h",
                color_discrete_sequence=[CLUSTER_COLORS[selected_c % len(CLUSTER_COLORS)]],
                labels={"genre_percentage_in_cluster": "Share in Cluster (%)", "genre": "Genre"},
                title=f"Dominant Genres (Cluster {selected_c})",
            )
            fig_genre.update_layout(
                template="plotly_dark",
                height=380,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_genre, use_container_width=True)

        st.markdown("---")
        st.subheader("All Clusters Comparison Table")
        st.dataframe(
            df_audio_means.style.format(
                {
                    "track_count": "{:,}",
                    "percentage": "{:.2f}%",
                    "danceability": "{:.3f}",
                    "energy": "{:.3f}",
                    "loudness": "{:.2f}",
                    "speechiness": "{:.3f}",
                    "acousticness": "{:.3f}",
                    "instrumentalness": "{:.3f}",
                    "liveness": "{:.3f}",
                    "valence": "{:.3f}",
                    "tempo": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 5: 2D PCA CLUSTER MAP
# ============================================================

elif selected_page == "🗺️ 2D PCA Cluster Map":
    st.title("🗺️ 2D PCA Cluster Projection (PC1 vs. PC2)")
    st.markdown(
        "Interactive 2D projection along the two dominant principal components (**PC1: 32.12% variance**, **PC2: 15.87% variance**). "
        "PC1 separates loud/high-energy produced songs from quiet acoustic songs, while PC2 captures danceability and valence."
    )

    df_tracks = load_clustered_tracks(sample_for_interactive=True)

    if df_tracks is not None:
        c1, c2 = st.columns([3, 1])
        with c2:
            st.markdown("#### Filter Clusters")
            all_clusters = sorted(df_tracks["cluster"].unique())
            active_clusters = st.multiselect("Display Clusters:", all_clusters, default=all_clusters)
            sample_size = st.slider("Display Points:", 1000, min(10000, len(df_tracks)), 3000, step=1000)

        filtered_df = df_tracks[df_tracks["cluster"].isin(active_clusters)].sample(
            n=min(sample_size, len(df_tracks[df_tracks["cluster"].isin(active_clusters)])),
            random_state=42,
        )

        with c1:
            fig_scatter = px.scatter(
                filtered_df,
                x="PC1",
                y="PC2",
                color=filtered_df["cluster"].astype(str),
                color_discrete_sequence=CLUSTER_COLORS,
                hover_data=["track_name", "artists", "track_genre", "popularity"],
                labels={"color": "Cluster"},
                title=f"Spotify Tracks in PCA Space ({len(filtered_df):,} Sample Points)",
                opacity=0.6,
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                height=580,
                xaxis_title="PC1 (Loudness, Energy vs. Acousticness)",
                yaxis_title="PC2 (Danceability, Valence)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    elif IMG_PCA_CLUSTERS.exists():
        st.image(str(IMG_PCA_CLUSTERS), caption="K-Means Clusters in PCA Space (PC1 vs. PC2)", use_column_width=True)


# ============================================================
# PAGE 6: INTERACTIVE TRACK EXPLORER
# ============================================================

elif selected_page == "🔍 Interactive Track Explorer":
    st.title("🔍 Interactive Spotify Track Explorer")
    st.markdown("Search tracks, inspect their assigned PCA cluster, and analyze their audio coordinates.")

    df_tracks = load_clustered_tracks(sample_for_interactive=False)

    if df_tracks is not None:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search_query = st.text_input("Search Track or Artist Name:", placeholder="e.g. Queen, Taylor Swift, Chopin...")
        with c2:
            genre_filter = st.selectbox(
                "Filter Genre:",
                ["All"] + sorted([g for g in df_tracks["track_genre"].dropna().unique()]),
            )
        with c3:
            cluster_filter = st.selectbox(
                "Filter Cluster:",
                ["All"] + [f"Cluster {c}" for c in sorted(df_tracks["cluster"].unique())],
            )

        # Apply filtering
        results = df_tracks.copy()
        if search_query:
            query = search_query.lower()
            results = results[
                results["track_name"].fillna("").str.lower().str.contains(query)
                | results["artists"].fillna("").str.lower().str.contains(query)
            ]
        if genre_filter != "All":
            results = results[results["track_genre"] == genre_filter]
        if cluster_filter != "All":
            c_num = int(cluster_filter.split(" ")[1])
            results = results[results["cluster"] == c_num]

        st.markdown(f"**Found {len(results):,} matching tracks** (Showing top 50)")

        cols_to_show = [
            "track_name",
            "artists",
            "album_name",
            "track_genre",
            "popularity",
            "cluster",
            "PC1",
            "PC2",
        ]
        available_cols = [c for c in cols_to_show if c in results.columns]

        st.dataframe(
            results[available_cols].head(50),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Track dataset not found. Please run src/clustering.py to generate clustered data.")
