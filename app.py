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
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Spotify Music PCA & Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern light theme with Spotify accents & top navbar
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    .main {
        background-color: #F8FAFC;
    }

    /* Branded Top Header */
    .brand-header-box {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }
    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 13px;
        color: #64748B;
        margin: 0;
    }

    /* Top Navbar Segmented Control */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 8px;
        background: #FFFFFF;
        padding: 8px 14px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
    }
    div[data-testid="stRadio"] label {
        display: flex;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 8px 16px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 0;
    }
    /* Hide the radio input circle completely */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #F1F5F9;
        border-color: #CBD5E1;
        color: #0F172A;
    }
    /* Active Navbar Button */
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #1DB954 !important;
        border-color: #1DB954 !important;
        box-shadow: 0 4px 14px rgba(29, 185, 84, 0.3) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Light Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: #1DB954;
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(29, 185, 84, 0.12);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }

    /* Audio Player & Recommendation Cards */
    .player-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
    }
    .recommendation-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #0F172A;
        transition: all 0.2s ease;
    }
    .recommendation-item:hover {
        background: #F1F5F9;
        border-color: #1DB954;
        transform: translateX(4px);
    }

    /* Cluster Tag Badge */
    .cluster-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        margin-right: 6px;
        letter-spacing: 0.3px;
    }

    /* Section Headers */
    .section-title {
        color: #0F172A;
        font-weight: 800;
        letter-spacing: -0.5px;
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
GENERIC_SAMPLE_FILE = RESULTS_DIR / "generic_sample_customer_data.csv"
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

# Vibrant cluster palette optimized for light background contrast
CLUSTER_COLORS = [
    "#10B981",  # Emerald Green (Cluster 0)
    "#EF4444",  # Coral Red (Cluster 1)
    "#2563EB",  # Royal Blue (Cluster 2)
    "#F59E0B",  # Amber Yellow (Cluster 3)
    "#8B5CF6",  # Purple (Cluster 4)
    "#06B6D4",  # Cyan (Cluster 5)
    "#EC4899",  # Magenta Pink (Cluster 6)
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
# 4. TOP NAVIGATION BAR & BRANDED HEADER
# ============================================================

# Top Branded Header Box
st.markdown(
    """
    <div class="brand-header-box">
        <div style="display:flex; align-items:center; gap:12px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="36" />
            <div>
                <h2 class="brand-title">Spotify Music Feature Reduction</h2>
                <p class="brand-subtitle">Unsupervised Dimensionality Reduction (PCA) & K-Means Clustering Pipeline</p>
            </div>
        </div>
        <div style="text-align:right;">
            <span style="background:#DCFCE7; color:#15803D; font-weight:700; font-size:12px; padding:4px 12px; border-radius:20px; border:1px solid #BBF7D0;">
                89,741 Tracks &bull; 9 Features &bull; 7 PCs &bull; K=7
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

navigation_options = [
    "📊 Dashboard",
    "📉 PCA Analysis",
    "📐 K-Means Evaluation",
    "🎯 Cluster Profiles",
    "🗺️ 2D/3D Cluster Map",
    "🔍 Track Explorer",
    "🧪 Run Your Own PCA",
]

selected_page = st.radio(
    "Navigation Menu",
    options=navigation_options,
    horizontal=True,
    label_visibility="collapsed",
    key="top_navbar_selection",
)

# Collapsible Sidebar Summary Drawer
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg",
    width=50,
)
st.sidebar.title("Project Summary")
st.sidebar.markdown(
    """
    **Dataset Architecture:**
    - **Raw Tracks:** 114,000 Tracks
    - **Cleaned (Deduplicated):** 89,741 Tracks
    - **Audio Features:** 9 Features
    - **PCA Reduction:** 7 PCs (94.81% Variance)
    - **Clustering Model:** K-Means ($K=7$)
    """
)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Navigate using the top navbar pills on the main screen.")


# ============================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================

if selected_page == "📊 Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown(
        "Unsupervised dimensionality reduction using **Principal Component Analysis (PCA)** "
        "and music segmentation with **K-Means Clustering** across 89,741 Spotify tracks."
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
            fig_bar.update_layout(showlegend=False, template="plotly_white", height=380)
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
            fig_pie.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================
# PAGE 2: PCA REDUCTION ANALYSIS
# ============================================================

elif selected_page == "📉 PCA Analysis":
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
                template="plotly_white",
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

elif selected_page == "📐 K-Means Evaluation":
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
            fig_elb.update_layout(template="plotly_white", height=380)
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
            fig_sil.update_layout(template="plotly_white", height=380)
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

elif selected_page == "🎯 Cluster Profiles":
    st.title("🎯 Cluster Profiles & Musical Interpretation")
    st.markdown(
        "By examining the average values of the **original 9 audio features** and the **dominant genres** "
        "associated with each cluster, we can interpret what each musical cluster represents in practice."
    )

    if df_audio_means is not None and df_genres is not None:
        tab_single, tab_multi = st.tabs(["🎯 Single Cluster Deep Dive", "🧬 Multi-Cluster Radar Comparison"])

        with tab_single:
            # Cluster Selection
            cluster_list = sorted(df_audio_means["cluster"].unique())
            selected_c = st.selectbox(
                "Select Cluster to Profile:",
                cluster_list,
                format_func=lambda x: f"Cluster {x} ({df_audio_means.loc[df_audio_means['cluster']==x, 'percentage'].values[0]:.1f}% of tracks)",
                key="cluster_profile_select",
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
                    template="plotly_white",
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
                    template="plotly_white",
                    height=380,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_genre, use_container_width=True)

        with tab_multi:
            st.subheader("🧬 Multi-Cluster Audio DNA Comparison")
            st.caption("Compare the normalized audio fingerprints of multiple clusters side-by-side. Click cluster names in the legend to isolate specific profiles.")

            radar_feats = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "valence"]
            fig_multi_radar = go.Figure()

            for _, r in df_audio_means.iterrows():
                c_id = int(r["cluster"])
                c_vals = [r[f] for f in radar_feats]
                fig_multi_radar.add_trace(
                    go.Scatterpolar(
                        r=c_vals + [c_vals[0]],
                        theta=radar_feats + [radar_feats[0]],
                        fill="none",
                        name=f"Cluster {c_id}",
                        line=dict(color=CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)], width=2.5),
                    )
                )

            fig_multi_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                template="plotly_white",
                title="All 7 Clusters Audio Fingerprint Radar",
                height=480,
            )
            st.plotly_chart(fig_multi_radar, use_container_width=True)

            st.markdown("---")
            st.subheader("Comprehensive Audio Means Comparison Table")
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

elif selected_page == "🗺️ 2D/3D Cluster Map":
    st.title("🗺️ PCA Cluster Space Navigator (2D & 3D)")
    st.markdown(
        "Explore how tracks cluster in reduced principal component space. "
        "**PC1 (32.12%)** separates high-energy produced tracks from acoustic music, "
        "**PC2 (15.87%)** captures danceability and valence, and "
        "**PC3 (7.78%)** captures speechiness, vocal presence, and acoustic nuances."
    )

    df_tracks = load_clustered_tracks(sample_for_interactive=True)

    if df_tracks is not None:
        ctrl_col1, ctrl_col2 = st.columns([1.8, 1.2])
        with ctrl_col1:
            proj_mode = st.radio(
                "Select Dimensionality View:",
                ["🗺️ 2D Cluster Map (PC1 vs PC2)", "🌐 3D Interactive Space (PC1 vs PC2 vs PC3)"],
                horizontal=True,
                key="pca_proj_mode",
            )

        c1, c2 = st.columns([3.2, 1])
        with c2:
            st.markdown("#### Filter Clusters")
            all_clusters = sorted(df_tracks["cluster"].unique())
            active_clusters = st.multiselect("Display Clusters:", all_clusters, default=all_clusters, key="pca_active_clusters")
            max_points = min(5000, len(df_tracks))
            sample_size = st.slider("Display Points:", 1000, max_points, min(3000, max_points), step=500, key="pca_display_slider")

        matched_tracks = df_tracks[df_tracks["cluster"].isin(active_clusters)]

        if len(matched_tracks) == 0:
            st.warning("Please select at least one cluster to display.")
        else:
            filtered_df = matched_tracks.sample(
                n=min(sample_size, len(matched_tracks)),
                random_state=42,
            )

            with c1:
                hover_cols = [c for c in ["track_name", "artists", "track_genre", "popularity"] if c in filtered_df.columns]

                if proj_mode == "🗺️ 2D Cluster Map (PC1 vs PC2)":
                    fig_scatter = px.scatter(
                        filtered_df,
                        x="PC1",
                        y="PC2",
                        color=filtered_df["cluster"].astype(str),
                        color_discrete_sequence=CLUSTER_COLORS,
                        hover_data=hover_cols,
                        labels={"color": "Cluster"},
                        title=f"2D PCA Projection ({len(filtered_df):,} Tracks — 47.99% Cumulative Variance)",
                        opacity=0.65,
                    )
                    fig_scatter.update_layout(
                        template="plotly_white",
                        height=580,
                        xaxis_title="PC1: Loudness & Energy vs. Acousticness (32.12% Var)",
                        yaxis_title="PC2: Danceability & Valence (15.87% Var)",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    if "PC3" in filtered_df.columns:
                        fig_3d = px.scatter_3d(
                            filtered_df,
                            x="PC1",
                            y="PC2",
                            z="PC3",
                            color=filtered_df["cluster"].astype(str),
                            color_discrete_sequence=CLUSTER_COLORS,
                            hover_data=hover_cols,
                            labels={"color": "Cluster"},
                            title=f"3D PCA Interactive Space ({len(filtered_df):,} Tracks — 55.77% Cumulative Variance)",
                            opacity=0.75,
                        )
                        fig_3d.update_traces(marker=dict(size=3.5))
                        fig_3d.update_layout(
                            template="plotly_white",
                            height=640,
                            scene=dict(
                                xaxis_title="PC1 (32.12%)",
                                yaxis_title="PC2 (15.87%)",
                                zaxis_title="PC3 (7.78%)",
                            ),
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info("PC3 not present in current view. Showing 2D projection.")

    elif IMG_PCA_CLUSTERS.exists():
        st.image(str(IMG_PCA_CLUSTERS), caption="K-Means Clusters in PCA Space (PC1 vs. PC2)", use_column_width=True)


# ============================================================
# PAGE 6: INTERACTIVE TRACK EXPLORER
# ============================================================

elif selected_page == "🔍 Track Explorer":
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
                results["track_name"].fillna("").str.lower().str.contains(query, regex=False)
                | results["artists"].fillna("").str.lower().str.contains(query, regex=False)
            ]
        if genre_filter != "All":
            results = results[results["track_genre"] == genre_filter]
        if cluster_filter != "All":
            c_num = int(cluster_filter.split(" ")[1])
            results = results[results["cluster"] == c_num]

        st.markdown(f"**Found {len(results):,} matching tracks**")

        if len(results) > 0:
            # 1. Interactive Track Selection for Live Inspection & Audio Preview
            track_options = results.head(30).reset_index(drop=True)
            track_labels = [
                f"{r['track_name']} — {r['artists']} [Cluster {r['cluster']}, {r.get('track_genre', 'N/A')}]"
                for _, r in track_options.iterrows()
            ]

            st.markdown("#### 🎧 Track Inspector & Live Preview")
            selected_idx = st.selectbox(
                "Select a Track to Inspect & Listen:",
                range(len(track_options)),
                format_func=lambda i: track_labels[i],
                key="track_inspector_selectbox",
            )
            selected_track = track_options.iloc[selected_idx]

            # Track Details & Player Card
            st.markdown('<div class="player-card">', unsafe_allow_html=True)
            col_info, col_player = st.columns([1.3, 1.2])

            with col_info:
                st.markdown(f"### 🎵 {selected_track['track_name']}")
                st.markdown(
                    f"**Artist:** {selected_track['artists']}  \n"
                    f"**Album:** {selected_track.get('album_name', 'N/A')}  \n"
                    f"**Genre:** `{selected_track.get('track_genre', 'N/A')}` &nbsp;|&nbsp; "
                    f"**Popularity:** `{selected_track.get('popularity', 'N/A')}/100`"
                )
                c_num = int(selected_track["cluster"])
                c_color = CLUSTER_COLORS[c_num % len(CLUSTER_COLORS)]
                st.markdown(
                    f'<div><span class="cluster-tag" style="background-color:{c_color}; color:#000000;">Cluster {c_num}</span> '
                    f'<span style="color:#B3B3B3; font-size:13px;">PC1: {selected_track.get("PC1", 0.0):.2f} &bull; PC2: {selected_track.get("PC2", 0.0):.2f}</span></div>',
                    unsafe_allow_html=True,
                )

            with col_player:
                track_id_val = str(selected_track.get("track_id", "")).strip()
                if track_id_val and len(track_id_val) == 22:
                    st.markdown("**Live Spotify Preview:**")
                    components.html(
                        f"""<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{track_id_val}?utm_source=generator&theme=0" width="100%" height="80" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>""",
                        height=90,
                    )
                else:
                    st.info("💡 Spotify live audio preview is available for tracks with authentic Spotify track IDs.")

            st.markdown('</div>', unsafe_allow_html=True)

            # 2. PCA Recommendation Engine ("Find 5 Acoustically Similar Tracks")
            pc_cols = [f"PC{i+1}" for i in range(7) if f"PC{i+1}" in df_tracks.columns]
            if len(pc_cols) >= 2:
                with st.expander("🔍 Acoustically Similar Tracks (PCA Nearest Neighbors)", expanded=True):
                    st.caption("Identifies the closest matching songs based on Euclidean distance in 7-dimensional PCA coordinate space.")

                    target_pc_vals = selected_track[pc_cols].values.astype(float)
                    all_pc_vals = df_tracks[pc_cols].values.astype(float)
                    distances = np.linalg.norm(all_pc_vals - target_pc_vals, axis=1)

                    if "track_id" in df_tracks.columns and track_id_val:
                        other_mask = df_tracks["track_id"] != track_id_val
                    else:
                        other_mask = np.ones(len(df_tracks), dtype=bool)

                    top_5_idx = np.argsort(distances[other_mask])[:5]
                    similar_tracks = df_tracks[other_mask].iloc[top_5_idx].copy()
                    similar_tracks["pca_distance"] = np.round(distances[other_mask][top_5_idx], 3)
                    similar_tracks["match_score"] = np.round(100 / (1 + similar_tracks["pca_distance"]), 1)

                    for _, sim_row in similar_tracks.iterrows():
                        sim_id = str(sim_row.get("track_id", "")).strip()
                        listen_link = f" &bull; [🎧 Listen on Spotify](https://open.spotify.com/track/{sim_id})" if (sim_id and len(sim_id) == 22) else ""
                        c_color_sim = CLUSTER_COLORS[int(sim_row['cluster']) % len(CLUSTER_COLORS)]

                        st.markdown(
                            f'<div class="recommendation-item">'
                            f'<strong>{sim_row["track_name"]}</strong> by <em>{sim_row["artists"]}</em> '
                            f'<span class="cluster-tag" style="background-color:{c_color_sim}; color:#000; margin-left:8px;">Cluster {sim_row["cluster"]}</span> '
                            f'<span style="color:#1DB954; font-weight:700; margin-left:8px;">{sim_row["match_score"]}% Match</span> '
                            f'<span style="color:#888; font-size:12px; margin-left:6px;">(PCA Dist: {sim_row["pca_distance"]})</span>'
                            f'{listen_link}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # 3. Full Search Results Table
            st.markdown("#### 📋 All Matching Search Results (Showing Top 50)")
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
            st.info("No tracks matched your search criteria. Try a different query or reset filters.")
    else:
        st.warning("Track dataset not found. Please run src/clustering.py to generate clustered data.")


# ============================================================
# PAGE 7: RUN YOUR OWN PCA + K-MEANS
# ============================================================

elif selected_page == "🧪 Run Your Own PCA":
    st.title("🧪 Run Your Own PCA + K-Means Pipeline")
    st.markdown(
        "Interactively configure and execute **Principal Component Analysis (PCA)** and **K-Means Clustering** "
        "on either standard **Spotify Audio Features** or **Any Generic Numerical Dataset** (customer analytics, finance, healthcare, IoT sensors, etc.)."
    )

    REQUIRED_AUDIO_FEATURES = [
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

    # Initialize Session State Keys for Pipeline Persistence
    if "custom_df" not in st.session_state:
        st.session_state["custom_df"] = None
    if "custom_file_label" not in st.session_state:
        st.session_state["custom_file_label"] = ""
    if "custom_results" not in st.session_state:
        st.session_state["custom_results"] = None
    if "custom_file_id" not in st.session_state:
        st.session_state["custom_file_id"] = None

    # 1. File Uploader
    st.subheader("1. Dataset Selection & Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV Dataset (Spotify Audio or Any Numerical CSV)",
        type=["csv"],
        help="Upload any CSV file. The pipeline will automatically detect numerical features for dimensionality reduction and clustering.",
        key="custom_csv_uploader",
    )

    # Handle file upload
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state["custom_file_id"] != file_id:
            try:
                df_loaded = pd.read_csv(uploaded_file)
                # Normalize column names: strip whitespace and convert to lowercase
                df_loaded.columns = [str(c).strip().lower() for c in df_loaded.columns]
                # Drop uninformative unnamed index columns if present
                df_loaded = df_loaded.loc[:, ~df_loaded.columns.str.startswith("unnamed")]

                # Performance & Server Protection Guardrail for Cloud Deployment
                if len(df_loaded) > 10000:
                    st.info(
                        f"⚡ **Server Performance Guardrail**: The uploaded dataset contains **{len(df_loaded):,}** rows. "
                        f"To ensure sub-second interactive execution and prevent server memory spikes on Streamlit Cloud, "
                        f"the pipeline will analyze a representative sample of **10,000** rows."
                    )
                    df_loaded = df_loaded.sample(n=10000, random_state=42).reset_index(drop=True)

                st.session_state["custom_df"] = df_loaded
                st.session_state["custom_file_label"] = uploaded_file.name
                st.session_state["custom_file_id"] = file_id
                st.session_state["custom_results"] = None
            except Exception as e:
                st.error(f"Error reading uploaded CSV file: {e}")

    # If no file uploaded and no dataset loaded in session state, offer sample datasets
    if st.session_state["custom_df"] is None:
        st.info("💡 **Tip**: Don't have a CSV handy? Test this interactive pipeline immediately with built-in samples:")
        sample_c1, sample_c2 = st.columns(2)
        with sample_c1:
            if st.button("🎵 Load Spotify Sample (5,000 Tracks)", key="btn_load_sample_spotify"):
                if CLUSTERED_SAMPLE_FILE.exists():
                    df_loaded = pd.read_csv(CLUSTERED_SAMPLE_FILE)
                    df_loaded.columns = [str(c).strip().lower() for c in df_loaded.columns]
                    st.session_state["custom_df"] = df_loaded
                    st.session_state["custom_file_label"] = "spotify_sample_dataset.csv (5,000 Tracks)"
                    st.session_state["custom_file_id"] = "sample_spotify_5000"
                    st.session_state["custom_results"] = None
                    st.rerun()
                else:
                    st.warning("Spotify sample dataset file not found on disk.")
        with sample_c2:
            if st.button("📊 Load Generic Customer Analytics Sample (500 Records)", key="btn_load_sample_generic"):
                if GENERIC_SAMPLE_FILE.exists():
                    df_loaded = pd.read_csv(GENERIC_SAMPLE_FILE)
                    df_loaded.columns = [str(c).strip().lower() for c in df_loaded.columns]
                    st.session_state["custom_df"] = df_loaded
                    st.session_state["custom_file_label"] = "customer_analytics_sample.csv (500 Records)"
                    st.session_state["custom_file_id"] = "sample_generic_500"
                    st.session_state["custom_results"] = None
                    st.rerun()
                else:
                    st.warning("Generic customer sample dataset file not found on disk.")

    # When a dataset is loaded:
    if st.session_state["custom_df"] is not None:
        df_raw = st.session_state["custom_df"]
        file_label = st.session_state["custom_file_label"]

        col_st1, col_st2 = st.columns([3, 1])
        with col_st1:
            st.success(f"Active Dataset: **{file_label}** — {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        with col_st2:
            if st.button("🔄 Clear / Reset Dataset", key="btn_clear_dataset"):
                st.session_state["custom_df"] = None
                st.session_state["custom_file_label"] = ""
                st.session_state["custom_results"] = None
                st.session_state["custom_file_id"] = None
                st.rerun()

        with st.expander("🔍 Dataset Preview (First 5 Rows)", expanded=False):
            st.dataframe(df_raw.head(5), use_container_width=True)

        # Check Spotify audio features compatibility
        has_all_spotify = all(f in df_raw.columns for f in REQUIRED_AUDIO_FEATURES)

        # Detect all numerical columns
        detected_numeric_cols = []
        for col in df_raw.columns:
            if pd.api.types.is_numeric_dtype(df_raw[col]):
                detected_numeric_cols.append(col)
            else:
                converted = pd.to_numeric(df_raw[col], errors="coerce")
                if converted.notna().sum() >= len(df_raw) * 0.7:
                    detected_numeric_cols.append(col)

        st.markdown("---")
        st.subheader("2. Pipeline Configuration")

        col_m1, col_m2 = st.columns([1.5, 2.5])
        with col_m1:
            mode_options = ["🎵 Spotify Audio Features", "📊 Generic Numerical Dataset"]
            default_mode_idx = 0 if has_all_spotify else 1
            dataset_mode = st.radio(
                "Dataset Mode:",
                options=mode_options,
                index=default_mode_idx,
                horizontal=True,
                help="Choose Spotify mode for standard 9 audio features or Generic mode for any numerical dataset (finance, customer demographics, healthcare, IoT sensors).",
                key="custom_dataset_mode",
            )

        features_valid = False
        feature_options = []
        default_features = []
        entity_label = "Records"

        if dataset_mode == "🎵 Spotify Audio Features":
            entity_label = "Tracks"
            missing_required = [f for f in REQUIRED_AUDIO_FEATURES if f not in df_raw.columns]
            if missing_required:
                st.error(
                    f"⚠️ The active dataset is missing standard Spotify audio features: "
                    f"**{', '.join(missing_required)}**. "
                    f"Please switch to **📊 Generic Numerical Dataset** mode above to select custom columns, or upload a dataset containing the 9 Spotify features."
                )
            else:
                features_valid = True
                feature_options = REQUIRED_AUDIO_FEATURES
                default_features = REQUIRED_AUDIO_FEATURES
        else:
            entity_label = "Records"
            # Filter out existing PC or cluster columns if someone re-uploads a previously clustered file
            feature_options = [
                c for c in detected_numeric_cols
                if not (c.startswith("pc") and len(c) <= 4 and c[2:].isdigit()) and c != "cluster"
            ]
            if len(feature_options) < 2:
                # If too few, fall back to detected_numeric_cols
                feature_options = detected_numeric_cols

            if len(feature_options) < 2:
                st.error("⚠️ The dataset must have at least 2 numerical columns for PCA and K-Means. Fewer than 2 numeric columns were detected.")
            else:
                features_valid = True
                default_features = feature_options[:min(10, len(feature_options))]

        if features_valid:
            col_config1, col_config2 = st.columns(2)

            with col_config1:
                feat_type_name = "Audio" if dataset_mode.startswith("🎵") else "Numerical"
                st.markdown(f"#### {feat_type_name} Feature Selection")
                selected_features = st.multiselect(
                    f"Select {feat_type_name} Features for PCA ({len(feature_options)} available):",
                    options=feature_options,
                    default=default_features,
                    help="Choose at least 2 numerical features to include in the PCA dimensionality reduction.",
                    key="custom_feat_multiselect",
                )

                if len(selected_features) < 2:
                    st.warning("⚠️ Please select at least 2 numerical features to perform PCA dimensionality reduction.")

            with col_config2:
                st.markdown("#### PCA Component Settings")
                st.caption(
                    "PCA transforms correlated numerical features into orthogonal principal components while retaining maximum variance."
                )
                pca_mode = st.radio(
                    "PCA Selection Mode:",
                    ["Variance-based selection", "Manual number of components"],
                    horizontal=True,
                    key="custom_pca_mode",
                )

                if pca_mode == "Variance-based selection":
                    var_threshold = st.select_slider(
                        "Target Cumulative Explained Variance:",
                        options=[80, 85, 90, 95, 99],
                        value=90,
                        format_func=lambda x: f"{x}%",
                        key="custom_var_threshold",
                    )
                else:
                    max_pcs = max(1, len(selected_features))
                    manual_pcs = st.slider(
                        "Number of Principal Components:",
                        min_value=1,
                        max_value=max_pcs,
                        value=min(7, max_pcs),
                        key="custom_manual_pcs",
                    )

            st.markdown("#### 3. K-Means Clustering Settings")
            st.caption(f"K-Means partitions {entity_label.lower()} into K clusters based on Euclidean distance across all selected principal components.")
            k_clusters = st.slider(
                "Number of clusters (K):",
                min_value=2,
                max_value=10,
                value=min(7, max(2, len(df_raw))),
                help=f"Select the number of clusters to segment the {entity_label.lower()} into.",
                key="custom_k_slider",
            )

            # Pre-filter numeric validation & report counts
            clean_sub = df_raw[selected_features].apply(pd.to_numeric, errors="coerce")
            valid_mask = clean_sub.notna().all(axis=1)
            valid_count = int(valid_mask.sum())
            removed_count = len(df_raw) - valid_count

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Original {entity_label}", f"{len(df_raw):,}")
            m2.metric("Rows with Missing Values", f"{removed_count:,}", delta=f"-{removed_count}" if removed_count > 0 else "0", delta_color="inverse")
            m3.metric(f"Valid {entity_label} Used", f"{valid_count:,}")
            m4.metric("Features Selected", f"{len(selected_features)}")

            if valid_count < k_clusters:
                st.error(f"Not enough valid numeric rows ({valid_count}) to fit {k_clusters} clusters. Please provide a dataset with at least {k_clusters} valid rows.")
            elif len(selected_features) >= 2:
                if st.button("🚀 Run PCA + K-Means Analysis", type="primary", key="btn_run_pipeline"):
                    with st.spinner(f"Standardizing features, fitting PCA, and clustering {entity_label.lower()}..."):
                        try:
                            # 1. Prepare data without mutating raw uploaded dataframe
                            df_valid = df_raw[valid_mask].copy().reset_index(drop=True)
                            X_raw = clean_sub[valid_mask].values

                            # 2. Standardize
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_raw)

                            # 3. Fit full PCA
                            n_max_components = min(X_scaled.shape[0], X_scaled.shape[1])
                            full_pca = PCA(n_components=n_max_components)
                            X_pca_full = full_pca.fit_transform(X_scaled)
                            cum_var = np.cumsum(full_pca.explained_variance_ratio_) * 100
                            ind_var = full_pca.explained_variance_ratio_ * 100

                            # Determine number of components
                            if pca_mode == "Variance-based selection":
                                n_pcs = int(np.argmax(cum_var >= var_threshold) + 1)
                            else:
                                n_pcs = min(manual_pcs, n_max_components)

                            retained_var = float(cum_var[n_pcs - 1])
                            X_pca_selected = X_pca_full[:, :n_pcs]
                            pc_col_names = [f"PC{i+1}" for i in range(n_pcs)]

                            # 4. Fit K-Means on ALL selected components
                            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                            cluster_labels = kmeans.fit_predict(X_pca_selected)

                            # 5. Build Result Dataframe
                            df_results = df_valid.copy()
                            for idx, col_name in enumerate(pc_col_names):
                                df_results[col_name] = np.round(X_pca_selected[:, idx], 4)
                            df_results["cluster"] = cluster_labels

                            # 6. Cluster Summary
                            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                            cluster_summary = pd.DataFrame({
                                "cluster": cluster_counts.index,
                                "count": cluster_counts.values,
                                "percentage": np.round((cluster_counts.values / len(cluster_labels)) * 100, 2),
                            })
                            for i in range(n_pcs):
                                cluster_summary[f"mean_PC{i+1}"] = np.round(
                                    [X_pca_selected[cluster_labels == c, i].mean() for c in cluster_summary["cluster"]], 4
                                )

                            # 7. Cluster Profiles (Original Feature Scales)
                            df_profiles = df_results.groupby("cluster")[selected_features].mean().round(4).reset_index()

                            # Persist results in session state
                            st.session_state["custom_results"] = {
                                "df_results": df_results,
                                "cluster_summary": cluster_summary,
                                "df_profiles": df_profiles,
                                "selected_features": selected_features,
                                "n_pcs": n_pcs,
                                "retained_var": retained_var,
                                "ind_var": ind_var,
                                "cum_var": cum_var,
                                "n_max_components": n_max_components,
                                "k_clusters": k_clusters,
                                "orig_features_count": len(df_raw.columns),
                                "entity_label": entity_label,
                                "dataset_mode": dataset_mode,
                            }
                            st.success("Analysis complete! See results below.")
                        except Exception as err:
                            st.error(f"An error occurred during PCA or K-Means execution: {err}")

            # ============================================================
            # RENDER RESULTS (PERSISTED VIA SESSION STATE)
            # ============================================================
            if st.session_state.get("custom_results") is not None:
                res = st.session_state["custom_results"]
                df_res = res["df_results"]
                c_summ = res["cluster_summary"]
                d_prof = res["df_profiles"]
                n_pcs = res["n_pcs"]
                retained_var = res["retained_var"]
                ind_var = res["ind_var"]
                cum_var = res["cum_var"]
                n_max_components = res["n_max_components"]
                res_entity = res.get("entity_label", "Records")

                st.markdown("---")
                st.header("📊 Analysis Results")

                # A. PCA Summary Cards
                res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                res_c1.metric("Original Columns", res["orig_features_count"])
                res_c2.metric("Selected Features", len(res["selected_features"]))
                res_c3.metric("Selected PCs", f"{n_pcs} PCs")
                res_c4.metric("Variance Retained", f"{retained_var:.2f}%")

                # B. Explained Variance Visualization
                st.subheader("1. Explained Variance by Principal Component")
                fig_exp = go.Figure()
                comp_labels = [f"PC{i+1}" for i in range(n_max_components)]
                fig_exp.add_trace(
                    go.Bar(
                        x=comp_labels,
                        y=ind_var,
                        name="Individual Variance (%)",
                        marker_color=["#1DB954" if i < n_pcs else "#555555" for i in range(n_max_components)],
                    )
                )
                fig_exp.add_trace(
                    go.Scatter(
                        x=comp_labels,
                        y=cum_var,
                        name="Cumulative Variance (%)",
                        line=dict(color="#4A90E2", width=3),
                        mode="lines+markers",
                    )
                )
                fig_exp.add_hline(
                    y=retained_var,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Retained: {retained_var:.1f}% ({n_pcs} PCs)",
                )
                fig_exp.update_layout(
                    title=f"PCA Explained Variance ({n_pcs} Selected Components)",
                    xaxis_title="Principal Component",
                    yaxis_title="Variance (%)",
                    template="plotly_white",
                    height=380,
                )
                st.plotly_chart(fig_exp, use_container_width=True)

                # C. Cluster Summary (Sizes & Share)
                st.subheader("2. K-Means Cluster Distribution")
                col_c1, col_c2 = st.columns([1.2, 1])
                with col_c1:
                    fig_sz = px.bar(
                        c_summ,
                        x=[f"Cluster {c}" for c in c_summ["cluster"]],
                        y="count",
                        text="count",
                        color=[f"Cluster {c}" for c in c_summ["cluster"]],
                        color_discrete_sequence=CLUSTER_COLORS,
                        labels={"x": "Cluster", "y": f"Number of {res_entity}"},
                        title=f"{res_entity} Count per Cluster",
                    )
                    fig_sz.update_traces(textposition="outside")
                    fig_sz.update_layout(showlegend=False, template="plotly_white", height=350)
                    st.plotly_chart(fig_sz, use_container_width=True)

                with col_c2:
                    st.markdown(f"##### {res_entity} Distribution Table")
                    st.dataframe(
                        c_summ[["cluster", "count", "percentage"]].rename(
                            columns={"cluster": "Cluster", "count": res_entity, "percentage": "Share (%)"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                # D. Cluster Profiles (Original Scales)
                st.subheader("3. Cluster Feature Profiles (Original Scales)")
                st.caption("Mean values of selected features for each cluster (unscaled for domain interpretation).")
                st.dataframe(d_prof, use_container_width=True, hide_index=True)

                # E. 2D PCA Visualization
                st.subheader("4. 2D PCA Cluster Map (PC1 vs. PC2)")
                if n_pcs >= 2:
                    plot_df = df_res
                    sample_note = ""
                    if len(df_res) > 5000:
                        plot_df = df_res.sample(n=5000, random_state=42)
                        sample_note = f" (Displaying representative sample of 5,000 / {len(df_res):,} {res_entity.lower()} for responsiveness)"

                    # Intelligent hover columns based on available metadata
                    potential_hover = [
                        c for c in ["track_name", "artists", "track_genre", "album_name", "customer_id", "segment_category", "name", "id", "category"]
                        if c in plot_df.columns
                    ]
                    if not potential_hover:
                        potential_hover = [
                            c for c in plot_df.columns
                            if c not in res["selected_features"] and not c.startswith("PC") and c != "cluster"
                        ][:4]

                    fig_scatter = px.scatter(
                        plot_df,
                        x="PC1",
                        y="PC2",
                        color=plot_df["cluster"].astype(str),
                        color_discrete_sequence=CLUSTER_COLORS,
                        hover_data=potential_hover if potential_hover else None,
                        labels={"color": "Cluster"},
                        title=f"{res_entity} in PC1-PC2 Space{sample_note}",
                        opacity=0.6,
                    )
                    fig_scatter.update_layout(
                        template="plotly_white",
                        height=520,
                        xaxis_title="PC1",
                        yaxis_title="PC2",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Only 1 principal component was selected. 2D scatter plot requires at least 2 components.")

                # F. Download Results
                st.subheader("5. Export & Download Results")
                d_col1, d_col2, d_col3 = st.columns(3)

                csv_clustered = df_res.to_csv(index=False).encode("utf-8")
                csv_summary = c_summ.to_csv(index=False).encode("utf-8")
                csv_profiles = d_prof.to_csv(index=False).encode("utf-8")

                with d_col1:
                    st.download_button(
                        label=f"📥 Download Clustered {res_entity} (CSV)",
                        data=csv_clustered,
                        file_name="custom_pca_kmeans_clustered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_clustered_csv",
                    )
                with d_col2:
                    st.download_button(
                        label="📥 Download Cluster Summary (CSV)",
                        data=csv_summary,
                        file_name="custom_cluster_summary.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_summary_csv",
                    )
                with d_col3:
                    st.download_button(
                        label="📥 Download Feature Profiles (CSV)",
                        data=csv_profiles,
                        file_name="custom_cluster_feature_profiles.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_profiles_csv",
                    )


