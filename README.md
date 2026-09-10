# Spotify Music Feature Reduction using PCA

## Project Overview

Music datasets contain a lot of information about each song. A Spotify track, for example, can be described using features such as danceability, energy, loudness, tempo, valence, acousticness, and instrumentalness.

When many features are used together, it can become difficult to understand the data and visualize patterns between songs.

In this project, we use **Principal Component Analysis (PCA)** to reduce the number of dimensions in Spotify music data while retaining as much important information as possible.

After reducing the dimensions, we will use the reduced data for **visualization and clustering** to explore groups of songs with similar characteristics.

---

## Problem Statement

Spotify music data contains several numerical audio features for every track. Analyzing all these features at the same time can make the dataset complex and difficult to visualize.

The main question we want to explore is:

> **How can PCA reduce the number of Spotify audio features while retaining important information and making patterns between songs easier to understand?**

---

## Objectives

The main objectives of this project are:

- Understand the concept of dimensionality reduction.
- Study and apply Principal Component Analysis (PCA) to audio features.
- Reduce multiple Spotify audio features into fewer dimensions.
- Analyze how much information is retained using explained variance.
- Evaluate reconstruction error across different component counts.
- Apply clustering to explore groups of similar songs (Phase 3).
- Visualize the reduced data using principal components.
- Make a high-dimensional music dataset easier to analyze and understand.

---

## Why PCA?

Principal Component Analysis (PCA) is a dimensionality reduction technique.

Instead of working with all the original features separately, PCA transforms them into new variables called **Principal Components**.

- **PC1 (Principal Component 1)** captures the maximum possible variance in the data.
- **PC2 (Principal Component 2)** captures the next highest amount of variance.
- Additional components capture the remaining variation in decreasing order.

By selecting the important components, we can represent the original dataset using fewer dimensions. We also use **explained variance** and **reconstruction error** to measure how much information is retained.

---

## Why Spotify Music Data?

Spotify tracks are a good example for this project because each song can be represented using several numerical audio features.

The 9 audio features used in this project include:

1. **Danceability**: How suitable a track is for dancing.
2. **Energy**: Perceptual measure of intensity and activity.
3. **Loudness**: Overall loudness of a track in decibels (dB).
4. **Speechiness**: Presence of spoken words in a track.
5. **Acousticness**: Confidence measure of whether the track is acoustic.
6. **Instrumentalness**: Predicts whether a track contains no vocals.
7. **Liveness**: Detects the presence of an audience in the recording.
8. **Valence**: Musical positiveness conveyed by a track.
9. **Tempo**: Overall estimated tempo in beats per minute (BPM).

---

### Proposed Workflow

```text
Spotify Tracks Dataset (Raw: 114,000 tracks)
                    ↓
   Duplicate Investigation & Cleaning (89,741 unique track_ids)
                    ↓
          Select 9 Audio Features
                    ↓
  Feature Standardization (StandardScaler)
                    ↓
        Principal Component Analysis (PCA)
                    ↓
 Select Components for >= 90% Variance (7 PCs: 94.81%)
                    ↓
  Reduced Dataset (89,741 x 7 PCs) + Aligned Metadata
                    ↓
 K-Means Clustering Evaluation (Elbow & Silhouette across K=2..10)
                    ↓
     Final Model Selection (K = 7 Clusters)
                    ↓
  Cluster Interpretation & Audio Feature Profiling
                    ↓
  PC1 vs. PC2 2D Visualization & Size Distribution
                    ↓
  Interactive Streamlit Application & Cloud Deployment
```

---

## Project Structure

```text
SpotifyMusicReduction/
│
├── README.md                           # Project overview, methodology, and results
├── requirements.txt                    # Python dependencies (including Streamlit)
├── app.py                              # Interactive Streamlit Web Application
│
├── .streamlit/                         # Streamlit configuration
│   └── config.toml                     # Dark theme and server configuration
│
├── data/                               # Dataset folder (local execution only)
│   └── spotify-tracks-dataset-detailed.csv  (Raw dataset: 114,000 rows)
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── data_exploration.py             # Phase 2: Exploratory data analysis
│   ├── preprocessing.py                # Phase 2: Track_id deduplication & standardization
│   ├── pca_analysis.py                 # Phase 2: Deep PCA analysis & loadings
│   ├── clustering.py                   # Phase 3: K-Means evaluation, fitting & profiling
│   └── visualization.py                # Phase 3: 2D PCA cluster map & size charts
│
└── results/                            # Generated datasets, metrics & figures
    ├── .gitkeep
    ├── cleaned_selected_audio_features.csv
    ├── cleaned_standardized_audio_features.csv
    ├── cleaned_track_metadata.csv
    ├── cleaned_pca_explained_variance.csv
    ├── cleaned_pca_loadings.csv
    ├── cleaned_pca_reconstruction_error.csv
    ├── cleaned_pca_reduced_data.csv
    ├── explained_variance.png
    ├── kmeans_elbow_scores.csv
    ├── kmeans_elbow.png
    ├── kmeans_silhouette_scores.csv
    ├── kmeans_silhouette.png
    ├── kmeans_clustered_data.csv       # (89,741 rows x 7 PCs + cluster + metadata)
    ├── kmeans_clustered_sample.csv     # (5,000-track sample for web app deployment)
    ├── cluster_summary.csv
    ├── cluster_audio_feature_means.csv
    ├── cluster_genre_distribution.csv
    ├── cluster_sizes.png
    └── kmeans_pca_clusters.png
```

> **Note on Data Files**: Raw dataset files in `data/` and large generated CSV files in `results/` are intended for local execution and are kept out of version control to avoid repository bloat. Lightweight summary metrics and sampled data are used for cloud deployment.

---

## Current Project Status

- **Phase 1 — Planning and Assessment**: ✅ COMPLETED
- **Phase 2 — Data Preparation and Deep PCA Analysis**: ✅ COMPLETED
- **Phase 3 — Clustering, Visualization & Interactive Deployment**: ✅ COMPLETED

---

## Phase 2: Preprocessing & PCA Results

### 1. Dataset Preprocessing & Cleaning
- **Raw Data**: 114,000 tracks with 20 columns.
- **Duplicate Investigation**: Analysis showed that 24,259 rows had duplicate `track_id`s caused by songs appearing in multiple genre playlists. For repeated track_id values, all 9 selected audio features were identical, while differences were primarily found in track_genre and, for some tracks, popularity.
- **Deduplication**: Cleaning by `track_id` retained **89,741 unique tracks**.
- **Missing Values**: 0 missing values found in selected PCA features.
- **Standardization**: Applied `StandardScaler` to ensure all 9 features have mean ≈ 0 and standard deviation ≈ 1.

### 2. PCA Explained Variance

| Principal Component | Explained Variance | Cumulative Variance | Interpretation / Top Contributing Features |
| :--- | :---: | :---: | :--- |
| **PC1** | **32.12%** | 32.12% | Loudness (+0.515), Energy (+0.505), Acousticness (-0.433) |
| **PC2** | **15.87%** | 47.99% | Danceability (+0.562), Valence (+0.527), Acousticness (+0.315) |
| **PC3** | **13.89%** | 61.88% | Liveness (+0.673), Speechiness (+0.641), Acousticness (+0.217) |
| **PC4** | **10.00%** | 71.88% | Tempo (+0.657), Instrumentalness (-0.495), Danceability (-0.317) |
| **PC5** | **9.69%** | 81.57% | Tempo (+0.627), Speechiness (+0.531), Instrumentalness (+0.415) |
| **PC6** | **8.17%** | 89.74% | Liveness (+0.585), Instrumentalness (+0.463), Valence (+0.441) |
| **PC7** | **5.07%** | **94.81%** | Danceability (+0.627), Valence (-0.604), Liveness (+0.296) |
| **PC8** | 3.63% | 98.44% | Acousticness (+0.681), Loudness (+0.627), Instrumentalness (+0.298) |
| **PC9** | 1.56% | 100.00% | Energy (+0.732), Loudness (-0.513), Acousticness (+0.310) |

### 3. Component Selection & Reconstruction Error
- **$\ge 90\%$ Variance Threshold**: Requires **7 components**, which retain **94.81%** of total variance (6 components reach 89.74%).
- **2D Representation (PC1 + PC2)**: Retains **47.99%** of cumulative variance.
- **Reconstruction MSE**:
  - $k = 2$: MSE = 0.5201 (47.99% variance)
  - $k = 4$: MSE = 0.2812 (71.88% variance)
  - $k = 7$: MSE = 0.0519 (94.81% variance)

### 4. Baseline vs. Cleaned PCA Comparison

| Metric | Baseline (Exact-Row Deduplication) | Cleaned (`track_id` Deduplication) |
| :--- | :---: | :---: |
| **Dataset Size** | 113,550 tracks | 89,741 unique tracks |
| **PCA Audio Features** | 9 | 9 |
| **PCs for $\ge 90\%$ Variance** | **7 PCs** | **7 PCs** |
| **Cumulative Variance (7 PCs)** | **94.80%** | **94.81%** |
| **2D Variance (PC1 + PC2)** | 47.73% | 47.99% |

---

## Phase 3: K-Means Clustering & Visualization Results

### 1. Methodology
- **Clustering Input**: The cleaned 7-dimensional PCA dataset (`results/cleaned_pca_reduced_data.csv`, 89,741 tracks $\times$ 7 PCs). PCA was used strictly for dimensionality reduction; K-Means performed unsupervised clustering.
- **Reproducibility**: Models evaluated with `random_state=42`, `n_init=10`.
- **Evaluation**: Evaluated $K \in [2, 10]$ using Within-Cluster Sum of Squares (Inertia/WCSS) on the full dataset and Silhouette Scores on a representative sample of 10,000 tracks.

### 2. Cluster Evaluation (Elbow & Silhouette)

| $K$ | Inertia (WCSS) | Marginal % Drop | Silhouette Score (10,000 sample) | Evaluation Note |
| :---: | :---: | :---: | :---: | :--- |
| **2** | 591,132.89 | — | **0.2720** | Global silhouette peak (binary macro-split: high energy vs. acoustic) |
| **3** | 520,026.42 | 12.03% | 0.1759 | Steep inertia reduction |
| **4** | 463,963.26 | 10.78% | 0.1845 | Continuous steady reduction |
| **5** | 415,332.84 | 10.48% | 0.1903 | Granular sub-clusters emerging |
| **6** | 373,839.45 | 9.99% | 0.2030 | Rising silhouette score |
| **7** | **337,715.45** | **9.66%** | **0.2108** | **Selected Multi-Cluster Solution (Elbow boundary & multi-cluster peak)** |
| **8** | 317,238.14 | 6.06% | 0.1909 | Sharp drop in marginal gain (<7%); silhouette declines |
| **9** | 298,832.81 | 5.80% | 0.1984 | Diminishing returns |
| **10** | 285,040.40 | 4.62% | 0.1844 | Over-fragmentation |

#### Rationale for Selecting $K = 7$:
> **$K=7$ was selected as the most useful multi-cluster solution.** $K=2$ achieved the highest silhouette score (0.2720) but represents a coarse two-group partition (high-energy produced tracks vs. quiet acoustic tracks). Among $K \ge 3$, $K=7$ provided the strongest silhouette score (0.2108) and was consistent with the elbow/diminishing-return region.

1. **Elbow Method Inflection**: The marginal WCSS drop stays between 9.6% and 12.0% from $K=2$ to $K=7$. At $K=8$, the rate of reduction sharply declines to **6.06%**, signaling clear diminishing returns beyond $K=7$.
2. **Silhouette Peak Among Multi-Clusters**: While $K=2$ exhibits the global mathematical silhouette peak (0.2720), it merely bisects the music dataset into broad high-energy vs. quiet tracks. Among all granular multi-cluster candidates ($K \ge 3$), **$K = 7$ achieves the highest silhouette score (0.2108)**.
3. **Domain Interpretability**: $K=7$ captures distinct, actionable music profiles (spoken-word/comedy, rock/metal, dancehall/salsa, acoustic/ambient, Brazilian rhythm, melodic ballads, and electronic/techno).

---

### 3. Cluster Summary & Audio Feature Profiles

| Cluster | Tracks | Share | Danceability | Energy | Loudness (dB) | Acousticness | Instrumentalness | Liveness | Valence | Tempo (BPM) | Dominant Genres | Data-Driven Profile Description |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **C0** | 1,047 | 1.17% | 0.573 | 0.674 | -11.24 | 0.745 | 0.007 | 0.627 | 0.443 | 101.0 | comedy (76.7%), show-tunes, kids | **Spoken Word & Comedy**: Extremely high speechiness (0.845) and acousticness; negligible instrumentalness. |
| **C1** | 20,175 | 22.48% | 0.472 | **0.818** | **-5.29** | 0.077 | 0.038 | 0.228 | 0.372 | **140.0** | heavy-metal, hardstyle, grunge, death-metal | **High-Energy Rock & Metal**: Highest loudness, highest energy, fastest tempo, lowest acousticness. |
| **C2** | 27,542 | 30.69% | **0.705** | 0.717 | -6.50 | 0.227 | 0.021 | 0.177 | **0.701** | 118.2 | salsa, dancehall, kids, forro, dance | **Upbeat & Danceable Pop/Latin**: Highest danceability, highest musical positiveness (valence). |
| **C3** | 6,600 | 7.35% | 0.344 | **0.172** | **-21.17** | **0.863** | **0.792** | 0.143 | 0.182 | 102.4 | new-age, sleep, ambient, classical, iranian | **Calm, Ambient & Classical**: Lowest energy, quietest (-21 dB), highest acousticness, high instrumentalness. |
| **C4** | 6,179 | 6.89% | 0.520 | 0.754 | -7.08 | 0.291 | 0.079 | **0.703** | 0.504 | 123.5 | pagode, sertanejo, samba, mpb, forro | **Live & Percussive Acoustic Rhythm**: High liveness (0.703), energetic rhythm, balanced acoustic/valence. |
| **C5** | 18,170 | 20.25% | 0.526 | 0.372 | -10.78 | 0.677 | 0.031 | 0.165 | 0.390 | 113.2 | tango, honky-tonk, romance, cantopop, acoustic | **Melodic Ballads & Acoustic Melodies**: Moderate tempo, high acousticness, lower energy, vocal-driven. |
| **C6** | 10,028 | 11.17% | 0.585 | 0.741 | -8.56 | 0.108 | **0.792** | 0.188 | 0.345 | 126.9 | minimal-techno, detroit-techno, study, house | **Electronic & Instrumental Groove**: High instrumentalness (0.792) paired with high energy (0.741) and electronic tempo. |

---

### 4. Cluster Visualizations
- **PC1 vs. PC2 Scatter Projection** (`results/kmeans_pca_clusters.png`):
  - PC1 cleanly separates acoustic/ambient tracks (Cluster 3 & 5, negative PC1) from loud, high-energy rock/pop/dance (Cluster 1 & 2, positive PC1).
  - PC2 differentiates high danceability and valence tracks (Cluster 2, positive PC2) from lower-valence and live/spoken tracks.
- **Cluster Size Distribution** (`results/cluster_sizes.png`):
  - Shows the distribution across all 7 clusters, confirming that the largest music groups are Upbeat Pop/Latin (30.7%), High-Energy Rock/Metal (22.5%), and Melodic Ballads (20.3%), with specialized subsets like Spoken Word (1.2%).

---

## Deployment

The project includes an interactive web application built with **Streamlit** for visual exploration, model evaluation, and track discovery.

- **Application Framework**: [Streamlit](https://streamlit.io/)
- **Deployment Platform**: Streamlit Community Cloud (via GitHub repository)
- **Status**: Tested locally and ready for cloud deployment
- **Public URL Placeholder**: `[Streamlit App (Deployment Pending Review)]`

### How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app.py
```

The application runs on `http://localhost:8501` and provides:
1. **Executive Dashboard**: KPI metrics, pipeline workflow, and cluster share charts.
2. **PCA Reduction Analysis**: Cumulative explained variance interactive chart and loadings breakdown.
3. **K-Means Evaluation**: Side-by-side interactive Elbow and Silhouette score curves with data-backed reasoning.
4. **Cluster Profiles**: Dynamic radar charts and top genre breakdowns for each cluster.
5. **2D PCA Cluster Map**: Interactive scatter plot with cluster filters and point sampling.
6. **Track Explorer**: Live search by song or artist name, genre filtering, and cluster inspection.

---

## Complete Execution Pipeline

To run the entire pipeline from data preprocessing to web app:

```bash
# Step 1: Exploratory Data Analysis
python src/data_exploration.py

# Step 2: Preprocessing, Deduplication & Standardization
python src/preprocessing.py

# Step 3: Deep PCA Analysis & Dimensionality Reduction
python src/pca_analysis.py

# Step 4: K-Means Clustering, Elbow/Silhouette Evaluation & Profiling
python src/clustering.py

# Step 5: Visualizations (PC1 vs PC2 Scatter & Cluster Sizes)
python src/visualization.py

# Step 6: Launch Web App
streamlit run app.py
```
