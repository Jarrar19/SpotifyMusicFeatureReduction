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

## Proposed Workflow

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
       [Phase 3 - Pending]
       K-Means Clustering & PC1 vs PC2 Visualization
                    ↓
       Analyze Music Groups & Patterns
```

---

## Project Structure

```text
SpotifyMusicFeatureReduction/
│
├── README.md              # Project overview, methodology, and results
├── requirements.txt       # Python dependencies
│
├── data/                  # Dataset folder
│   └── spotify-tracks-dataset-detailed.csv  (Raw dataset: 114,000 rows)
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── data_exploration.py  # Phase 2: Exploratory data analysis
│   ├── preprocessing.py     # Phase 2: Track_id deduplication & standardization
│   └── pca_analysis.py      # Phase 2: Deep PCA analysis & loadings
│
└── results/               # Generated datasets and metrics (Local outputs)
    ├── cleaned_selected_audio_features.csv
    ├── cleaned_standardized_audio_features.csv
    ├── cleaned_track_metadata.csv
    ├── cleaned_pca_explained_variance.csv
    ├── cleaned_pca_loadings.csv
    ├── cleaned_pca_reconstruction_error.csv
    └── cleaned_pca_reduced_data.csv
```

> **Note on Data Files**: Raw dataset files in `data/` and generated CSV files in `results/` are intended for local execution and are kept out of version control to avoid repository bloat.

---

## Current Project Status

- **Phase 1 — Planning and Assessment**: ✅ COMPLETED
- **Phase 2 — Data Preparation and Deep PCA Analysis**: ✅ COMPLETED
- **Phase 3 — Clustering and Visualization**: ⏳ NOT YET STARTED (Pending)

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

**Key Finding**: Removing cross-genre duplicate tracks reduced the dataset size by 24,259 tracks but had virtually no impact on the underlying PCA variance structure, showing that the PCA variance structure remained almost unchanged after removing duplicate track IDs (94.80% vs. 94.81% variance retained at 7 PCs).

---

## Next Steps (Phase 3 Plan)

Once Phase 3 begins, the remaining project steps will be:

1. **K-Means Clustering**: Apply K-Means clustering on the reduced 7-dimensional PCA dataset (`cleaned_pca_reduced_data.csv`).
2. **Cluster Optimization**: Determine the optimal number of clusters using standard evaluation methods (such as the Elbow method).
3. **2D Visualization**: Plot tracks using PC1 vs. PC2 colored by cluster labels to visualize song groupings.
4. **Cluster Interpretation**: Use the aligned metadata (`cleaned_track_metadata.csv`) to profile and characterize the music styles within each cluster.

---

## How to Run

### 1. Prerequisites
Install required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Execute Pipeline
Run the preprocessing and PCA scripts:
```bash
# Step 1: Data exploration
python src/data_exploration.py

# Step 2: Data deduplication, standardization & metadata alignment
python src/preprocessing.py

# Step 3: Full PCA fitting, loadings analysis & dimensionality reduction
python src/pca_analysis.py
```
