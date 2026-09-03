"""
Spotify Music Feature Reduction using PCA
------------------------------------------
Phase 2: Deep Cleaned PCA Analysis

This script:
1. Loads cleaned standardized Spotify audio features (89,741 tracks)
2. Fits PCA with all 9 components
3. Evaluates explained variance and determines components for >=90% variance
4. Computes and displays the PCA loading matrix (features as rows, PCs as columns)
5. Identifies the top 3 contributing features for each PC based on absolute loading magnitude
6. Computes reconstruction MSE for k = 2 through 9
7. Compares cleaned PCA against the baseline PCA (113,550 rows, 7 PCs, 94.80% variance)
8. Saves cleaned PCA variance, loadings, reconstruction error, and reduced dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = PROJECT_ROOT / "results"

CLEANED_STANDARDIZED_FILE = (
    RESULTS_FOLDER / "cleaned_standardized_audio_features.csv"
)
CLEANED_METADATA_FILE = (
    RESULTS_FOLDER / "cleaned_track_metadata.csv"
)

# Baseline metrics for comparison
BASELINE_ROWS = 113550
BASELINE_FEATURES = 9
BASELINE_PCS_90 = 7
BASELINE_VARIANCE_90 = 0.9480


# ============================================================
# 2. AUDIO FEATURES DEFINITION
# ============================================================

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


# ============================================================
# 3. LOAD CLEANED STANDARDIZED DATA
# ============================================================

def load_cleaned_data():
    """Load the cleaned standardized audio features."""

    print("\n" + "=" * 60)
    print("LOADING CLEANED STANDARDIZED DATA")
    print("=" * 60)

    if not CLEANED_STANDARDIZED_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned standardized data was not found at: {CLEANED_STANDARDIZED_FILE}\n"
            "Please run preprocessing.py first."
        )

    df = pd.read_csv(CLEANED_STANDARDIZED_FILE)

    print(f"File    : {CLEANED_STANDARDIZED_FILE.name}")
    print(f"Tracks  : {len(df):,}")
    print(f"Features: {len(df.columns)}")

    return df


# ============================================================
# 4. FIT FULL PCA (9 COMPONENTS)
# ============================================================

def fit_full_pca(df):
    """Fit PCA with all 9 components."""

    print("\n" + "=" * 60)
    print("FITTING PCA (ALL 9 COMPONENTS)")
    print("=" * 60)

    pca = PCA(n_components=9)
    transformed_data = pca.fit_transform(df)

    print("PCA fitting completed successfully.")
    print(f"Input dimensions     : {df.shape[1]}")
    print(f"Components calculated: {pca.n_components_}")

    return pca, transformed_data


# ============================================================
# 5. EXPLAINED VARIANCE ANALYSIS
# ============================================================

def analyze_explained_variance(pca):
    """
    Calculate and display explained variance table and
    determine components needed for >=90% variance.
    """

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    variance_df = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(explained_variance))],
        "Explained_Variance": explained_variance,
        "Cumulative_Variance": cumulative_variance
    })

    print("\n" + "=" * 60)
    print("EXPLAINED VARIANCE TABLE")
    print("=" * 60)
    print(
        variance_df.to_string(
            index=False,
            formatters={
                "Explained_Variance": "{:.4f}".format,
                "Cumulative_Variance": "{:.4f}".format
            }
        )
    )

    # Determine minimum components for >=90% variance
    target_variance = 0.90
    components_needed = int((cumulative_variance >= target_variance).argmax() + 1)
    retained_variance = cumulative_variance[components_needed - 1]

    print("\n" + "-" * 60)
    print("COMPONENT SELECTION CRITERION (>= 90% Variance)")
    print("-" * 60)
    print(f"Target threshold      : {target_variance * 100:.0f}%")
    print(f"Components required   : {components_needed} PCs")
    print(f"Actual variance kept  : {retained_variance * 100:.2f}%")

    return variance_df, components_needed, retained_variance


# ============================================================
# 6. PCA LOADINGS MATRIX & TOP CONTRIBUTORS
# ============================================================

def analyze_loadings(pca):
    """
    Compute PCA loading matrix (features as rows, PCs as columns)
    and identify top 3 contributing features per PC.
    """

    pc_columns = [f"PC{i + 1}" for i in range(9)]

    # Loadings matrix: rows = original features, cols = PCs
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=AUDIO_FEATURES,
        columns=pc_columns
    )

    print("\n" + "=" * 60)
    print("PCA LOADINGS MATRIX (Features x PCs)")
    print("=" * 60)
    print(loadings_df.round(3).to_string())

    print("\n" + "=" * 60)
    print("TOP 3 CONTRIBUTING FEATURES PER PRINCIPAL COMPONENT")
    print("=" * 60)

    for pc in pc_columns:
        # Sort by absolute loading magnitude
        sorted_loadings = loadings_df[pc].reindex(
            loadings_df[pc].abs().sort_values(ascending=False).index
        )
        top_3 = sorted_loadings.head(3)
        top_3_str = ", ".join(
            [f"{feat} ({val:+.3f})" for feat, val in top_3.items()]
        )
        print(f"{pc:4} -> {top_3_str}")

    return loadings_df


# ============================================================
# 7. RECONSTRUCTION ERROR (MSE) FOR k = 2 THROUGH 9
# ============================================================

def evaluate_reconstruction_error(df, pca):
    """Calculate reconstruction MSE and cumulative variance for k = 2..9."""

    X_scaled = df.values
    reconstruction_records = []

    for k in range(2, 10):
        pca_k = PCA(n_components=k)
        X_k = pca_k.fit_transform(X_scaled)
        X_recon = pca_k.inverse_transform(X_k)

        mse = float(np.mean((X_scaled - X_recon) ** 2))
        cum_var = float(np.sum(pca.explained_variance_ratio_[:k]))

        reconstruction_records.append({
            "Components": k,
            "Cumulative_Variance": cum_var,
            "Reconstruction_MSE": mse
        })

    recon_df = pd.DataFrame(reconstruction_records)

    print("\n" + "=" * 60)
    print("RECONSTRUCTION MSE COMPARISON TABLE (k = 2 to 9)")
    print("=" * 60)
    print(
        recon_df.to_string(
            index=False,
            formatters={
                "Cumulative_Variance": "{:.4f}".format,
                "Reconstruction_MSE": "{:.4f}".format
            }
        )
    )

    return recon_df


# ============================================================
# 8. BASELINE VS CLEANED COMPARISON
# ============================================================

def compare_with_baseline(df, components_needed, retained_variance, recon_df):
    """Compare Cleaned PCA results against Baseline PCA."""

    print("\n" + "=" * 60)
    print("BASELINE VS CLEANED PCA COMPARISON")
    print("=" * 60)

    print(f"{'Metric':<38} | {'Baseline':<12} | {'Cleaned':<12}")
    print("-" * 68)
    print(f"{'Dataset Rows':<38} | {BASELINE_ROWS:<12,} | {len(df):<12,}")
    print(f"{'PCA Audio Features':<38} | {BASELINE_FEATURES:<12} | {9:<12}")
    print(f"{'PCs required for >=90% variance':<38} | {BASELINE_PCS_90:<12} | {components_needed:<12}")
    print(f"{'Variance retained at selected PCs':<38} | {BASELINE_VARIANCE_90 * 100:.2f}%{'':<6} | {retained_variance * 100:.2f}%")
    print(f"{'2D Variance (PC1 + PC2)':<38} | {'47.73%':<12} | {recon_df.loc[recon_df['Components'] == 2, 'Cumulative_Variance'].values[0] * 100:.2f}%")
    print("-" * 68)
    print("* Note: Cleaned dataset contains 89,741 unique tracks.")


# ============================================================
# 9. SAVE CLEANED PCA RESULTS
# ============================================================

def save_cleaned_results(variance_df, loadings_df, recon_df, df, components_needed):
    """
    Save cleaned PCA variance table, loadings, reconstruction error,
    and the reduced dataset (at >=90% variance).
    """

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    variance_file = RESULTS_FOLDER / "cleaned_pca_explained_variance.csv"
    loadings_file = RESULTS_FOLDER / "cleaned_pca_loadings.csv"
    recon_file = RESULTS_FOLDER / "cleaned_pca_reconstruction_error.csv"
    reduced_file = RESULTS_FOLDER / "cleaned_pca_reduced_data.csv"

    variance_df.to_csv(variance_file, index=False)
    loadings_df.to_csv(loadings_file, index=True, index_label="feature")
    recon_df.to_csv(recon_file, index=False)

    # Generate reduced dataset with components_needed
    reduced_pca = PCA(n_components=components_needed)
    reduced_array = reduced_pca.fit_transform(df.values)
    pc_headers = [f"PC{i + 1}" for i in range(components_needed)]

    reduced_df = pd.DataFrame(
        reduced_array,
        columns=pc_headers,
        index=df.index
    )
    reduced_df.to_csv(reduced_file, index=False)

    print("\n" + "=" * 60)
    print("SAVING CLEANED PCA RESULTS")
    print("=" * 60)
    print(f"Cleaned Explained Variance   -> {variance_file}")
    print(f"Cleaned Loadings Matrix      -> {loadings_file}")
    print(f"Cleaned Reconstruction Error -> {recon_file}")
    print(f"Cleaned Reduced Data (7 PCs) -> {reduced_file} ({len(reduced_df):,} rows)")


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SPOTIFY MUSIC FEATURE REDUCTION USING PCA")
    print("PHASE 2 - DEEP CLEANED PCA ANALYSIS")
    print("=" * 60)

    # 1. Load cleaned standardized features
    df = load_cleaned_data()

    # 2. Fit PCA with all 9 components
    pca, transformed_data = fit_full_pca(df)

    # 3. Explained variance analysis
    variance_df, components_needed, retained_variance = analyze_explained_variance(pca)

    # 4. Loadings matrix & top contributors
    loadings_df = analyze_loadings(pca)

    # 5. Reconstruction error for k = 2..9
    recon_df = evaluate_reconstruction_error(df, pca)

    # 6. Baseline vs Cleaned comparison
    compare_with_baseline(df, components_needed, retained_variance, recon_df)

    # 7. Save all cleaned PCA results
    save_cleaned_results(
        variance_df,
        loadings_df,
        recon_df,
        df,
        components_needed
    )

    print("\n" + "=" * 60)
    print("CLEANED PCA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()