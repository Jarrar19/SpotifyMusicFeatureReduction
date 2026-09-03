"""
Spotify Music Feature Reduction using PCA
------------------------------------------
Phase 2: Data Preprocessing (Cleaned Dataset)

This script:
1. Loads the raw Spotify dataset (114,000 tracks)
2. Deduplicates tracks by unique `track_id` (keeping the first occurrence)
3. Extracts aligned metadata (track_id, track_name, artists, album_name, track_genre, popularity)
4. Selects the 9 numerical audio features for PCA
5. Checks for missing values in selected features
6. Standardizes features using StandardScaler
7. Saves the cleaned selected features, standardized features, and aligned metadata
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
RESULTS_FOLDER = PROJECT_ROOT / "results"


# ============================================================
# 2. COLUMN DEFINITIONS
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

METADATA_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity"
]


# ============================================================
# 3. FIND DATASET
# ============================================================

def find_dataset():
    """Find the Spotify CSV file in the data folder."""

    csv_files = list(DATA_FOLDER.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in: {DATA_FOLDER}"
        )

    return csv_files[0]


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data(file_path):
    """Load the raw Spotify dataset."""

    print("\n" + "=" * 60)
    print("LOADING RAW DATASET")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"Dataset file : {file_path.name}")
    print(f"Raw rows     : {len(df):,}")
    print(f"Raw columns  : {len(df.columns)}")

    return df


# ============================================================
# 5. DEDUPLICATE BY TRACK_ID
# ============================================================

def deduplicate_by_track_id(df):
    """
    Deduplicate dataset by unique track_id.
    Retains the first occurrence for each unique track_id.
    """

    raw_rows = len(df)

    cleaned_df = df.drop_duplicates(
        subset=["track_id"],
        keep="first"
    ).reset_index(drop=True)

    cleaned_rows = len(cleaned_df)
    removed_rows = raw_rows - cleaned_rows

    print("\n" + "=" * 60)
    print("TRACK_ID DEDUPLICATION")
    print("=" * 60)

    print(f"Raw rows                          : {raw_rows:,}")
    print(f"Rows after track_id deduplication : {cleaned_rows:,}")
    print(f"Number of rows removed            : {removed_rows:,}")

    return cleaned_df


# ============================================================
# 6. EXTRACT ALIGNED METADATA
# ============================================================

def extract_metadata(df):
    """Extract aligned track metadata."""

    missing_cols = [
        col for col in METADATA_COLUMNS if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing metadata columns: {missing_cols}"
        )

    metadata = df[METADATA_COLUMNS].copy()

    print("\n" + "=" * 60)
    print("EXTRACTING METADATA")
    print("=" * 60)

    print(f"Metadata columns  : {', '.join(METADATA_COLUMNS)}")
    print(f"Metadata rows     : {len(metadata):,}")

    return metadata


# ============================================================
# 7. SELECT AUDIO FEATURES
# ============================================================

def select_audio_features(df):
    """Select the nine audio features used for PCA."""

    print("\n" + "=" * 60)
    print("SELECTING PCA FEATURES")
    print("=" * 60)

    missing_features = [
        feature
        for feature in AUDIO_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following required features are missing:\n"
            + "\n".join(missing_features)
        )

    features = df[AUDIO_FEATURES].copy()

    print("Selected PCA audio features:")
    for feature in AUDIO_FEATURES:
        print(f"  - {feature}")

    print(f"\nNumber of selected PCA features : {features.shape[1]}")
    print(f"Number of tracks                : {features.shape[0]:,}")

    return features


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

def check_missing_values(features):
    """Check missing values in PCA features."""

    print("\n" + "=" * 60)
    print("MISSING VALUES IN SELECTED FEATURES")
    print("=" * 60)

    missing = features.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found in selected PCA features.")
    else:
        print("Missing values found:")
        print(missing[missing > 0])


# ============================================================
# 9. STANDARDIZE FEATURES
# ============================================================

def standardize_features(features):
    """
    Standardize the selected features using StandardScaler.

    Each feature is transformed to mean ≈ 0 and standard deviation ≈ 1.
    """

    print("\n" + "=" * 60)
    print("STANDARDIZING FEATURES (StandardScaler)")
    print("=" * 60)

    scaler = StandardScaler()

    standardized_array = scaler.fit_transform(features)

    standardized_df = pd.DataFrame(
        standardized_array,
        columns=AUDIO_FEATURES,
        index=features.index
    )

    print("Standardization completed successfully.")

    print("\nMean after StandardScaler (should be ~0.0):")
    print(standardized_df.mean().round(6).to_string())

    print("\nStandard deviation after StandardScaler (should be ~1.0):")
    print(standardized_df.std().round(6).to_string())

    return standardized_df, scaler


# ============================================================
# 10. SAVE CLEANED AND STANDARDIZED DATA
# ============================================================

def save_cleaned_data(features, standardized_features, metadata):
    """
    Save cleaned selected features, standardized features, and metadata.
    All files maintain identical row order and row counts.
    """

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    selected_path = RESULTS_FOLDER / "cleaned_selected_audio_features.csv"
    standardized_path = RESULTS_FOLDER / "cleaned_standardized_audio_features.csv"
    metadata_path = RESULTS_FOLDER / "cleaned_track_metadata.csv"

    features.to_csv(selected_path, index=False)
    standardized_features.to_csv(standardized_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    print(f"Cleaned selected features     -> {selected_path} ({len(features):,} rows)")
    print(f"Cleaned standardized features -> {standardized_path} ({len(standardized_features):,} rows)")
    print(f"Cleaned aligned metadata      -> {metadata_path} ({len(metadata):,} rows)")


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SPOTIFY MUSIC FEATURE REDUCTION USING PCA")
    print("PHASE 2 - DATA PREPROCESSING (CLEANED DATASET)")
    print("=" * 60)

    # 1. Find dataset
    dataset_path = find_dataset()

    # 2. Load raw dataset
    df = load_data(dataset_path)

    # 3. Deduplicate by track_id
    df_cleaned = deduplicate_by_track_id(df)

    # 4. Extract aligned metadata
    metadata = extract_metadata(df_cleaned)

    # 5. Select 9 audio features
    features = select_audio_features(df_cleaned)

    # 6. Check missing values
    check_missing_values(features)

    # 7. Standardize features
    standardized_features, scaler = standardize_features(features)

    # 8. Save cleaned data
    save_cleaned_data(
        features,
        standardized_features,
        metadata
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Raw rows                          : {len(df):,}")
    print(f"Rows after track_id deduplication : {len(df_cleaned):,}")
    print(f"Number of rows removed            : {len(df) - len(df_cleaned):,}")
    print(f"Number of selected PCA features   : {features.shape[1]}")
    print(f"Missing values in PCA features    : {features.isnull().sum().sum()}")
    print("All output files saved and row-aligned.")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()