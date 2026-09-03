"""
Spotify Music Feature Reduction using PCA
------------------------------------------
Phase 2: Data Preprocessing

This script:
1. Loads the raw Spotify dataset
2. Removes duplicate rows
3. Selects the audio features required for PCA
4. Checks the selected data
5. Standardizes the features
6. Saves the processed data
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
# 2. FEATURES FOR PCA
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
    """Load the Spotify dataset."""

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"Dataset : {file_path.name}")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""

    before = len(df)

    df = df.drop_duplicates().copy()

    after = len(df)

    print("\n" + "=" * 60)
    print("DUPLICATE REMOVAL")
    print("=" * 60)

    print(f"Rows before removing duplicates : {before:,}")
    print(f"Duplicate rows removed          : {before - after:,}")
    print(f"Rows after removing duplicates  : {after:,}")

    return df


# ============================================================
# 6. SELECT AUDIO FEATURES
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

    print("Features selected for PCA:")

    for feature in AUDIO_FEATURES:
        print(f"- {feature}")

    print(f"\nNumber of features: {features.shape[1]}")
    print(f"Number of tracks  : {features.shape[0]:,}")

    return features


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

def check_missing_values(features):
    """Check missing values in PCA features."""

    print("\n" + "=" * 60)
    print("CHECKING PCA FEATURES")
    print("=" * 60)

    missing = features.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found in PCA features.")
    else:
        print("Missing values found:")
        print(missing[missing > 0])


# ============================================================
# 8. STANDARDIZE FEATURES
# ============================================================

def standardize_features(features):
    """
    Standardize the selected features.

    Each feature is transformed so that it has approximately:
    mean = 0
    standard deviation = 1
    """

    print("\n" + "=" * 60)
    print("STANDARDIZING FEATURES")
    print("=" * 60)

    scaler = StandardScaler()

    standardized_array = scaler.fit_transform(features)

    standardized_df = pd.DataFrame(
        standardized_array,
        columns=AUDIO_FEATURES,
        index=features.index
    )

    print("Standardization completed.")
    print("\nMean after standardization:")

    print(standardized_df.mean().round(4))

    print("\nStandard deviation after standardization:")

    print(standardized_df.std().round(4))

    return standardized_df, scaler


# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

def save_processed_data(features, standardized_features):
    """Save cleaned and standardized data."""

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    selected_path = RESULTS_FOLDER / "selected_audio_features.csv"
    standardized_path = RESULTS_FOLDER / "standardized_audio_features.csv"

    features.to_csv(selected_path, index=False)
    standardized_features.to_csv(standardized_path, index=False)

    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    print(f"Selected features     → {selected_path}")
    print(f"Standardized features → {standardized_path}")


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SPOTIFY MUSIC FEATURE REDUCTION USING PCA")
    print("PHASE 2 - DATA PREPROCESSING")
    print("=" * 60)

    # Find dataset
    dataset_path = find_dataset()

    # Load dataset
    df = load_data(dataset_path)

    # Remove duplicate rows
    df = remove_duplicates(df)

    # Select PCA features
    features = select_audio_features(df)

    # Check missing values
    check_missing_values(features)

    # Standardize features
    standardized_features, scaler = standardize_features(features)

    # Save processed data
    save_processed_data(
        features,
        standardized_features
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()