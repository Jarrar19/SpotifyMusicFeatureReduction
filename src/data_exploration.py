"""
Spotify Music Feature Reduction using PCA
------------------------------------------
Phase 2: Dataset Exploration

This script loads the Spotify Tracks Dataset and performs
basic data exploration before preprocessing and PCA.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. FIND THE PROJECT AND DATASET
# ============================================================

# Project root = folder containing this src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"


def find_dataset():
    """Find the first CSV file inside the data folder."""

    csv_files = list(DATA_FOLDER.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in the data folder.\n"
            f"Please place the Spotify dataset inside: {DATA_FOLDER}"
        )

    if len(csv_files) > 1:
        print("Multiple CSV files found:")
        for file in csv_files:
            print(f"  - {file.name}")
        print("\nUsing the first CSV file.")

    return csv_files[0]


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """Load the Spotify dataset."""

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"Dataset file : {file_path.name}")
    print(f"Rows         : {df.shape[0]:,}")
    print(f"Columns      : {df.shape[1]}")

    return df


# ============================================================
# 3. DISPLAY COLUMN INFORMATION
# ============================================================

def show_columns(df):
    """Display all columns in the dataset."""

    print("\n" + "=" * 60)
    print("DATASET COLUMNS")
    print("=" * 60)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number:2}. {column}")


# ============================================================
# 4. DATA TYPES
# ============================================================

def show_data_types(df):
    """Display data types of all columns."""

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

def check_missing_values(df):
    """Check for missing values."""

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0])


# ============================================================
# 6. DUPLICATE ROWS
# ============================================================

def check_duplicates(df):
    """Check for duplicate rows."""

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates:,}")


# ============================================================
# 7. NUMERICAL FEATURES
# ============================================================

def show_numerical_features(df):
    """Display numerical columns."""

    print("\n" + "=" * 60)
    print("NUMERICAL FEATURES")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for column in numerical_columns:
        print(f"- {column}")

    print(f"\nTotal numerical columns: {len(numerical_columns)}")

    return numerical_columns


# ============================================================
# 8. SPOTIFY AUDIO FEATURES
# ============================================================

def identify_audio_features(df):
    """
    Identify the Spotify audio features that are relevant
    for our PCA analysis.
    """

    planned_features = [
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

    available_features = [
        feature
        for feature in planned_features
        if feature in df.columns
    ]

    missing_features = [
        feature
        for feature in planned_features
        if feature not in df.columns
    ]

    print("\n" + "=" * 60)
    print("AUDIO FEATURES FOR PCA")
    print("=" * 60)

    print("Available planned features:")

    for feature in available_features:
        print(f"- {feature}")

    if missing_features:
        print("\nFeatures not found in dataset:")

        for feature in missing_features:
            print(f"- {feature}")

    print(
        f"\nAudio features available for PCA: "
        f"{len(available_features)}"
    )

    return available_features


# ============================================================
# 9. BASIC STATISTICS
# ============================================================

def show_statistics(df, audio_features):
    """Display basic statistics for audio features."""

    print("\n" + "=" * 60)
    print("BASIC STATISTICS OF AUDIO FEATURES")
    print("=" * 60)

    if audio_features:
        print(df[audio_features].describe().T)
    else:
        print("No audio features available.")


# ============================================================
# 10. FIRST FEW RECORDS
# ============================================================

def show_sample(df):
    """Display the first five records."""

    print("\n" + "=" * 60)
    print("FIRST 5 RECORDS")
    print("=" * 60)

    print(df.head())


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SPOTIFY MUSIC FEATURE REDUCTION USING PCA")
    print("PHASE 2 - DATASET EXPLORATION")
    print("=" * 60)

    # Find CSV
    dataset_path = find_dataset()

    # Load dataset
    df = load_dataset(dataset_path)

    # Show sample
    show_sample(df)

    # Show columns
    show_columns(df)

    # Show data types
    show_data_types(df)

    # Check missing values
    check_missing_values(df)

    # Check duplicates
    check_duplicates(df)

    # Show numerical columns
    show_numerical_features(df)

    # Identify audio features
    audio_features = identify_audio_features(df)

    # Show statistics
    show_statistics(df, audio_features)

    print("\n" + "=" * 60)
    print("DATA EXPLORATION COMPLETED")
    print("=" * 60)


# Run the program
if __name__ == "__main__":
    main()