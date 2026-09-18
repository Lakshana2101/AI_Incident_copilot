"""
preprocessing.py
-----------------
Simple, beginner-friendly data cleaning functions for the incident dataset.
"""

import pandas as pd

NUMERIC_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "response_time",
    "error_count",
    "network_latency",
]


def load_and_clean_data(csv_path):
    """
    Load the incidents CSV and apply basic cleaning:
    - drop duplicate rows
    - handle missing values
    - fix data types
    - clean up text fields
    """
    df = pd.read_csv(csv_path)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Convert timestamp to a real datetime column
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 3. Fill missing numeric values with the column median (simple + safe)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # 4. Fill missing categorical/text values with sensible defaults
    df["category"] = df["category"].fillna("Application")
    df["severity"] = df["severity"].fillna("Low")
    df["error_message"] = df["error_message"].fillna("No error message provided")
    df["resolution"] = df["resolution"].fillna("No resolution recorded")

    # 5. Basic text cleaning on the error message (strip whitespace, etc.)
    df["error_message"] = df["error_message"].astype(str).str.strip()

    # 6. Drop rows where timestamp could not be parsed at all
    df = df.dropna(subset=["timestamp"])

    # 7. Reset index after cleaning
    df = df.reset_index(drop=True)

    return df


def get_feature_matrix(df):
    """Return just the numeric feature columns used by the ML models."""
    return df[NUMERIC_COLUMNS].copy()
