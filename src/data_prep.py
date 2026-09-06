"""
Data Ingestion, Validation, and Preprocessing Module
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_default_data_path():
    """Locate the dataset path reliably regardless of invocation directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary_path = os.path.join(base_dir, "data", "Advertising.csv")
    if os.path.exists(primary_path):
        return primary_path
    # Fallback checks
    alt_path = os.path.join(base_dir, "data", "advertising_classic.csv")
    if os.path.exists(alt_path):
        return alt_path
    return primary_path


def load_and_validate_data(filepath=None):
    """
    Load advertising dataset, remove artifacts, and validate schema.
    Returns: cleaned DataFrame.
    """
    if filepath is None:
        filepath = get_default_data_path()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    df = pd.read_csv(filepath)

    # 1. Clean index artifacts
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2. Schema check
    required_cols = ["TV", "Radio", "Newspaper", "Sales"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset.")

    # 3. Data Integrity & Validation
    # Cast to float
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values if any
    if df[required_cols].isnull().any().any():
        df[required_cols] = df[required_cols].fillna(df[required_cols].median())

    # Check for negative spend
    for col in ["TV", "Radio", "Newspaper", "Sales"]:
        if (df[col] < 0).any():
            df[col] = df[col].clip(lower=0)

    return df[required_cols]


def get_train_test_data(filepath=None, test_size=0.20, random_state=42):
    """
    Load, validate, and partition data into train and test sets.
    Features: ['TV', 'Radio', 'Newspaper']
    Target: 'Sales'
    """
    df = load_and_validate_data(filepath)
    feature_cols = ["TV", "Radio", "Newspaper"]
    target_col = "Sales"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
