"""
Unit Tests for Data Ingestion, Schema Validation, and Integrity.
"""

import pytest
import pandas as pd
from pathlib import Path

from config.settings import RAW_DATASET_PATH, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
from src.data.load_data import load_raw_dataset, validate_raw_dataset, check_dataset_exists


def test_dataset_file_exists():
    """Verify that the raw diabetes dataset exists and is non-empty."""
    assert check_dataset_exists(RAW_DATASET_PATH), f"Raw dataset not found at {RAW_DATASET_PATH}"


def test_load_raw_dataset_schema():
    """Verify that the loaded dataset contains all expected features and target."""
    df = load_raw_dataset(RAW_DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [TARGET_COLUMN]
    for col in expected_features:
        assert col in df.columns, f"Expected feature '{col}' missing from raw dataset"


def test_dataset_dimensions():
    """Verify the expected record count is roughly 100,000 and has 9 columns."""
    df = load_raw_dataset(RAW_DATASET_PATH)
    assert df.shape[0] == 100000, f"Expected 100,000 rows, found {df.shape[0]}"
    assert df.shape[1] == 9, f"Expected 9 columns, found {df.shape[1]}"


def test_target_column_binary_values():
    """Verify target column contains only binary values 0 and 1."""
    df = load_raw_dataset(RAW_DATASET_PATH)
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    assert unique_targets == {0, 1}, f"Unexpected target classes: {unique_targets}"


def test_validate_raw_dataset_report():
    """Verify data validation report generates expected metrics and passes."""
    df = load_raw_dataset(RAW_DATASET_PATH)
    report = validate_raw_dataset(df)

    assert report["validation_passed"] is True
    assert report["total_records"] == 100000
    assert report["target_column"] == TARGET_COLUMN
    assert "Female" in report["gender_distribution"]
    assert "Male" in report["gender_distribution"]

