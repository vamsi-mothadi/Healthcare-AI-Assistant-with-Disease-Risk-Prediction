"""
Data Ingestion and Initial Validation Module.
Handles downloading, schema verification, data integrity checks, and inventory reporting
for the Diabetes Prediction Dataset.
"""

import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import (
    RAW_DATASET_PATH,
    DATA_RAW_DIR,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)
from src.utils.logger import logger

# Public direct mirror for the Kaggle dataset 'iammustafatz/diabetes-prediction-dataset'
DATASET_PUBLIC_URL = "https://raw.githubusercontent.com/hdg204/DoctorsAsDataScientists/main/diabetes_prediction_dataset.csv"
KAGGLE_SOURCE_URL = "https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"


def check_dataset_exists(file_path: Path = RAW_DATASET_PATH) -> bool:
    """Checks whether the raw CSV dataset file already exists and is non-empty."""
    return file_path.exists() and file_path.stat().st_size > 0


def download_dataset_if_missing(
    destination_path: Path = RAW_DATASET_PATH,
    url: str = DATASET_PUBLIC_URL
) -> bool:
    """
    Downloads the diabetes prediction dataset if not already present.
    
    Args:
        destination_path: Path where the CSV should be saved.
        url: Remote URL pointing to the raw CSV dataset.
        
    Returns:
        bool: True if file exists or download succeeded, False otherwise.
    """
    if check_dataset_exists(destination_path):
        logger.info(f"Dataset already exists at {destination_path}. Skipping download.")
        return True

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading diabetes prediction dataset from {url}...")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(destination_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        logger.info(f"Dataset successfully downloaded to {destination_path} ({len(data)} bytes).")
        return True
    except Exception as e:
        logger.error(
            f"Automated dataset download failed: {e}\n"
            f"Manual Download Instructions:\n"
            f"1. Visit the Kaggle URL: {KAGGLE_SOURCE_URL}\n"
            f"2. Sign in to your Kaggle account and click 'Download' (archive.zip).\n"
            f"3. Extract 'diabetes_prediction_dataset.csv'.\n"
            f"4. Place the file at: '{destination_path.resolve()}'."
        )
        return False


def load_raw_dataset(file_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """
    Loads raw CSV dataset into a pandas DataFrame.
    
    Args:
        file_path: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the file is empty or corrupted.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV dataset at {file_path}: {e}")

    if df.empty:
        raise ValueError(f"Dataset loaded from {file_path} is empty.")

    logger.info(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def validate_raw_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive schema and data health checks on the raw dataset.
    
    Args:
        df: Input DataFrame to inspect.
        
    Returns:
        dict: Summary report of data validation metrics.
    """
    expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [TARGET_COLUMN]
    actual_columns = list(df.columns)
    missing_expected_cols = [c for c in expected_features if c not in actual_columns]
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_count / len(df)) * 100, 2)
    
    # Target distribution
    if TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts().to_dict()
        target_ratio = {
            str(k): round(v / len(df), 4) for k, v in target_counts.items()
        }
    else:
        target_counts, target_ratio = {}, {}

    # Gender breakdown
    gender_breakdown = df["gender"].value_counts().to_dict() if "gender" in df.columns else {}

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": actual_columns,
        "missing_expected_columns": missing_expected_cols,
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "total_missing_values": total_missing,
        "missing_per_column": missing_counts,
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "target_column": TARGET_COLUMN,
        "target_distribution_counts": target_counts,
        "target_distribution_ratios": target_ratio,
        "gender_distribution": gender_breakdown,
        "validation_passed": len(missing_expected_cols) == 0 and len(df) > 0
    }
    return report


def generate_data_inventory_report(
    df: pd.DataFrame,
    output_path: Path = DATA_RAW_DIR / "data_inventory_report.json"
) -> Dict[str, Any]:
    """
    Generates and saves the formal data inventory report to disk.
    
    Args:
        df: Input DataFrame.
        output_path: Destination path for JSON report.
        
    Returns:
        dict: The inventory report dictionary.
    """
    report = validate_raw_dataset(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Data inventory report saved to {output_path}")
    return report


if __name__ == "__main__":
    logger.info("Initiating Phase 2 Data Ingestion & Validation...")
    success = download_dataset_if_missing()
    if success:
        dataset = load_raw_dataset()
        inv_report = generate_data_inventory_report(dataset)
        print("\n" + "="*60)
        print("DATA INVENTORY SUMMARY")
        print("="*60)
        print(f"Total Records: {inv_report['total_records']:,}")
        print(f"Total Features: {inv_report['total_columns']}")
        print(f"Target Column: '{inv_report['target_column']}'")
        print(f"Target Distribution: {inv_report['target_distribution_counts']}")
        print(f"Missing Values: {inv_report['total_missing_values']}")
        print(f"Duplicate Rows: {inv_report['duplicate_rows']} ({inv_report['duplicate_percentage']}%)")
        print(f"Gender Breakdown: {inv_report['gender_distribution']}")
        print(f"Validation Status: {'PASSED' if inv_report['validation_passed'] else 'FAILED'}")
        print("="*60 + "\n")
