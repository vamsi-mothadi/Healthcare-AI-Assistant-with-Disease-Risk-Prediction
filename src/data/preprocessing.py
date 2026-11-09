"""
Data Preprocessing Pipeline Module.
Implements leak-free data cleaning, stratified train-test splitting, and Scikit-learn ColumnTransformer.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config.settings import (
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils.logger import logger


def clean_and_validate_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    Cleans raw dataframe: audits invalid values, checks sanity ranges,
    and optionally removes exact duplicate records to prevent train-test data leakage.
    
    Args:
        df: Raw DataFrame.
        drop_duplicates: Whether to drop exact duplicate rows.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_count = len(df)
    cleaned = df.copy()

    # Range sanity filters (preserving extreme but clinically valid values)
    # Age between 0 and 120
    cleaned = cleaned[(cleaned["age"] >= 0) & (cleaned["age"] <= 120)]
    # BMI between 10 and 100
    cleaned = cleaned[(cleaned["bmi"] >= 10.0) & (cleaned["bmi"] <= 100.0)]
    # Blood glucose between 50 and 500
    cleaned = cleaned[(cleaned["blood_glucose_level"] >= 50) & (cleaned["blood_glucose_level"] <= 500)]

    if drop_duplicates:
        dups = cleaned.duplicated().sum()
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dropped {dups} duplicate rows ({dups / initial_count * 100:.2f}%). Cleaned records: {len(cleaned)}")
    else:
        logger.info(f"Retained duplicates. Cleaned records: {len(cleaned)}")

    return cleaned


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits DataFrame into features X and target y."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def get_stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test split to preserve exact positive/negative class ratio.
    
    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset to hold out as test.
        random_state: Reproducible seed.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting dataset: test_size={test_size}, random_state={random_state}, stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(
        f"Train size: {len(X_train)} (Positive: {y_train.sum()} / {y_train.mean()*100:.2f}%) | "
        f"Test size: {len(X_test)} (Positive: {y_test.sum()} / {y_test.mean()*100:.2f}%)"
    )
    return X_train, X_test, y_train, y_test


def build_column_preprocessor(
    numerical_cols: List[str] = NUMERICAL_FEATURES,
    categorical_cols: List[str] = CATEGORICAL_FEATURES,
    binary_cols: List[str] = BINARY_FEATURES,
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Constructs a leak-free ColumnTransformer pipeline.
    
    - Numerical: Median Imputation -> (Optional) StandardScaler
    - Categorical: Mode Imputation -> OneHotEncoder(handle_unknown='ignore')
    - Binary: Pass-through (hypertension, heart_disease)
    
    Args:
        numerical_cols: List of continuous features.
        categorical_cols: List of categorical features.
        binary_cols: List of binary features.
        scale_numeric: Whether to apply StandardScaler.
        
    Returns:
        ColumnTransformer: Configured transformer ready to be fit on X_train.
    """
    # 1. Numerical Pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numerical_transformer = Pipeline(steps=num_steps)

    # 2. Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. Binary Pipeline (Ensure non-null)
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("bin", binary_transformer, binary_cols)
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    logger.info("Testing Preprocessing Module...")
    raw_df = pd.read_csv(RAW_DATASET_PATH)
    clean_df = clean_and_validate_data(raw_df, drop_duplicates=True)
    X, y = split_features_and_target(clean_df)
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(X, y)
    
    preprocessor = build_column_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE VERIFICATION")
    print("="*60)
    print(f"Original shape: {raw_df.shape}")
    print(f"Cleaned shape:  {clean_df.shape}")
    print(f"X_train shape after ColumnTransformer: {X_train_trans.shape}")
    print(f"X_test shape after ColumnTransformer:  {X_test_trans.shape}")
    print("Zero Data Leakage: Preprocessor fit solely on X_train!")
    print("="*60 + "\n")

