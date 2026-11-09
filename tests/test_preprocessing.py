"""
Unit Tests for Preprocessing and Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import clean_and_validate_data, build_column_preprocessor
from src.features.feature_engineering import ClinicalFeatureEngineer


def test_clean_and_validate_data():
    """Verify duplicate removal and sanity filtering."""
    df_raw = pd.DataFrame({
        "gender": ["Female", "Female", "Male"],
        "age": [30.0, 30.0, 45.0],
        "hypertension": [0, 0, 1],
        "heart_disease": [0, 0, 0],
        "smoking_history": ["never", "never", "former"],
        "bmi": [24.0, 24.0, 28.5],
        "HbA1c_level": [5.5, 5.5, 6.2],
        "blood_glucose_level": [100, 100, 140],
        "diabetes": [0, 0, 0]
    })
    cleaned = clean_and_validate_data(df_raw, drop_duplicates=True)
    assert len(cleaned) == 2, "Failed to drop exact duplicate row"


def test_clinical_feature_engineer():
    """Verify derivation of BMI categories, age brackets, and comorbidities."""
    df = pd.DataFrame({
        "age": [25.0, 45.0, 70.0],
        "bmi": [17.5, 26.0, 34.0],
        "hypertension": [0, 1, 1],
        "heart_disease": [0, 0, 1]
    })
    fe = ClinicalFeatureEngineer()
    trans = fe.transform(df)

    assert "bmi_category" in trans.columns
    assert "age_group" in trans.columns
    assert "cardio_comorbidity" in trans.columns
    assert list(trans["bmi_category"]) == ["underweight", "overweight", "obese"]
    assert list(trans["age_group"]) == ["young", "middle", "senior"]
    assert list(trans["cardio_comorbidity"]) == [0, 0, 1]


def test_column_preprocessor_shape():
    """Verify preprocessor produces correct column dimension without leaking."""
    preprocessor = build_column_preprocessor(
        numerical_cols=["age", "bmi"],
        categorical_cols=["gender", "smoking_history"],
        binary_cols=["hypertension"],
        scale_numeric=True
    )
    df_train = pd.DataFrame({
        "age": [20.0, 40.0, 60.0],
        "bmi": [22.0, 28.0, 32.0],
        "gender": ["Female", "Male", "Female"],
        "smoking_history": ["never", "current", "former"],
        "hypertension": [0, 1, 0]
    })
    out = preprocessor.fit_transform(df_train)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 3
    # 2 numeric + 2 gender + 3 smoking + 1 binary = 8 features
    assert out.shape[1] == 8

