"""
Feature Engineering Module.
Implements clinically grounded derived features as a Scikit-Learn compatible transformer.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import logger


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer that derives clinically validated indicators:
    1. bmi_category (Underweight, Normal, Overweight, Obese) based on WHO guidelines.
    2. age_group (Young, Middle, Senior) based on ADA/USPSTF risk brackets.
    3. cardio_comorbidity (1 if patient has both hypertension and heart disease, else 0).
    
    Operates safely in both training and real-time inference pipelines.
    """
    def __init__(self, include_comorbidity: bool = True, include_binned: bool = True):
        self.include_comorbidity = include_comorbidity
        self.include_binned = include_binned

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame by appending derived clinical features.
        
        Args:
            X: Input DataFrame with raw features.
            
        Returns:
            pd.DataFrame: Transformed DataFrame with engineered features.
        """
        df = X.copy()

        # 1. Cardiovascular Comorbidity Interaction
        if self.include_comorbidity and "hypertension" in df.columns and "heart_disease" in df.columns:
            df["cardio_comorbidity"] = (
                (df["hypertension"] == 1) & (df["heart_disease"] == 1)
            ).astype(int)

        # 2. WHO BMI Categories
        if self.include_binned and "bmi" in df.columns:
            df["bmi_category"] = pd.cut(
                df["bmi"],
                bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
                labels=["underweight", "normal", "overweight", "obese"],
                right=True
            ).astype(str)

        # 3. Clinical Age Brackets
        if self.include_binned and "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[-np.inf, 35.0, 54.0, np.inf],
                labels=["young", "middle", "senior"],
                right=True
            ).astype(str)

        return df


if __name__ == "__main__":
    logger.info("Testing ClinicalFeatureEngineer Transformer...")
    sample_df = pd.DataFrame({
        "gender": ["Female", "Male", "Female"],
        "age": [28.0, 50.0, 68.0],
        "hypertension": [0, 1, 1],
        "heart_disease": [0, 0, 1],
        "smoking_history": ["never", "current", "former"],
        "bmi": [22.4, 28.1, 35.6],
        "HbA1c_level": [5.2, 6.8, 7.5],
        "blood_glucose_level": [95, 160, 210]
    })
    fe = ClinicalFeatureEngineer()
    transformed = fe.transform(sample_df)
    print("\n" + "="*60)
    print("FEATURE ENGINEERING TEST")
    print("="*60)
    print("Original Columns:", list(sample_df.columns))
    print("Transformed Columns:", list(transformed.columns))
    print("\nSample Output:")
    print(transformed[["age", "age_group", "bmi", "bmi_category", "cardio_comorbidity"]])
    print("="*60 + "\n")

