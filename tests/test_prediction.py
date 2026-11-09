"""
Unit Tests for Diabetes Risk Prediction Engine.
"""

import pytest
import pandas as pd
from pydantic import ValidationError

from src.ml.predict import DiabetesPredictor, PatientInputSchema, MEDICAL_DISCLAIMER


@pytest.fixture(scope="module")
def predictor():
    """Initializes and caches predictor instance."""
    return DiabetesPredictor()


def test_predictor_healthy_patient(predictor):
    """Verify prediction output for a healthy patient profile."""
    healthy = {
        "gender": "Female",
        "age": 25.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 21.0,
        "HbA1c_level": 4.5,
        "blood_glucose_level": 85.0
    }
    result = predictor.predict_single(healthy)

    assert result["prediction_class"] == 0
    assert result["risk_tier"] == "Low Risk"
    assert result["probability_diabetic"] < 0.30
    assert "disclaimer" in result
    assert MEDICAL_DISCLAIMER in result["disclaimer"]


def test_predictor_high_risk_patient(predictor):
    """Verify prediction output for an elevated-risk patient profile."""
    high_risk = {
        "gender": "Male",
        "age": 65.0,
        "hypertension": 1,
        "heart_disease": 1,
        "smoking_history": "current",
        "bmi": 38.0,
        "HbA1c_level": 8.0,
        "blood_glucose_level": 240.0
    }
    result = predictor.predict_single(high_risk)

    assert result["prediction_class"] == 1
    assert result["risk_tier"] == "High Risk"
    assert result["probability_diabetic"] > 0.60


def test_predictor_gender_support(predictor):
    """Verify predictor supports Female, Male, and Other gender categories."""
    for gender in ["Female", "Male", "Other"]:
        patient = {
            "gender": gender,
            "age": 40.0,
            "hypertension": 0,
            "heart_disease": 0,
            "smoking_history": "No Info",
            "bmi": 26.0,
            "HbA1c_level": 5.5,
            "blood_glucose_level": 110.0
        }
        res = predictor.predict_single(patient)
        assert res["prediction_class"] in [0, 1]
        assert 0.0 <= res["probability_diabetic"] <= 1.0


def test_patient_input_schema_rejection():
    """Verify Pydantic rejects physiologically impossible or out-of-bound inputs."""
    with pytest.raises(ValidationError):
        # Negative age
        PatientInputSchema(
            gender="Female", age=-5.0, hypertension=0, heart_disease=0,
            smoking_history="never", bmi=22.0, HbA1c_level=5.0, blood_glucose_level=100.0
        )

    with pytest.raises(ValidationError):
        # Out-of-bounds glucose level (1200 mg/dL)
        PatientInputSchema(
            gender="Male", age=50.0, hypertension=0, heart_disease=0,
            smoking_history="never", bmi=25.0, HbA1c_level=5.0, blood_glucose_level=1200.0
        )

