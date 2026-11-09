"""
Unit Tests for Streamlit Application Helpers and Component Integration.
"""

import pytest
import os
from pathlib import Path

from config.settings import MODEL_PIPELINE_PATH, DOCUMENTS_INVENTORY_PATH, EVALUATION_DIR
from src.ml.predict import DiabetesPredictor
from src.rag.chatbot import HealthcareChatbot


def test_app_assets_exist():
    """Verify that all core artifacts and charts required by the Streamlit app exist."""
    assert MODEL_PIPELINE_PATH.exists(), "Trained model pipeline missing"
    assert (EVALUATION_DIR / "roc_curves.png").exists(), "ROC curves chart missing"
    assert (EVALUATION_DIR / "confusion_matrix.png").exists(), "Confusion matrix chart missing"
    assert (EVALUATION_DIR / "precision_recall_curves.png").exists(), "PR curves chart missing"
    assert (EVALUATION_DIR / "feature_importance.png").exists(), "Feature importance chart missing"
    assert DOCUMENTS_INVENTORY_PATH.exists(), "Documents inventory missing"


def test_missing_model_handling(tmp_path):
    """Verify DiabetesPredictor raises clean FileNotFoundError if model is missing."""
    non_existent = tmp_path / "missing_model.joblib"
    with pytest.raises(FileNotFoundError):
        DiabetesPredictor(model_path=non_existent)


def test_missing_api_key_fallback():
    """Verify HealthcareChatbot provides graceful fallback when API key is missing."""
    # Test with empty API key
    bot = HealthcareChatbot(api_key="", mock_mode=False)
    response = bot.ask("What is prediabetes?")
    assert "answer" in response
    assert response["context_used"] is True
    assert "fallback-retrieval" in response.get("model", "")
    assert len(response["citations"]) > 0

