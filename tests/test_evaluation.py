"""
Unit Tests for Model Evaluation and Clinical Metrics.
"""

import pytest
import numpy as np

from src.ml.evaluate import calculate_clinical_metrics


def test_calculate_clinical_metrics_perfect_classifier():
    """Verify metrics calculation on perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_clinical_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["recall_sensitivity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_negatives"] == 0
    assert metrics["false_positives"] == 0


def test_calculate_clinical_metrics_imperfect_classifier():
    """Verify metrics calculation on realistic confusion matrix."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.4, 0.85])

    metrics = calculate_clinical_metrics(y_true, y_pred, y_prob)

    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["recall_sensitivity"] == 0.5
    assert round(metrics["precision"], 2) == 0.5

