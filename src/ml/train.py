"""
Model Training and Comparison Orchestration Module.
Trains, tunes, compares, and serializes Logistic Regression and Random Forest classification pipelines
with cost-sensitive class balancing, cross-validation, and leak-free ColumnTransformers.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure matplotlib config directory is writable
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache"))

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from config.settings import (
    RAW_DATASET_PATH,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    EVALUATION_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils.logger import logger
from src.data.preprocessing import (
    clean_and_validate_data,
    split_features_and_target,
    get_stratified_train_test_split,
    build_column_preprocessor,
)
from src.features.feature_engineering import ClinicalFeatureEngineer
from src.ml.evaluate import (
    calculate_clinical_metrics,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curves,
    plot_and_save_pr_curves,
    evaluate_demographic_fairness,
)


def create_model_pipeline(
    classifier: Any,
    scale_numeric: bool = True
) -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn Pipeline combining:
    1. ClinicalFeatureEngineer (deriving bmi_category, age_group, cardio_comorbidity)
    2. ColumnTransformer (imputation, scaling, one-hot encoding)
    3. Classifier (LogisticRegression or RandomForestClassifier)
    
    Args:
        classifier: Estimator instance.
        scale_numeric: Whether to standard-scale numeric features.
        
    Returns:
        Pipeline: Untrained complete pipeline.
    """
    feature_engineer = ClinicalFeatureEngineer(include_comorbidity=True, include_binned=True)

    # Feature lists after engineering
    extended_num_cols = NUMERICAL_FEATURES
    extended_cat_cols = CATEGORICAL_FEATURES + ["bmi_category", "age_group"]
    extended_bin_cols = BINARY_FEATURES + ["cardio_comorbidity"]

    preprocessor = build_column_preprocessor(
        numerical_cols=extended_num_cols,
        categorical_cols=extended_cat_cols,
        binary_cols=extended_bin_cols,
        scale_numeric=scale_numeric
    )

    pipeline = Pipeline(steps=[
        ("feature_engineer", feature_engineer),
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline


def run_model_training_and_evaluation() -> Dict[str, Any]:
    """
    Executes complete end-to-end machine learning workflow:
    1. Ingestion & cleaning
    2. Stratified train-test split (80/20)
    3. 5-fold cross-validation on training data
    4. Fit Logistic Regression & Random Forest
    5. Test set evaluation & comparison
    6. Subgroup fairness auditing (Male vs Female)
    7. Artifact generation & pipeline serialization
    
    Returns:
        dict: Complete comparison metrics and model metadata.
    """
    logger.info("Starting Phase 5 Model Training & Evaluation Workflow...")

    # Step 1: Load and clean raw data
    raw_df = pd.read_csv(RAW_DATASET_PATH)
    clean_df = clean_and_validate_data(raw_df, drop_duplicates=True)
    X, y = split_features_and_target(clean_df)

    # Step 2: Stratified train-test split
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 3: Define Candidate Models with class balancing
    candidate_configs = {
        "Logistic_Regression": {
            "display_name": "Logistic Regression (Balanced)",
            "estimator": LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=RANDOM_STATE
            ),
            "scale_numeric": True
        },
        "Random_Forest": {
            "display_name": "Random Forest Classifier (Balanced)",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "scale_numeric": False # Trees are invariant to monotonic feature scale
        }
    }

    cv_results_summary: Dict[str, Any] = {}
    test_results_summary: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    roc_plot_data: list = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for model_key, config in candidate_configs.items():
        name = config["display_name"]
        logger.info(f"Building pipeline for {name}...")
        pipeline = create_model_pipeline(
            classifier=config["estimator"],
            scale_numeric=config["scale_numeric"]
        )

        # 5-fold Cross-Validation on Training Split ONLY
        logger.info(f"Running 5-fold Cross-Validation on X_train for {name}...")
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_summary = {
            "cv_accuracy_mean": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
            "cv_precision_mean": round(float(np.mean(cv_scores["test_precision"])), 4),
            "cv_recall_mean": round(float(np.mean(cv_scores["test_recall"])), 4),
            "cv_f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
            "cv_roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
        }
        cv_results_summary[model_key] = cv_summary
        logger.info(f"{name} 5-Fold CV Recall: {cv_summary['cv_recall_mean']:.4f} | ROC-AUC: {cv_summary['cv_roc_auc_mean']:.4f}")

        # Fit on full X_train
        logger.info(f"Fitting {name} on full X_train ({len(X_train)} samples)...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_key] = pipeline

        # Predict on untouched test set
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate Test Metrics
        test_metrics = calculate_clinical_metrics(y_test.values, y_test_pred, y_test_prob)
        test_results_summary[model_key] = test_metrics
        logger.info(f"{name} Test Set Evaluation: Recall={test_metrics['recall_sensitivity']:.4f}, Precision={test_metrics['precision']:.4f}, F1={test_metrics['f1_score']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}")

        roc_plot_data.append({
            "name": name,
            "y_true": y_test.values,
            "y_prob": y_test_prob
        })

    # Step 4: Generate Diagnostic Comparison Visualizations
    logger.info("Generating comparative evaluation charts in artifacts/evaluation/...")
    plot_and_save_roc_curves(roc_plot_data, EVALUATION_DIR / "roc_curves.png")
    plot_and_save_pr_curves(roc_plot_data, EVALUATION_DIR / "precision_recall_curves.png")

    # Step 5: Model Selection Decision
    # We choose the model based on F1-Score and ROC-AUC (combining high Recall and healthy Precision)
    rf_f1 = test_results_summary["Random_Forest"]["f1_score"]
    lr_f1 = test_results_summary["Logistic_Regression"]["f1_score"]
    
    selected_key = "Random_Forest" if rf_f1 >= lr_f1 else "Logistic_Regression"
    selected_pipeline = fitted_pipelines[selected_key]
    selected_name = candidate_configs[selected_key]["display_name"]
    logger.info(f"Model Selection Decision: '{selected_name}' chosen as production model (F1: {test_results_summary[selected_key]['f1_score']:.4f}, ROC-AUC: {test_results_summary[selected_key]['roc_auc']:.4f}).")

    # Save Confusion Matrix for the selected model
    best_y_pred = selected_pipeline.predict(X_test)
    plot_and_save_confusion_matrix(y_test.values, best_y_pred, selected_name, EVALUATION_DIR / "confusion_matrix.png")

    # Step 6: Feature Importance Analysis (for Random Forest)
    try:
        rf_model = fitted_pipelines["Random_Forest"].named_steps["classifier"]
        preproc = fitted_pipelines["Random_Forest"].named_steps["preprocessor"]
        cat_encoder = preproc.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES + ["bmi_category", "age_group"]))
        all_feature_names = NUMERICAL_FEATURES + encoded_cat_names + BINARY_FEATURES + ["cardio_comorbidity"]
        
        importances = rf_model.feature_importances_
        if len(all_feature_names) == len(importances):
            fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(12)

            plt.figure(figsize=(9, 5))
            sns.barplot(data=fi_df, x="importance", y="feature", palette="Blues_r")
            plt.title("Top 12 Most Important Clinical Features (Random Forest)")
            plt.xlabel("Gini Feature Importance")
            plt.tight_layout()
            plt.savefig(EVALUATION_DIR / "feature_importance.png", dpi=300)
            plt.close()
            logger.info("Saved feature_importance.png")
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")

    # Step 7: Subgroup Fairness Evaluation (Gender Cohorts)
    fairness_metrics = evaluate_demographic_fairness(
        selected_pipeline, X_test, y_test, EVALUATION_DIR / "subgroup_fairness_report.json"
    )

    # Step 8: Model Serialization
    MODEL_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PIPELINE_PATH)
    logger.info(f"Serialized winning pipeline to {MODEL_PIPELINE_PATH}")

    # Save Model Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "selected_model": selected_key,
        "selected_model_name": selected_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "models_evaluated": {
            "Logistic_Regression": {
                "cross_validation_5fold": cv_results_summary["Logistic_Regression"],
                "test_set_metrics": test_results_summary["Logistic_Regression"]
            },
            "Random_Forest": {
                "cross_validation_5fold": cv_results_summary["Random_Forest"],
                "test_set_metrics": test_results_summary["Random_Forest"]
            }
        },
        "demographic_fairness": fairness_metrics,
        "features_expected": list(X.columns)
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata cataloged at {MODEL_METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    meta = run_model_training_and_evaluation()
    print("\n" + "="*70)
    print("PHASE 5: MODEL TRAINING & COMPARISON COMPLETED")
    print("="*70)
    print(f"Selected Production Model: {meta['selected_model_name']}")
    print("\nModel Test Set Comparison:")
    for k, v in meta["models_evaluated"].items():
        tm = v["test_set_metrics"]
        print(f"  [{k}] Accuracy: {tm['accuracy']:.4f} | Recall: {tm['recall_sensitivity']:.4f} | Precision: {tm['precision']:.4f} | F1: {tm['f1_score']:.4f} | ROC-AUC: {tm['roc_auc']:.4f}")
    print("\nGender Subgroup Evaluation:")
    demo = meta["demographic_fairness"]
    if "gender_Female" in demo and "gender_Male" in demo:
        print(f"  Female Recall: {demo['gender_Female'].get('recall_sensitivity')} | Precision: {demo['gender_Female'].get('precision')}")
        print(f"  Male Recall:   {demo['gender_Male'].get('recall_sensitivity')} | Precision: {demo['gender_Male'].get('precision')}")
    print(f"\nModel serialized to: {MODEL_PIPELINE_PATH.resolve()}")
    print("="*70 + "\n")

