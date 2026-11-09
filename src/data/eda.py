"""
Exploratory Data Analysis (EDA) Module.
Performs thorough statistical analysis, data quality checks, and generates publication-quality
visualizations saved to artifacts/evaluation/.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Ensure matplotlib config directory is writable
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache"))

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import (
    RAW_DATASET_PATH,
    EVALUATION_DIR,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)
from src.utils.logger import logger

# Set aesthetic visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})


def run_full_eda(
    data_path: Path = RAW_DATASET_PATH,
    output_dir: Path = EVALUATION_DIR
) -> Dict[str, Any]:
    """
    Executes full statistical and visual exploratory data analysis.
    
    Args:
        data_path: Path to raw CSV.
        output_dir: Directory where evaluation plots and summaries are saved.
        
    Returns:
        dict: Detailed statistical findings and EDA observations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading raw dataset for EDA from {data_path}...")
    df = pd.read_csv(data_path)

    findings: Dict[str, Any] = {}
    total_records = len(df)

    # 1. Target Class Distribution Analysis
    class_counts = df[TARGET_COLUMN].value_counts()
    class_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    findings["class_distribution"] = {
        "negative_class_0": int(class_counts[0]),
        "positive_class_1": int(class_counts[1]),
        "negative_percentage": round(float(class_pct[0]), 2),
        "positive_percentage": round(float(class_pct[1]), 2),
        "imbalance_ratio": round(float(class_counts[0] / class_counts[1]), 2)
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(
        x=["Non-Diabetic (Class 0)", "Diabetic (Class 1)"],
        y=class_counts.values,
        palette=["#2b5c8f", "#d9534f"],
        ax=ax
    )
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}\n({height/total_records*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontweight="bold"
        )
    ax.set_title("Diabetes Target Class Distribution (Class Imbalance Analysis)")
    ax.set_ylabel("Patient Count")
    plt.tight_layout()
    chart1_path = output_dir / "eda_class_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    logger.info(f"Saved {chart1_path.name}")

    # 2. Demographic Analysis (Gender & Age)
    gender_counts = df["gender"].value_counts().to_dict()
    findings["gender_distribution"] = gender_counts
    
    # Calculate diabetes prevalence by gender
    gender_prevalence = (
        df.groupby("gender")[TARGET_COLUMN].agg(total="count", diabetic="sum")
    )
    gender_prevalence["prevalence_pct"] = (
        gender_prevalence["diabetic"] / gender_prevalence["total"]
    ) * 100
    findings["gender_prevalence"] = gender_prevalence.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Gender prevalence barplot
    sns.barplot(
        x=gender_prevalence.index,
        y=gender_prevalence["prevalence_pct"],
        palette="Blues_r",
        ax=axes[0]
    )
    axes[0].set_title("Diabetes Prevalence by Gender (%)")
    axes[0].set_ylabel("Diabetes Rate (%)")
    for bar in axes[0].patches:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold"
        )

    # Age distribution by diabetes status
    sns.kdeplot(
        data=df, x="age", hue=TARGET_COLUMN, common_norm=False,
        palette=["#2b5c8f", "#d9534f"], fill=True, alpha=0.3, ax=axes[1]
    )
    axes[1].set_title("Age Density Distribution by Diabetes Status")
    axes[1].set_xlabel("Age (Years)")
    axes[1].legend(["Diabetic", "Non-Diabetic"], title="Status")
    plt.tight_layout()
    chart2_path = output_dir / "eda_demographics.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    logger.info(f"Saved {chart2_path.name}")

    # 3. Clinical Biomarker Distributions (BMI, HbA1c, Blood Glucose)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    biomarkers = [
        ("bmi", "Body Mass Index (BMI, kg/m²)", "#2ca02c"),
        ("HbA1c_level", "HbA1c Level (%)", "#ff7f0e"),
        ("blood_glucose_level", "Blood Glucose Level (mg/dL)", "#d62728")
    ]
    for idx, (col, title, color) in enumerate(biomarkers):
        sns.histplot(
            data=df, x=col, hue=TARGET_COLUMN, kde=True,
            palette=["#2b5c8f", "#d9534f"], ax=axes[idx], bins=30, common_norm=False, stat="density"
        )
        axes[idx].set_title(f"{title} vs Diabetes")
        axes[idx].set_xlabel(title)
        axes[idx].legend(["Diabetic", "Non-Diabetic"], title="Status")
    plt.tight_layout()
    chart3_path = output_dir / "eda_clinical_biomarkers.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    logger.info(f"Saved {chart3_path.name}")

    # 4. Outlier & Boxplot Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (col, title, _) in enumerate(biomarkers):
        sns.boxplot(
            data=df, x=TARGET_COLUMN, y=col,
            palette=["#2b5c8f", "#d9534f"], ax=axes[idx], showmeans=True
        )
        axes[idx].set_title(f"Outlier Spread: {col} by Status")
        axes[idx].set_xticklabels(["Non-Diabetic", "Diabetic"])
    plt.tight_layout()
    chart4_path = output_dir / "eda_outlier_boxplots.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    logger.info(f"Saved {chart4_path.name}")

    # 5. Correlation Heatmap (Encoding categorical & binary features for full correlation)
    corr_df = df.copy()
    # Map binary/categorical for correlation inspection
    corr_df["gender_code"] = corr_df["gender"].map({"Male": 1, "Female": 0, "Other": 2}).fillna(0)
    corr_cols = NUMERICAL_FEATURES + BINARY_FEATURES + ["gender_code", TARGET_COLUMN]
    corr_matrix = corr_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix (Pearson)")
    plt.tight_layout()
    chart5_path = output_dir / "eda_correlation_matrix.png"
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    logger.info(f"Saved {chart5_path.name}")

    # 6. Categorical Risk Analysis: Smoking History & Pre-existing Conditions
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    smoking_risk = (
        df.groupby("smoking_history")[TARGET_COLUMN].agg(total="count", diabetic="sum")
    )
    smoking_risk["rate_pct"] = (smoking_risk["diabetic"] / smoking_risk["total"]) * 100
    smoking_risk = smoking_risk.sort_values(by="rate_pct", ascending=False)
    
    sns.barplot(
        x=smoking_risk.index, y=smoking_risk["rate_pct"], palette="Reds_r", ax=axes[0]
    )
    axes[0].set_title("Diabetes Prevalence by Smoking History (%)")
    axes[0].set_ylabel("Diabetes Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)
    for bar in axes[0].patches:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold"
        )

    # Hypertension & Heart Disease Interaction
    df["cardio_comorbidity"] = df.apply(
        lambda r: f"HTN:{int(r['hypertension'])}, HD:{int(r['heart_disease'])}", axis=1
    )
    comorb_risk = (
        df.groupby("cardio_comorbidity")[TARGET_COLUMN].agg(total="count", diabetic="sum")
    )
    comorb_risk["rate_pct"] = (comorb_risk["diabetic"] / comorb_risk["total"]) * 100
    comorb_risk = comorb_risk.sort_values(by="rate_pct", ascending=False)
    
    sns.barplot(
        x=comorb_risk.index, y=comorb_risk["rate_pct"], palette="Oranges_r", ax=axes[1]
    )
    axes[1].set_title("Prevalence by Comorbidities (HTN = Hypertension, HD = Heart Disease)")
    axes[1].set_ylabel("Diabetes Rate (%)")
    for bar in axes[1].patches:
        height = bar.get_height()
        axes[1].annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold"
        )
    plt.tight_layout()
    chart6_path = output_dir / "eda_risk_factors.png"
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    logger.info(f"Saved {chart6_path.name}")

    # Summary Statistics for Continuous Biomarkers
    stats_summary = df[NUMERICAL_FEATURES].describe().round(2).to_dict()
    findings["biomarker_statistics"] = stats_summary

    # Save structured findings
    summary_path = output_dir / "eda_findings_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=4)
    logger.info(f"EDA findings written to {summary_path}")

    return findings


if __name__ == "__main__":
    logger.info("Executing Phase 4 Exploratory Data Analysis Pipeline...")
    res = run_full_eda()
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS COMPLETE")
    print("="*60)
    print("Class Distribution:", res["class_distribution"])
    print("Charts successfully saved to:", EVALUATION_DIR.resolve())
    print("="*60 + "\n")
