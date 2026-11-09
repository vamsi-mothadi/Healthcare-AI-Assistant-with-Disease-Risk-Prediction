# Machine Learning Workflow

## End-to-End Pipeline for Diabetes Risk Prediction

This document describes the design, training strategy, validation protocol, and deployment lifecycle of the Machine Learning classification pipeline.

---

## 1. Machine Learning Lifecycle Diagram

```mermaid
flowchart TD
    A["Raw Dataset (data/raw/diabetes_prediction_dataset.csv)"] --> B["Data Ingestion & Schema Validation"]
    B --> C["Exploratory Data Analysis (EDA) & Outlier Audit"]
    C --> D["Stratified Train-Test Split (80/20, Seed=42)"]
    
    subgraph PreprocessingPipeline ["Scikit-Learn ColumnTransformer Pipeline"]
        direction TB
        NumPipe["Numerical Features (Age, BMI, HbA1c, Glucose)<br/>SimpleImputer(median) -> StandardScaler"]
        CatPipe["Categorical Features (Gender, Smoking History)<br/>SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')"]
    end

    D --> |Train Split Only| PreprocessingPipeline
    PreprocessingPipeline --> E["Class Imbalance Handling (Class Weighting / Resampling)"]
    E --> F["Model Training & Cross-Validation"]
    
    subgraph CandidateModels ["Evaluated Architectures"]
        M1["Logistic Regression (L2 regularized, class_weight='balanced')"]
        M2["Random Forest Classifier (Ensemble of Decision Trees)"]
    end
    
    F --> CandidateModels
    CandidateModels --> G["Hyperparameter Optimization (GridSearchCV on Train Set)"]
    G --> H["Model Evaluation on Untouched Test Set"]
    
    subgraph MetricsAudit ["Evaluation Metrics"]
        direction TB
        M_Overall["Overall Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC"]
        M_Demographic["Subgroup Fairness: Male vs Female Performance Metrics"]
        M_Calib["Probability Calibration Assessment"]
    end

    H --> MetricsAudit
    MetricsAudit --> I["Model Selection & Final Pipeline Serialization (Joblib)"]
    I --> J["Saved Model: models/diabetes_pipeline.joblib"]
```

---

## 2. Key Methodological Decisions

### 2.1 Preventing Data Leakage
- Preprocessing steps (imputation statistics, scaling means/standard deviations, and categorical encoder vocabularies) are **fit solely on the training partition**.
- The testing split is transformed using the fitted pipeline without re-estimating any parameters.

### 2.2 Class Imbalance Handling
- Medical datasets frequently suffer from class imbalance (e.g., ~8-10% positive diabetic cases).
- Optimizing strictly for raw accuracy yields misleading results (a model predicting everyone as non-diabetic could achieve 90% accuracy while failing 100% of diabetic patients).
- We prioritize **Recall** (Sensitivity) and **F1-Score / ROC-AUC**, employing cost-sensitive learning (`class_weight='balanced'`) to penalize false negatives severely.

### 2.3 Subgroup & Fairness Auditing
- Models can exhibit disparate error rates across demographic cohorts (e.g., differing sensitivity in men vs women).
- We systematically evaluate test performance sliced by gender to document model reliability across all user groups.

