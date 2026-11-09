"""
Application Configuration and Path Management.
Centralized settings ensuring reproducible paths across ML, RAG, and UI modules.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directory Paths
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATASET_PATH = DATA_RAW_DIR / "diabetes_prediction_dataset.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_diabetes_data.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "diabetes_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DOCUMENTS_DIR = KNOWLEDGE_BASE_DIR / "documents"
DOCUMENTS_INVENTORY_PATH = KNOWLEDGE_BASE_DIR / "documents_inventory.json"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"

LOGS_DIR = BASE_DIR / "logs"
APP_LOG_PATH = LOGS_DIR / "app.log"

# Machine Learning Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "diabetes"

# Numerical & Categorical Features for the Diabetes Dataset
NUMERICAL_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
BINARY_FEATURES = ["hypertension", "heart_disease"]

# RAG & Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 4

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
TEMPERATURE = 0.2

# Application Settings
APP_TITLE = "Healthcare AI Assistant"
APP_ICON = "🩺"
APP_VERSION = "1.0.0"

