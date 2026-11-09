# System Architecture

## Healthcare AI Assistant with Disease Risk Prediction

This document defines the high-level system architecture, design principles, and component interactions for the **Healthcare AI Assistant**.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph UI ["User Interface (Streamlit Dashboard)"]
        UI_Home["Page 1: Home & Overview"]
        UI_Pred["Page 2: Diabetes Risk Prediction Form"]
        UI_Perf["Page 3: Model Performance & Fair Demographics"]
        UI_Chat["Page 4: Healthcare AI RAG Chatbot"]
        UI_About["Page 5: Project Documentation & Architecture"]
    end

    subgraph ModuleA ["Module A: Machine Learning Risk Prediction Engine"]
        Input_Val["Input Validation (Pydantic / Custom)"]
        Preproc["Preprocessing Pipeline (Imputation, OneHot, Scaling)"]
        ML_Model["Trained Classifier (Random Forest / Logistic Regression)"]
        Calibrated_Prob["Calibrated Risk Scoring & Class Output"]
        
        Input_Val --> Preproc --> ML_Model --> Calibrated_Prob
    end

    subgraph ModuleB ["Module B: Retrieval-Augmented Generation (RAG) Engine"]
        KB["Healthcare Knowledge Base (NIDDK, CDC, WHO)"]
        DocLoader["Document Loader (PDF / TXT / Web Scraper)"]
        Chunker["Recursive Text Splitter (with Overlap & Metadata)"]
        Embedder["Local Embeddings (all-MiniLM-L6-v2)"]
        FAISS_DB["FAISS Vector Index (Persistent Local Vector DB)"]
        Retriever["Similarity Search (Top-k Chunks + Score Filter)"]
        SystemPrompt["Strict Medical System Prompt (Safety Guardrails)"]
        GeminiLLM["Gemini LLM (Google AI Studio)"]
        GroundedResp["Context-Grounded Response + Source Citations"]

        KB --> DocLoader --> Chunker --> Embedder --> FAISS_DB
        Retriever <--> FAISS_DB
        Retriever --> SystemPrompt
        SystemPrompt --> GeminiLLM --> GroundedResp
    end

    %% Interactions
    UI_Pred --> Input_Val
    Calibrated_Prob --> UI_Pred

    UI_Chat --> Retriever
    GroundedResp --> UI_Chat

    style UI fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    style ModuleA fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    style ModuleB fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
```

---

## 2. Architectural Pillars

### 2.1 Decoupling and Modularity
The codebase separates concerns into isolated layers:
- `src/data/`: Data loading and validation schemas.
- `src/features/`: Feature engineering and transformations.
- `src/ml/`: Training, evaluation, prediction pipelines, and fairness assessment.
- `src/rag/`: Document ingestion, chunking, vector indexing, retrieval, and LLM communication.
- `src/utils/`: Shared utilities, logging, and validation helpers.
- `app.py`: Streamlit presentation layer strictly calling `src/` modules.

### 2.2 Dual-Engine Design
1. **Deterministic / Statistical Machine Learning (Module A)**:
   - Predicts risk category (`Non-Diabetic` vs `Diabetic`) from patient biometrics.
   - Operates entirely locally with Scikit-learn pipelines, ensuring millisecond inference, zero external API costs, and full interpretability.
2. **Generative RAG Engine (Module B)**:
   - Provides contextual, medical-grade question answering based solely on trusted clinical guidelines (NIH, CDC, WHO).
   - Eliminates hallucination by enforcing context-grounded prompt constraints and displaying source attribution.

### 2.3 Responsible AI & Safety Guardrails
- **Medical Disclaimer**: Clear alerts reminding users that the assistant is an educational tool, not a diagnostic or prescribing physician.
- **Subgroup Parity**: Explicit evaluation across demographic groups (male vs female) to detect and document performance disparities.
- **Untrusted Context Guardrails**: RAG prompts treat retrieved knowledge as reference facts, neutralizing prompt injection attempts embedded in source texts.

