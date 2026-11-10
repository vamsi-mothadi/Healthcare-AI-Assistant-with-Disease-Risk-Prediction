# 🩺 Healthcare AI Assistant: Disease Risk Prediction & Grounded RAG Chatbot

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector--Search-green.svg)](https://github.com/facebookresearch/faiss)
[![Gemini API](https://img.shields.io/badge/Google--Gemini-3.8--Flash-purple.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-29%20Passed-brightgreen.svg)](https://pytest.org/)
[![Demographic Parity](https://img.shields.io/badge/Fairness--Parity-0.966-teal.svg)](docs/ml_workflow.md)

An end-to-end, clinical-grade AI platform integrating predictive machine learning for disease screening with Retrieval-Augmented Generation (RAG) for evidence-grounded patient education.

### 🌟 Key Capabilities:
- **Predictive ML (Disease Screening)**: Random Forest classifier trained on 100,000 clinical records; achieves **91.86% Sensitivity (Recall)**, **88.38% Specificity**, and **0.9719 ROC-AUC** to minimize dangerous false negatives.
- **Fairness & Bias Audit**: Evaluated for demographic equity across patient genders, achieving a certified **0.966 Demographic Parity Ratio** (adhering to the EEOC 4/5ths standard).
- **Clinical RAG Assistant**: Local, privacy-preserving **FAISS `IndexFlatIP`** vector database indexing official clinical guidelines from the **NIH NIDDK**, **CDC**, and **WHO** using `all-MiniLM-L6-v2` embeddings and MMR reranking.
- **Clinical Safety Guardrails**: Grounded generation with Google Gemini incorporating prompt-injection resistance, medical non-diagnosis enforcement, and traceable source citations.
- **Interactive UI**: Multi-page responsive **Streamlit** dashboard with dynamic Plotly risk gauges, diagnostic curves, and conversational chat.
- **Production QA**: Comprehensive test suite featuring **29 automated Pytest unit and integration tests**.
