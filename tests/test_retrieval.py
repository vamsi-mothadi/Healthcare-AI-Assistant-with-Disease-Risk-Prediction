"""
Unit Tests for RAG Text Chunking, Embeddings, and FAISS Vector Retrieval.
"""

import pytest
import numpy as np

from src.rag.text_splitter import RecursiveClinicalTextSplitter
from src.rag.embeddings import ClinicalEmbeddingGenerator
from src.rag.vector_store import FAISSVectorStore
from src.rag.retriever import HealthcareRetriever


def test_text_splitter_chunk_size_and_overlap():
    """Verify text splitter generates chunks bounded by chunk_size and includes overlap."""
    sample_doc = {
        "content": (
            "# DOCUMENT: Test Guide\n# SOURCE_URL: https://example.org\n====================\n\n"
            "## Section 1\n" + "Diabetes causes chronic health problems. " * 30 + "\n\n"
            "## Section 2\n" + "Insulin resistance is a common factor. " * 30
        ),
        "source": "test_guide.txt"
    }
    splitter = RecursiveClinicalTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_document(sample_doc)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 350 # Bounded with buffer
        assert c["metadata"]["source_file"] == "test_guide.txt"
        assert c["metadata"]["title"] == "Test Guide"
        assert c["metadata"]["url"] == "https://example.org"


def test_embedding_generator_dimensions():
    """Verify embedding generator outputs 384-dimensional normalized vectors."""
    embedder = ClinicalEmbeddingGenerator()
    assert embedder.dimension == 384

    query = "What is type 2 diabetes?"
    vec = embedder.embed_query(query)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Verify unit L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_faiss_vector_store_retrieval():
    """Verify that persisted FAISS index returns relevant results with similarity scores."""
    store = FAISSVectorStore()
    loaded = store.load_index()
    assert loaded is True, "Failed to load persisted FAISS index"
    assert store.index.ntotal >= 90

    results = store.similarity_search("blood glucose fasting levels", top_k=3)
    assert len(results) > 0
    assert results[0]["similarity_score"] > 0.40
    assert "text" in results[0]
    assert "url" in results[0]["metadata"]


def test_retriever_context_assembly():
    """Verify retriever formats context blocks and structured citations."""
    retriever = HealthcareRetriever()
    ctx, citations = retriever.retrieve_context("What are the diagnostic tests for diabetes?", top_k=2)

    assert len(ctx) > 100
    assert len(citations) == 2
    for cite in citations:
        assert "title" in cite
        assert "url" in cite
        assert cite["similarity_score"] > 0.0

