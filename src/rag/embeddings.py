"""
Local Vector Embedding Module for RAG.
Uses Sentence Transformers (all-MiniLM-L6-v2) to generate 384-dimensional dense embeddings
locally on CPU with L2 normalization for fast cosine similarity search.
"""

import sys
from pathlib import Path
from typing import List, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL_NAME
from src.utils.logger import logger


class ClinicalEmbeddingGenerator:
    """
    Encapsulates SentenceTransformer model for generating dense semantic vector embeddings.
    Runs 100% locally on CPU without external API fees or network latency.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._dimension = 384 # all-MiniLM-L6-v2 default
        self._load_model()

    def _load_model(self) -> None:
        """Loads SentenceTransformer model weights (preferring local cache)."""
        logger.info(f"Loading local embedding model: '{self.model_name}'...")
        try:
            # Try offline local cache first for zero network latency
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            self._dimension = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)()
            logger.info(f"Embedding model loaded from local cache. Vector dimension: {self._dimension}")
        except Exception:
            try:
                self._model = SentenceTransformer(self.model_name)
                self._dimension = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)()
                logger.info(f"Embedding model downloaded and loaded. Vector dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise

    @property
    def dimension(self) -> int:
        """Returns vector dimensionality (e.g., 384)."""
        return self._dimension

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized dense vector embeddings for a batch of text chunks.
        
        Args:
            texts: List of text strings.
            batch_size: Batch size for CPU inference.
            
        Returns:
            np.ndarray: Matrix of shape (N, dimension) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True # Normalization ensures dot product = cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates normalized vector embedding for a single user query.
        
        Args:
            query: User question string.
            
        Returns:
            np.ndarray: Vector of shape (dimension,) with float32 dtype.
        """
        cleaned_query = query.strip()
        embedding = self._model.encode(
            cleaned_query,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embedding, dtype=np.float32)


if __name__ == "__main__":
    logger.info("Testing ClinicalEmbeddingGenerator...")
    embedder = ClinicalEmbeddingGenerator()
    sample_texts = [
        "What are the diagnostic blood glucose levels for type 2 diabetes?",
        "High blood sugar and insulin resistance can cause diabetic symptoms."
    ]
    vecs = embedder.embed_texts(sample_texts)
    print("\n" + "="*60)
    print("EMBEDDING GENERATOR VERIFICATION")
    print("="*60)
    print(f"Model Name:        {embedder.model_name}")
    print(f"Vector Dimension:  {embedder.dimension}")
    print(f"Embeddings Shape:  {vecs.shape}")
    print(f"Vector Dtype:      {vecs.dtype}")
    # Compute cosine similarity between sample 0 and 1
    cosine_sim = np.dot(vecs[0], vecs[1])
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    print("Normalized L2 Norm of vector 0:", np.linalg.norm(vecs[0]))
    print("="*60 + "\n")
