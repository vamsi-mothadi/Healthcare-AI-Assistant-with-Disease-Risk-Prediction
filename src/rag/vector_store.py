"""
FAISS Vector Database Management Module.
Implements vector index creation, persistence, loading, and Cosine Similarity nearest-neighbor retrieval
for healthcare knowledge chunks.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np

from config.settings import VECTOR_STORE_DIR, FAISS_INDEX_DIR, RETRIEVAL_TOP_K
from src.utils.logger import logger
from src.rag.embeddings import ClinicalEmbeddingGenerator


class FAISSVectorStore:
    """
    Manages local FAISS vector indexing and disk persistence.
    Uses faiss.IndexFlatIP (Inner Product over normalized unit vectors = Cosine Similarity).
    """
    def __init__(
        self,
        index_dir: Path = FAISS_INDEX_DIR,
        embedder: Optional[ClinicalEmbeddingGenerator] = None
    ):
        self.index_dir = index_dir
        self.index_file = self.index_dir / "index.faiss"
        self.metadata_file = self.index_dir / "chunks_metadata.json"
        self.embedder = embedder or ClinicalEmbeddingGenerator()
        self.dimension = self.embedder.dimension

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def build_and_save_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds a list of text chunks, builds the FAISS index, and persists to disk.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index from empty chunk list.")

        logger.info(f"Extracting texts for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]

        logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts)

        logger.info(f"Building FAISS IndexFlatIP (Cosine Similarity) with dimension {self.dimension}...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks_metadata = chunks

        logger.info(f"FAISS index built. Total indexed vectors: {self.index.ntotal}")

        # Persist to disk
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file.resolve()))

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2)

        logger.info(f"Persisted FAISS index to {self.index_file} and metadata to {self.metadata_file}")

    def load_index(self) -> bool:
        """
        Loads persisted FAISS index and chunk metadata from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not self.index_file.exists() or not self.metadata_file.exists():
            logger.warning(f"FAISS index files not found in {self.index_dir}")
            return False

        try:
            logger.info(f"Loading FAISS index from {self.index_file}...")
            self.index = faiss.read_index(str(self.index_file.resolve()))

            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)

            logger.info(f"FAISS index successfully loaded ({self.index.ntotal} vectors).")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {self.index_dir}: {e}")
            return False

    def similarity_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Searches the FAISS vector index for chunks semantically closest to the query.
        
        Args:
            query: User's healthcare question.
            top_k: Number of nearest chunks to retrieve.
            score_threshold: Minimum cosine similarity score (0.0 to 1.0).
            
        Returns:
            List of matching chunk dicts with 'similarity_score', 'text', and 'metadata'.
        """
        if self.index is None:
            loaded = self.load_index()
            if not loaded:
                raise RuntimeError("FAISS index is not initialized or persisted. Build the index first.")

        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        k = min(top_k, self.index.ntotal)

        # Search index
        scores, indices = self.index.search(query_vec, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1:
                continue
            if score < score_threshold:
                continue

            chunk = self.chunks_metadata[idx]
            results.append({
                "rank": rank + 1,
                "similarity_score": round(float(score), 4),
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            })

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...' (Top score: {scores[0]:.4f})")
        return results


def build_knowledge_base_vector_store() -> FAISSVectorStore:
    """Helper function to load docs, chunk them, and build the persistent index."""
    from src.rag.document_loader import load_all_local_documents
    from src.rag.text_splitter import RecursiveClinicalTextSplitter
    from config.settings import DOCUMENTS_DIR

    logger.info("Initializing complete Knowledge Base Vector Indexing...")
    docs = load_all_local_documents(DOCUMENTS_DIR)
    if not docs:
        raise ValueError(f"No usable documents found in {DOCUMENTS_DIR}. Run document_loader first.")

    splitter = RecursiveClinicalTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_all_documents(docs)

    store = FAISSVectorStore()
    store.build_and_save_index(chunks)
    return store


if __name__ == "__main__":
    logger.info("Testing FAISS Vector Store Module...")
    store = build_knowledge_base_vector_store()
    
    test_query = "What are the fasting blood glucose thresholds for diagnosing diabetes?"
    results = store.similarity_search(test_query, top_k=3)
    
    print("\n" + "="*60)
    print("FAISS SIMILARITY SEARCH VERIFICATION")
    print("="*60)
    print(f"Query: '{test_query}'\n")
    for r in results:
        print(f"[Rank {r['rank']}] Score: {r['similarity_score']:.4f}")
        print(f"Source: {r['metadata'].get('title')} ({r['metadata'].get('url')})")
        print(f"Snippet: {r['text'][:250]}...\n" + "-"*60)
    print("="*60 + "\n")

