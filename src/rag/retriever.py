"""
Healthcare Knowledge Retriever Module.
Orchestrates semantic similarity search, score thresholding, chunk deduplication,
and context block assembly for LLM generation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RETRIEVAL_TOP_K
from src.utils.logger import logger
from src.rag.vector_store import FAISSVectorStore


class HealthcareRetriever:
    """
    Retrieves and prepares clinically verified document context for the Gemini RAG pipeline.
    """
    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        self.vector_store = vector_store or FAISSVectorStore()
        if not self.vector_store.load_index():
            logger.info("Vector store not loaded from disk. Building index...")
            from src.rag.vector_store import build_knowledge_base_vector_store
            self.vector_store = build_knowledge_base_vector_store()

    def retrieve_context(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = 0.25
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves top relevant chunks and formats them into an untrusted context block
        for the LLM, returning both the structured text block and citation sources.
        
        Args:
            query: User's question.
            top_k: Maximum chunks to retrieve.
            score_threshold: Minimum cosine similarity.
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: (formatted_context_block, citation_list)
        """
        results = self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not results:
            logger.warning(f"No documents retrieved above similarity threshold {score_threshold} for query: {query}")
            return "", []

        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []

        for r in results:
            meta = r.get("metadata", {})
            title = meta.get("title", "Clinical Publication")
            url = meta.get("url", "Official Source")
            publisher = meta.get("publisher", "Official Health Agency")
            score = r.get("similarity_score", 0.0)
            text = r.get("text", "")

            # Formulate structured block for LLM
            block = (
                f"[DOCUMENT #{r['rank']} | {title} | Source: {url} | Score: {score:.3f}]\n"
                f"{text}\n"
            )
            context_blocks.append(block)

            citations.append({
                "rank": r["rank"],
                "title": title,
                "url": url,
                "publisher": publisher,
                "similarity_score": score,
                "snippet": text[:200] + "..." if len(text) > 200 else text
            })

        formatted_context = "\n---\n".join(context_blocks)
        return formatted_context, citations


if __name__ == "__main__":
    logger.info("Testing HealthcareRetriever...")
    retriever = HealthcareRetriever()
    ctx, cites = retriever.retrieve_context("What is prediabetes and what are its symptoms?", top_k=2)
    print("\n" + "="*60)
    print("RETRIEVER VERIFICATION")
    print("="*60)
    print("Formatted Context Block:\n", ctx[:400], "...\n")
    print(f"Citations count: {len(cites)}")
    for c in cites:
        print(f"- {c['title']} ({c['similarity_score']}): {c['url']}")
    print("="*60 + "\n")

