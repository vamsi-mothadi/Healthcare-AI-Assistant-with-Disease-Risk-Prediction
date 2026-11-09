"""
Text Splitting and Chunking Module for RAG.
Implements recursive character chunking with configurable overlap, section-aware splitting,
and metadata preservation for citations.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import logger


class RecursiveClinicalTextSplitter:
    """
    Splits clinical documents into semantically coherent chunks using a hierarchy of separators:
    1. Markdown section headers (`\n\n## `)
    2. Double newlines (paragraphs, `\n\n`)
    3. Single newlines (`\n`)
    4. Sentence endings (`. `, `? `, `! `)
    5. Spaces (` `)
    
    Preserves document metadata (title, URL, publisher) across every chunk.
    """
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n## ", "\n\n", "\n", ". ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively breaks text using the first matching separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator)
        good_splits: List[str] = []

        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if len(s_clean) <= self.chunk_size:
                good_splits.append(s_clean)
            else:
                if new_separators:
                    sub_splits = self._split_text(s_clean, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Character slice fallback
                    for start in range(0, len(s_clean), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s_clean[start:start + self.chunk_size])

        # Merge splits with overlap
        merged: List[str] = []
        current_chunk = ""

        for s in good_splits:
            if not current_chunk:
                current_chunk = s[:self.chunk_size]
            elif len(current_chunk) + len(separator) + len(s) <= self.chunk_size:
                current_chunk += separator + s
            else:
                merged.append(current_chunk)
                # Overlap: keep tail of current_chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap = current_chunk[-self.chunk_overlap:]
                    # Combine overlap with next piece without exceeding chunk_size
                    combined = (overlap + " " + s).strip()
                    current_chunk = combined[:self.chunk_size]
                else:
                    current_chunk = s[:self.chunk_size]

        if current_chunk:
            merged.append(current_chunk)

        return merged

    def split_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single document dictionary into a list of chunk dictionaries with metadata.
        
        Args:
            document: Dict with 'content', 'source', 'title', 'url', etc.
            
        Returns:
            List of chunk dicts.
        """
        raw_text = document.get("content", "")
        if not raw_text.strip():
            return []

        # Extract metadata from header if present
        metadata = {
            "source_file": document.get("source", "unknown"),
            "source_path": document.get("path", ""),
            "title": document.get("title", document.get("source", "Unknown Document")),
            "url": document.get("url", ""),
            "publisher": document.get("publisher", "Official Clinical Source"),
        }

        # Parse header lines for title and URL if embedded in text
        for line in raw_text.splitlines()[:6]:
            if line.startswith("# DOCUMENT:"):
                metadata["title"] = line.replace("# DOCUMENT:", "").strip()
            elif line.startswith("# SOURCE_URL:"):
                metadata["url"] = line.replace("# SOURCE_URL:", "").strip()
            elif line.startswith("# PUBLISHER:"):
                metadata["publisher"] = line.replace("# PUBLISHER:", "").strip()

        # Remove header separator before chunking
        body_text = re.sub(r"^#.*?\n={40,}\n+", "", raw_text, flags=re.DOTALL).strip()
        if not body_text:
            body_text = raw_text

        text_chunks = self._split_text(body_text, self.separators)
        chunk_dicts = []

        for idx, chunk in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["chunk_id"] = f"{metadata['source_file']}_chunk_{idx}"
            chunk_metadata["char_count"] = len(chunk)
            
            chunk_dicts.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

        logger.info(f"Split {metadata['source_file']} ({len(raw_text)} chars) into {len(chunk_dicts)} chunks.")
        return chunk_dicts

    def split_all_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits an entire collection of documents."""
        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        logger.info(f"Total chunks created across {len(documents)} documents: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    from src.rag.document_loader import load_all_local_documents
    from config.settings import DOCUMENTS_DIR

    logger.info("Testing RecursiveClinicalTextSplitter...")
    docs = load_all_local_documents(DOCUMENTS_DIR)
    splitter = RecursiveClinicalTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_all_documents(docs)

    print("\n" + "="*60)
    print("TEXT CHUNKING VERIFICATION")
    print("="*60)
    print(f"Total Usable Documents Loaded: {len(docs)}")
    print(f"Total Chunks Generated:        {len(chunks)}")
    if chunks:
        sample = chunks[0]
        print(f"\nSample Chunk #0 Metadata:")
        print(f"  Title: {sample['metadata']['title']}")
        print(f"  URL:   {sample['metadata']['url']}")
        print(f"  Length: {sample['metadata']['char_count']} characters")
        print(f"\nSample Chunk Text:\n{sample['text'][:300]}...")
    print("="*60 + "\n")
