"""
Unit Tests for Healthcare Document Ingestion and Cataloging.
"""

import pytest
from pathlib import Path

from config.settings import DOCUMENTS_DIR, DOCUMENTS_INVENTORY_PATH
from src.rag.document_loader import load_all_local_documents, WebContentCleaner


def test_documents_directory_has_files():
    """Verify that clinical knowledge documents exist in knowledge_base/documents/."""
    assert DOCUMENTS_DIR.exists(), f"Documents directory does not exist: {DOCUMENTS_DIR}"
    docs = list(DOCUMENTS_DIR.glob("*.txt"))
    assert len(docs) >= 4, f"Expected at least 4 clinical documents, found {len(docs)}"


def test_load_all_local_documents():
    """Verify that document loader parses valid text documents with metadata."""
    loaded_docs = load_all_local_documents(DOCUMENTS_DIR)
    assert len(loaded_docs) >= 4

    for doc in loaded_docs:
        assert "source" in doc
        assert "content" in doc
        assert doc["char_count"] > 500
        assert len(doc["content"]) == doc["char_count"]


def test_web_content_cleaner():
    """Verify that HTML cleaner properly converts tags to markdown and ignores scripts."""
    html_sample = """
    <html>
        <head><title>Test</title><script>alert('malicious')</script></head>
        <body>
            <nav><a href='/'>Home</a></nav>
            <h1>Diabetes Overview</h1>
            <p>Blood glucose levels rise when insulin is insufficient.</p>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    parser = WebContentCleaner()
    parser.feed(html_sample)
    clean_text = parser.get_clean_text()

    assert "alert('malicious')" not in clean_text
    assert "Home" not in clean_text
    assert "Copyright" not in clean_text
    assert "Diabetes Overview" in clean_text
    assert "Blood glucose levels rise" in clean_text


def test_documents_inventory_file():
    """Verify that documents inventory JSON exists and lists all documents."""
    assert DOCUMENTS_INVENTORY_PATH.exists()
    import json
    with open(DOCUMENTS_INVENTORY_PATH, "r", encoding="utf-8") as f:
        inv = json.load(f)
    assert isinstance(inv, list)
    assert len(inv) >= 5
    filenames = [Path(item["file_path"]).name for item in inv]
    assert "niddk_diabetes_overview.txt" in filenames
    assert "who_diabetes_factsheet.txt" in filenames

