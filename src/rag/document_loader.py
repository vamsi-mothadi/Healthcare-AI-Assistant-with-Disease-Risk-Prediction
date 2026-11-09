"""
Healthcare Document Ingestion and Preprocessing Module for RAG.
Collects, extracts, cleans, and catalogs official clinical publications from trusted sources
(NIH/NIDDK, WHO, CDC) to power the Healthcare AI Assistant's knowledge base.
"""

import json
import logging
import re
import sys
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DOCUMENTS_DIR, DOCUMENTS_INVENTORY_PATH
from src.utils.logger import logger


class WebContentCleaner(HTMLParser):
    """
    Strips non-content HTML elements (scripts, styles, navigations, footers, etc.)
    and converts headings and paragraphs into clean, readable Markdown structure.
    """
    def __init__(self):
        super().__init__()
        self.text_blocks: List[str] = []
        self.in_ignored_tag: int = 0
        self.ignored_tags = {
            "script", "style", "nav", "header", "footer", "noscript",
            "svg", "aside", "form", "button", "iframe"
        }
        self.current_tag: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[tuple]):
        self.current_tag = tag.lower()
        if self.current_tag in self.ignored_tags:
            self.in_ignored_tag += 1
        elif self.current_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_blocks.append("\n\n## ")
        elif self.current_tag in {"p", "li", "div", "tr", "section", "article"}:
            self.text_blocks.append("\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self.ignored_tags:
            self.in_ignored_tag = max(0, self.in_ignored_tag - 1)
        elif t in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}:
            self.text_blocks.append("\n")

    def handle_data(self, data: str):
        if self.in_ignored_tag == 0:
            cleaned = data.strip()
            if cleaned:
                self.text_blocks.append(cleaned + " ")

    def get_clean_text(self) -> str:
        raw = "".join(self.text_blocks)
        # Normalize multiple spaces and multiple newlines
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


# Official Clinical Knowledge Registry
OFFICIAL_SOURCES = [
    {
        "id": "niddk_overview",
        "title": "NIDDK Diabetes Overview",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview",
        "filename": "niddk_diabetes_overview.txt",
        "publisher": "National Institute of Diabetes and Digestive and Kidney Diseases (NIH)"
    },
    {
        "id": "niddk_tests_diagnosis",
        "title": "NIDDK Diabetes Tests and Diagnosis",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis",
        "filename": "niddk_diabetes_tests_diagnosis.txt",
        "publisher": "National Institute of Diabetes and Digestive and Kidney Diseases (NIH)"
    },
    {
        "id": "who_diabetes_factsheet",
        "title": "WHO Diabetes Information Fact Sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "filename": "who_diabetes_factsheet.txt",
        "publisher": "World Health Organization (WHO)"
    },
    {
        "id": "niddk_clinical_tools",
        "title": "NIDDK Diabetes & Prediabetes Clinical Management",
        "url": "https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/diabetes/diabetes-prediabetes",
        "filename": "niddk_diabetes_prediabetes_clinical.txt",
        "publisher": "National Institute of Diabetes and Digestive and Kidney Diseases (NIH)"
    },
    {
        "id": "cdc_prevention",
        "title": "CDC Type 2 Diabetes Prevention",
        "url": "https://www.cdc.gov/diabetes/prevention-type-2/index.html",
        "filename": "cdc_type2_diabetes_prevention.txt",
        "publisher": "Centers for Disease Control and Prevention (CDC)"
    }
]


def fetch_and_clean_webpage(url: str, timeout: int = 15) -> Tuple[bool, str, Optional[str]]:
    """
    Downloads webpage HTML and extracts structured, cleaned clinical text.
    
    Args:
        url: Webpage URL.
        timeout: Network timeout in seconds.
        
    Returns:
        Tuple: (success_bool, extracted_text_or_empty, error_message_or_None)
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_html = response.read().decode("utf-8", errors="ignore")
        
        parser = WebContentCleaner()
        parser.feed(raw_html)
        text = parser.get_clean_text()
        return True, text, None
    except Exception as e:
        return False, "", str(e)


def collect_official_documents(
    sources: List[Dict[str, str]] = OFFICIAL_SOURCES,
    docs_dir: Path = DOCUMENTS_DIR,
    inventory_path: Path = DOCUMENTS_INVENTORY_PATH
) -> List[Dict[str, Any]]:
    """
    Orchestrates the ingestion, extraction, and cataloging of healthcare documents.
    
    Args:
        sources: Registry list of official sources.
        docs_dir: Directory where extracted files are saved.
        inventory_path: Destination path for documents inventory JSON.
        
    Returns:
        List[Dict[str, Any]]: Catalog inventory of all documents.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    inventory: List[Dict[str, Any]] = []

    for source in sources:
        doc_path = docs_dir / source["filename"]
        doc_id = source["id"]
        title = source["title"]
        url = source["url"]
        publisher = source["publisher"]

        # Check if already collected and non-empty
        if doc_path.exists() and doc_path.stat().st_size > 500:
            logger.info(f"Document already exists locally: {source['filename']}")
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            inventory.append({
                "document_id": doc_id,
                "document_name": title,
                "source_url": url,
                "publisher": publisher,
                "file_format": "txt",
                "file_path": str(doc_path.resolve()),
                "extraction_status": "Success (Cached)",
                "character_count": len(content),
                "last_retrieval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            continue

        logger.info(f"Fetching official document: '{title}' from {url}...")
        success, text, error = fetch_and_clean_webpage(url)

        if success and len(text) > 300:
            header_metadata = (
                f"# DOCUMENT: {title}\n"
                f"# SOURCE_URL: {url}\n"
                f"# PUBLISHER: {publisher}\n"
                f"# INGESTION_DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*80}\n\n"
            )
            full_document = header_metadata + text
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(full_document)
            
            logger.info(f"Successfully saved {source['filename']} ({len(full_document)} characters).")
            inventory.append({
                "document_id": doc_id,
                "document_name": title,
                "source_url": url,
                "publisher": publisher,
                "file_format": "txt",
                "file_path": str(doc_path.resolve()),
                "extraction_status": "Success",
                "character_count": len(full_document),
                "last_retrieval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            logger.warning(
                f"Automated extraction failed for '{title}' ({url}): {error}\n"
                f"Manual Fallback Instruction:\n"
                f"1. Open the URL in your web browser: {url}\n"
                f"2. Copy the text content (or save page as text).\n"
                f"3. Save the content to: '{doc_path.resolve()}'."
            )
            # Create a placeholder guide file if it doesn't exist
            if not doc_path.exists():
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"# DOCUMENT: {title}\n"
                        f"# SOURCE_URL: {url}\n"
                        f"# STATUS: Automated extraction blocked by host (HTTP 403 / anti-bot).\n"
                        f"# MANUAL_INSTRUCTION: Copy the official clinical text from the source URL above and paste it here.\n"
                    )
            inventory.append({
                "document_id": doc_id,
                "document_name": title,
                "source_url": url,
                "publisher": publisher,
                "file_format": "txt",
                "file_path": str(doc_path.resolve()),
                "extraction_status": f"Pending Manual Ingestion ({error or 'Insufficient text'})",
                "character_count": doc_path.stat().st_size if doc_path.exists() else 0,
                "last_retrieval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Save documents inventory
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=4)
    logger.info(f"Document inventory written to {inventory_path}")
    return inventory


def load_all_local_documents(docs_dir: Path = DOCUMENTS_DIR) -> List[Dict[str, Any]]:
    """
    Loads all validated local text and PDF documents from knowledge_base/documents/.
    
    Args:
        docs_dir: Directory containing documents.
        
    Returns:
        List of dicts containing parsed content and metadata for RAG chunking.
    """
    loaded_docs = []
    if not docs_dir.exists():
        logger.warning(f"Documents directory not found: {docs_dir}")
        return loaded_docs

    for file_path in docs_dir.glob("*.*"):
        if file_path.suffix.lower() == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Skip placeholder files with no real content
                if "STATUS: Automated extraction blocked" in content and len(content) < 500:
                    continue
                loaded_docs.append({
                    "source": str(file_path.name),
                    "path": str(file_path.resolve()),
                    "content": content,
                    "type": "txt",
                    "char_count": len(content)
                })
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        elif file_path.suffix.lower() == ".pdf":
            # PDF loading support via pypdf when installed
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                pdf_text = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text.append(f"\n--- Page {idx + 1} ---\n" + page_text)
                combined = "".join(pdf_text)
                if combined.strip():
                    loaded_docs.append({
                        "source": str(file_path.name),
                        "path": str(file_path.resolve()),
                        "content": combined,
                        "type": "pdf",
                        "char_count": len(combined)
                    })
            except ImportError:
                logger.warning(f"pypdf not installed. Skipping PDF parsing for {file_path.name}.")
            except Exception as e:
                logger.error(f"Error parsing PDF {file_path}: {e}")

    logger.info(f"Loaded {len(loaded_docs)} usable documents from {docs_dir}.")
    return loaded_docs


if __name__ == "__main__":
    logger.info("Executing Phase 2 Document Collection...")
    inv = collect_official_documents()
    print("\n" + "="*60)
    print("HEALTHCARE DOCUMENTS INVENTORY")
    print("="*60)
    for doc in inv:
        print(f"Title: {doc['document_name']}")
        print(f"Status: {doc['extraction_status']}")
        print(f"Characters: {doc['character_count']:,}")
        print(f"File: {Path(doc['file_path']).name}")
        print("-" * 60)
    print("="*60 + "\n")
