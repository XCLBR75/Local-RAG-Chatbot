from pathlib import Path
import fitz
from typing import List
from ..models.pdf_models import PDFChunk, SearchResultOut


def extract_pdf_chunks(file_path: str) -> List[PDFChunk]:
    """Extract raw text chunks (per page) from a PDF."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pdf = fitz.open(file_path)
    chunks = []
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if text:
            chunks.append(PDFChunk(page=page_num, text=text))
    return chunks


def summarize(file_path: str, max_sentences: int = 5) -> str:
    """Naive summarizer: grab first N sentences across all pages."""
    chunks = extract_pdf_chunks(file_path)
    all_text = " ".join(chunk.text for chunk in chunks)

    # Very naive sentence splitter
    sentences = [s.strip() for s in all_text.split(".") if s.strip()]
    summary = ". ".join(sentences[:max_sentences])
    if summary:
        summary += "."
    return summary


def search(file_path: str, keyword: str) -> List[PDFChunk]:
    """Return PDF chunks (pages) containing the given keyword."""
    keyword_lower = keyword.lower()
    chunks = extract_pdf_chunks(file_path)

    matches = []
    for chunk in chunks:
        if keyword_lower in chunk.text.lower():
            matches.append(chunk)
    return matches


def search_pdf_with_snippets(file_path: str, query: str) -> List[SearchResultOut]:
    """Search PDF and return results with snippets."""
    query_lower = query.lower()
    pdf = fitz.open(file_path)
    results = []

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if text and query_lower in text.lower():
            snippet_start = max(0, text.lower().index(query_lower) - 30)
            snippet_end = min(len(text), snippet_start + 100)
            snippet = text[snippet_start:snippet_end]
            results.append(SearchResultOut(page=page_num, snippet=snippet))

    return results
