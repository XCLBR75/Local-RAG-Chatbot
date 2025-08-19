import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import requests
import inspect
import logging

# Import from new modules
from .models.pdf_models import ParsePDFRequest, ParsePDFResponse, SearchPDFRequest, SearchPDFResponse, SearchResult, SummarizePDFRequest, SummarizePDFResponse
from .utils.pdf_utils import extract_pdf_chunks

logging.basicConfig(level=logging.DEBUG)

mcp = FastMCP(name="pdf_parser")


@mcp.tool()
def parse_pdf(request: ParsePDFRequest) -> ParsePDFResponse:
    chunks = extract_pdf_chunks(request.file_path)
    return ParsePDFResponse(file=request.file_path, chunks=chunks)

@mcp.tool()
def search_pdf(request: SearchPDFRequest) -> SearchPDFResponse:
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", request.file_path)
    file_path = os.path.abspath(file_path)

    chunks = extract_pdf_chunks(request.file_path)
    matches = []
    # Defensive: ensure query is a string and not None
    query = str(getattr(request, 'query', ''))
    query_lower = query.lower()

    # Debug logging
    print("\n=== DEBUG: search_pdf called ===")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print("Function arguments:")
    for arg in args:
        print(f"  {arg}: {type(values[arg])} = {values[arg]}")
    
    if hasattr(request, '__dict__'):
        print(f"Request attributes: {request.__dict__}")
    else:
        print(f"Raw request: {request}")
    
    print("Call stack:")
    for frame in inspect.stack()[1:]:  # Skip current frame
        print(f"  File {frame.filename}, line {frame.lineno}, in {frame.function}")
    print("=== END DEBUG ===\n")

    for chunk in chunks:
        text = str(chunk.text)

        if query_lower and query_lower in text.lower():
            snippet_start = max(0, text.lower().index(query_lower) - 30)
            snippet_end = min(len(text), snippet_start + 100)
            snippet = text[snippet_start:snippet_end]
            matches.append(SearchResult(page=chunk.page, snippet=snippet))

    return SearchPDFResponse(matches=matches)


@mcp.tool()
def summarize_pdf(request: SummarizePDFRequest) -> SummarizePDFResponse:
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", request.file_path)
    file_path = os.path.abspath(file_path)

    chunks = extract_pdf_chunks(file_path)

    all_text = " ".join(str(chunk.text) for chunk in chunks)

    sentences = all_text.split(". ")
    summary = ". ".join(sentences[:int(request.max_sentences)]) + "."

    return SummarizePDFResponse(summary=summary)



# -----------------
# TAVILY TOOL
# -----------------
from pydantic import BaseModel
from typing import Dict, Any

class TavilyRequest(BaseModel):
    query: str

class TavilyResponse(BaseModel):
    results: Dict[str, Any]

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

@mcp.tool()
def tavily_search(request: TavilyRequest) -> TavilyResponse:
    """Search the web via Tavily API."""
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": request.query}

    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()

    return TavilyResponse(results=resp.json())

# -----------------
# ENTRYPOINT
# -----------------
if __name__ == "__main__":
    mcp.run()