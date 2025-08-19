# Run with: python -m uvicorn pdf_mcp_server.pdf_http_server:app --host 127.0.0.1 --port 8000
from fastapi import FastAPI, HTTPException
from typing import List

from pydantic import BaseModel

from .utils.tavily_utils import tavily_search

from .models.pdf_models import ChunkOut, ParseRequest, ParseResponse, SearchRequest, SearchResponse, SummarizeRequest, SummarizeResponse, TavilySearchRequest, TavilySearchResponse, TavilySearchResult  
from .utils.pdf_utils import extract_pdf_chunks, summarize, search_pdf_with_snippets

app = FastAPI(title="PDF parser HTTP wrapper")




@app.post("/search_pdf", response_model=SearchResponse)
def search_pdf(req: SearchRequest):
    try:
        matches = search_pdf_with_snippets(req.file_path, req.query)
        return SearchResponse(file=req.file_path, query=req.query, matches=matches)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_pdf", response_model=SummarizeResponse)
def summarize_pdf(req: SummarizeRequest):
    try:
        summary = summarize(req.file_path, req.max_sentences)
        return SummarizeResponse(file=req.file_path, summary=summary)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse_pdf", response_model=ParseResponse)
def parse_pdf(req: ParseRequest):
    try:
        chunks = extract_pdf_chunks(req.file_path)
        return ParseResponse(
            file=req.file_path,
            chunks=[ChunkOut(page=c.page, text=c.text) for c in chunks]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tavily_search", response_model=TavilySearchResponse)
async def search_tavily_endpoint(req: TavilySearchRequest):
    try:
        results = await tavily_search(req.query, req.max_results)
        return TavilySearchResponse(
            query=req.query,
            results=[
                TavilySearchResult(
                    title=r.get("title", ""),
                    content=r.get("content", ""),
                    url=r.get("url", "")
                ) for r in results
            ]
        )
    except Exception as e:
        import traceback; traceback.print_exc()  
        raise HTTPException(status_code=500, detail=str(e))