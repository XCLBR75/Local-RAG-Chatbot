from typing import List
from pydantic import BaseModel

class ParsePDFRequest(BaseModel):
    file_path: str

class PDFChunk(BaseModel):
    page: int
    text: str

class ParsePDFResponse(BaseModel):
    file: str
    chunks: List[PDFChunk]

class SummarizePDFRequest(BaseModel):
    file_path: str
    max_sentences: int = 5

class SummarizePDFResponse(BaseModel):
    summary: str

class SearchPDFRequest(BaseModel):
    file_path: str
    query: str

class SearchResult(BaseModel):
    page: int
    snippet: str

class SearchPDFResponse(BaseModel):
    matches: List[SearchResult]

class ParseRequest(BaseModel):
    file_path: str

class ChunkOut(BaseModel):
    page: int
    text: str

class ParseResponse(BaseModel):
    file: str
    chunks: List[ChunkOut]

class SummarizeRequest(BaseModel):
    file_path: str
    max_sentences: int = 5

class SummarizeResponse(BaseModel):
    file: str
    summary: str

class SearchRequest(BaseModel):
    file_path: str
    query: str

class SearchResultOut(BaseModel):
    page: int
    snippet: str

class SearchResponse(BaseModel):
    file: str
    query: str
    matches: List[SearchResultOut]

class TavilySearchRequest(BaseModel):
    query: str
    max_results: int = 5

class TavilySearchResult(BaseModel):
    title: str
    content: str
    url: str = ""

class TavilySearchResponse(BaseModel):
    query: str
    results: List[TavilySearchResult]

class TavilyAnswerRequest(BaseModel):
    question: str

class TavilyAnswerResponse(BaseModel):
    question: str
    answer: str