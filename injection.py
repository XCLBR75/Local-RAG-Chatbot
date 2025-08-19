import os
import logging
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.config import Configure
from langchain_weaviate.vectorstores import WeaviateVectorStore
from embeddings import OllamaEmbedding
from PyPDF2 import PdfReader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pdf_mcp_server.mcp_client import mcp_parse_pdf


load_dotenv()
logging.basicConfig(level=logging.INFO)
executor = ThreadPoolExecutor()


def deterministic_id(text: str) -> str:
    """Stable UUID based on the text payload.
    We use uuid5 (namespace-based) so the same text always maps to the same id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def _estimate_tokens(text: str) -> int:
    """Token estimator.
    Approximate tokens as ~1.3 * words.
    """
    words = len(text.split())
    return max(1, int(words * 1.3))


def _normalize_page_text(raw: str) -> str:
    """Normalize PDF-extracted text while keeping paragraph boundaries.
    """
    if not raw:
        return ""

    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Merge hyphenated line breaks
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    # Replace single newlines inside paragraphs with space
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # Collapse whitespaces
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences based on punctuation and sentence starts."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"])", text)
    return [p.strip() for p in parts if p and p.strip()]


@dataclass
class Chunk:
    content: str
    page: Optional[int]
    index: int  
    tokens: int
    source: str


def _chunk_sentences(
    sentences: List[str],
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> List[Chunk]:
    """Greedy sentence-level chunking with sliding-window overlap.

    - Accumulates sentences until adding the next one would exceed max_tokens.
    - Emits a chunk, then starts the next with an overlap of ~overlap_tokens
      worth of trailing sentences from the previous chunk.
    """
    chunks: List[Chunk] = []
    if not sentences:
        return chunks

    current: List[str] = []
    current_tokens = 0
    running_index = 0

    for sent in sentences:
        t = _estimate_tokens(sent)
        if current and current_tokens + t > max_tokens:
            content = " ".join(current).strip()
            chunks.append(Chunk(content=content, page=None, index=running_index,
                                 tokens=_estimate_tokens(content), source=""))
            running_index += 1
            # Build overlap window 
            overlap: List[str] = []
            acc = 0
            for s in reversed(current):
                acc += _estimate_tokens(s)
                overlap.insert(0, s)
                if acc >= overlap_tokens:
                    break
            current = overlap + [sent]
            current_tokens = sum(_estimate_tokens(s) for s in current)
        else:
            current.append(sent)
            current_tokens += t

    if current:
        content = " ".join(current).strip()
        chunks.append(Chunk(content=content, page=None, index=running_index,
                             tokens=_estimate_tokens(content), source=""))

    return chunks


async def pdf_to_chunks_async(
    file_path: str,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
    use_mcp: bool = False,
) -> List[Chunk]:
    """Async version of pdf_to_chunks that can handle MCP parsing properly."""
    source = Path(file_path).name
    all_chunks: List[Chunk] = []

    if use_mcp:
        # Use MCP server for PDF parsing
        try:
            result = await mcp_parse_pdf(file_path)
            pdf_chunks = result.get('chunks', [])
            
            for chunk_data in pdf_chunks:
                page_num = chunk_data.get('page', 1)
                raw_text = chunk_data.get('text', '')
                
                norm = _normalize_page_text(raw_text)
                if not norm:
                    continue
                
                sents = _split_sentences(norm)
                page_chunks = _chunk_sentences(sents, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
               
                for idx, ch in enumerate(page_chunks):
                    all_chunks.append(Chunk(
                        content=ch.content,
                        page=page_num,
                        index=idx,
                        tokens=ch.tokens,
                        source=source,
                    ))
                    
        except Exception as e:
            logging.warning(f"MCP parsing failed for {file_path}, falling back to PyPDF2: {e}")
            # Fall back to PyPDF2 if MCP fails
            use_mcp = False
    
    if not use_mcp:
        # Use PyPDF2 for PDF parsing
        reader = PdfReader(file_path)
        
        for pageno, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            norm = _normalize_page_text(raw)
            if not norm:
                continue
            sents = _split_sentences(norm)
            page_chunks = _chunk_sentences(sents, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
           
            for idx, ch in enumerate(page_chunks):
                all_chunks.append(Chunk(
                    content=ch.content,
                    page=pageno,
                    index=idx,
                    tokens=ch.tokens,
                    source=source,
                ))

    return all_chunks


def semantic_textfile_to_chunks(file_path: str, max_tokens: int = 120, overlap_tokens: int = 20):
    """Fine-grained semantic chunking for fact-style .txt files.
    
    - Each sentence as its own unit.
    - Merge until hitting a small max token size (default 120).
    - Keeps small overlaps for context.
    """
    with open(file_path, encoding="utf-8") as f:
        raw = f.read()

    norm = _normalize_page_text(raw)
    facts = re.split(r"(?:\n\s*\n|^- |\.\s+)", norm)
    facts = [f.strip() for f in facts if f.strip()]

    chunks = []
    current = []
    current_tokens = 0
    idx = 0
    for fact in facts:
        t = _estimate_tokens(fact)
        if current and current_tokens + t > max_tokens:
            # emit chunk
            content = " ".join(current)
            chunks.append(Chunk(content=content, page=None, index=idx, 
                                tokens=_estimate_tokens(content), source=Path(file_path).name))
            idx += 1
            # overlap
            overlap = []
            acc = 0
            for s in reversed(current):
                acc += _estimate_tokens(s)
                overlap.insert(0, s)
                if acc >= overlap_tokens:
                    break
            current = overlap + [fact]
            current_tokens = sum(_estimate_tokens(s) for s in current)
        else:
            current.append(fact)
            current_tokens += t

    if current:
        content = " ".join(current)
        chunks.append(Chunk(content=content, page=None, index=idx, 
                            tokens=_estimate_tokens(content), source=Path(file_path).name))

    return chunks


async def async_exists(collection, uid: str) -> bool:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(executor, collection.data.exists, uid)
    except Exception as e:
        logging.warning(f"Exist check failed for {uid}: {e}")
        return False


async def inject_dataset(
    file_path: str,
    topic: str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 60,
    use_mcp_parser: bool = False,
) -> Tuple[Dict[str, WeaviateVectorStore], weaviate.WeaviateClient]:
    """Extracts, chunks, deduplicates, and upserts content into Weaviate.

    - For PDFs, performs page-aware sentence chunking with overlap.
    - For text files, performs sentence chunking across the whole file.
    - Uses deterministic UUIDs for idempotent deduplication.
    
    Args:
        file_path: Path to the file to process
        topic: Topic name for the Weaviate collection
        chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        use_mcp_parser: If True, use MCP server for PDF parsing
    """
    is_pdf = file_path.lower().endswith('.pdf')

    if is_pdf:
        chunks = await pdf_to_chunks_async(file_path, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens, use_mcp=use_mcp_parser)
    else:
        chunks = semantic_textfile_to_chunks(file_path, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

    parser_method = "MCP server" if use_mcp_parser and is_pdf else "PyPDF2/direct"
    logging.info(
        f"Prepared {len(chunks)} chunks from '{file_path}' using {parser_method} (chunk≈{chunk_tokens} tokens, overlap≈{overlap_tokens})."
    )

    # Connect to Weaviate
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    assert client.is_ready(), "Weaviate not ready"

    # Ensure collection exists with metadata fields we plan to write
    try:
        collection = client.collections.get(topic)
        logging.info(f"Using existing collection '{topic}'")
    except WeaviateBaseError:
        logging.info(f"Collection '{topic}' does not exist – creating it.")
        collection = client.collections.create(
            name=topic,
            properties=[
                {"name": "content", "dataType": "text"},
                {"name": "source", "dataType": "text"},
                {"name": "page", "dataType": "int"},
                {"name": "chunk", "dataType": "int"},
                {"name": "tokens", "dataType": "int"},
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )

    embedding = OllamaEmbedding("nomic-embed-text")
    vectorstore = WeaviateVectorStore(
        client=client, index_name=topic, text_key="content", embedding=embedding
    )

    tasks = []
    packed: List[Tuple[Chunk, str]] = []
    for ch in chunks:
        id_seed = f"{ch.source}|p{ch.page if ch.page is not None else 0}|c{ch.index}|{ch.content}"
        uid = deterministic_id(id_seed)
        tasks.append(async_exists(collection, uid))
        packed.append((ch, uid))

    results = await asyncio.gather(*tasks)

    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict] = []

    for (ch, uid), exists in zip(packed, results):
        if not exists:
            texts.append(ch.content)
            ids.append(uid)
            metadatas.append({
                "source": ch.source,
                "page": ch.page if ch.page is not None else -1,
                "chunk": ch.index,
                "tokens": ch.tokens,
            })
        else:
            logging.debug(f"Skipped duplicate: {uid}")

    if texts:
        vectorstore.add_texts(texts=texts, ids=ids, metadatas=metadatas)
        logging.info(f"Injected {len(texts)} new chunks (skipped {len(chunks) - len(texts)} duplicates).")
    else:
        logging.info("No new chunks to inject – all items existed already.")

    return {topic: vectorstore}, client