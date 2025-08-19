import traceback
import httpx
import asyncio
import sys
import requests
from typing import Dict, Any

MCP_BASE_URL = "http://127.0.0.1:8000"

# -------------------------------
# Call Tools
# -------------------------------
async def call_mcp_tool(endpoint: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    """Generic async MCP tool caller with debug output."""
    url = f"{MCP_BASE_URL}/{endpoint}"
    print(f"[DEBUG] Sending MCP request → {url}")
    print(f"[DEBUG] Payload: {payload}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=payload)
            print(f"[DEBUG] Response status: {resp.status_code}")
            print(f"[DEBUG] Raw response text: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            print(f"[DEBUG] Parsed JSON: {data}")
            return data
        except Exception as e:
            print(f"[ERROR] MCP call failed: {e}")
            traceback.print_exc()
            # If the exception is an HTTPStatusError, print the response content
            if hasattr(e, 'response') and e.response is not None:
                print(f"[ERROR] Response content: {e.response.text}")
            raise

async def mcp_tavily_search(query: str, timeout: float = 60.0) -> Dict[str, Any]:
    return await call_mcp_tool("tavily_search", {"query": query}, timeout)

async def mcp_parse_pdf(file_path: str, timeout: float = 60.0) -> Dict[str, Any]:
    return await call_mcp_tool("parse_pdf", {"file_path": file_path}, timeout)

async def mcp_summarize_pdf(file_path: str, max_sentences: int = 5, timeout: float = 60.0) -> Dict[str, Any]:
    return await call_mcp_tool("summarize_pdf", {"file_path": file_path, "max_sentences": max_sentences}, timeout)

async def mcp_search_pdf(file_path: str, query: str, timeout: float = 60.0) -> Dict[str, Any]:
    return await call_mcp_tool("search_pdf", {"file_path": file_path, "query": query}, timeout)


# -------------------------------
# Main CLI for debugging
# -------------------------------
if __name__ == "__main__":
    tool = sys.argv[1] if len(sys.argv) > 1 else "parse_pdf"
    file_path = sys.argv[2] if len(sys.argv) > 2 else "sample.pdf"

    if tool == "parse_pdf":
        data = asyncio.run(mcp_parse_pdf(file_path))
    elif tool == "summarize_pdf":
        data = asyncio.run(mcp_summarize_pdf(file_path, max_sentences=5))
    elif tool == "search_pdf":
        if len(sys.argv) < 4:
            print("Usage: python mcp_client.py search_pdf <file_path> <query>")
            sys.exit(1)
        query = sys.argv[3]
        data = asyncio.run(mcp_search_pdf(file_path, query))
    else:
        raise ValueError(f"Unknown tool: {tool}")

    print(data)
