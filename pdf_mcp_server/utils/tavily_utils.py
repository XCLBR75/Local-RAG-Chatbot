from typing import List, Dict, Any
import httpx
import os
import logging
from dotenv import load_dotenv

# load .env from project root
load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # load from env var


async def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Run a search query on Tavily and return results."""
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": query, "num_results": max_results}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(BASE_URL, json=payload, headers=headers)
            logger.info(f"Tavily response status: {resp.status_code}")
            logger.info(f"Tavily response text: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.exception("Error calling Tavily API")
            raise

    return data.get("results", [])
