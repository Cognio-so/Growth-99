# scrape_url_enhanced.py — Python equivalent of scrape_url_enhanced.ts (POST handler)
# - No web framework; callable directly from main_app.py
# - Uses LangChain RunnableLambda + a minimal LangGraph node to call Firecrawl
# - Preserves request/response shapes, sanitization, and messages

from typing import Any, Dict, Optional
import os
import json
import re
from datetime import datetime, timezone

import httpx
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END


# --- Function to sanitize smart quotes and other problematic characters ---
def sanitize_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Replace smart single quotes
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    # Replace smart double quotes
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)
    # Replace guillemets
    text = re.sub(r"[\u00AB\u00BB]", '"', text)
    text = re.sub(r"[\u2039\u203A]", "'", text)
    # Replace en/em dashes with hyphen
    text = re.sub(r"[\u2013\u2014]", "-", text)
    # Ellipsis
    text = re.sub(r"\u2026", "...", text)
    # Non-breaking space
    text = re.sub(r"\u00A0", " ", text)
    return text


async def _firecrawl_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Firecrawl API to scrape markdown + HTML with caching (maxAge).
    Mirrors the TS logic and error handling closely.
    """
    url: Optional[str] = payload.get("url")
    if not url:
        raise ValueError("URL is required")

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "url": url,
        "formats": ["markdown", "html"],
        "waitFor": 3000,
        "timeout": 30000,
        "blockAds": True,
        "maxAge": 3600000,  # Use cached data if < 1 hour old (500% faster!)
        "actions": [
            {"type": "wait", "milliseconds": 2000}
        ],
    }

    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers=headers,
            content=json.dumps(body),
        )

    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"Firecrawl API error: {resp.text}")

    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Firecrawl API error: invalid JSON response ({e})")


def _compile_graph():
    """
    Minimal LangGraph: START -> fetch -> END
    """
    graph = StateGraph(dict)

    async def fetch_node(state: Dict[str, Any]) -> Dict[str, Any]:
        chain = RunnableLambda(_firecrawl_fetch)
        data = await chain.ainvoke(state)
        return {"data": data}

    graph.add_node("fetch", fetch_node)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", END)
    return graph.compile()


_GRAPH = _compile_graph()


async def POST(req: Any) -> Dict[str, Any]:
    """
    Python equivalent of the Next.js POST handler.
    Accepts either:
      - an object with async .json() method, or
      - a plain dict (already-parsed JSON body).
    Returns a dict mirroring NextResponse.json payloads.
    """
    try:
        # Parse JSON body
        if hasattr(req, "json"):
            body = await req.json()
        elif isinstance(req, dict):
            body = req
        else:
            body = {}

        url = body.get("url")
        if not url:
            return {"success": False, "error": "URL is required", "status": 400}

        print("[scrape-url-enhanced] Scraping with Firecrawl:", url)

        # Run the Firecrawl request via the one-node graph
        result = await _GRAPH.ainvoke({"url": url})
        data = result.get("data", {})

        if not data.get("success") or not data.get("data"):
            raise RuntimeError("Failed to scrape content")

        scraped = data["data"]
        markdown = scraped.get("markdown") or ""
        # html is fetched but not needed in the final payload per the TS file
        metadata = scraped.get("metadata") or {}

        # Sanitize markdown and top fields
        sanitized_markdown = sanitize_quotes(markdown)
        title = sanitize_quotes(metadata.get("title", "") or "")
        description = sanitize_quotes(metadata.get("description", "") or "")

        # Format content for AI (same structure as TS)
        formatted_content = f"""
Title: {title}
Description: {description}
URL: {url}

Main Content:
{sanitized_markdown}
""".strip()

        # Build metadata with extras; preserve spread of Firecrawl metadata
        meta_out: Dict[str, Any] = {
            "scraper": "firecrawl-enhanced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "contentLength": len(formatted_content),
            "cached": bool(scraped.get("cached", False)),
        }
        if isinstance(metadata, dict):
            meta_out.update(metadata)

        return {
            "success": True,
            "url": url,
            "content": formatted_content,
            "structured": {
                "title": title,
                "description": description,
                "content": sanitized_markdown,
                "url": url,
            },
            "metadata": meta_out,
            "message": "URL scraped successfully with Firecrawl (with caching for 500% faster performance)",
        }

    except Exception as error:
        print("[scrape-url-enhanced] Error:", error)
        return {"success": False, "error": str(error), "status": 500}
