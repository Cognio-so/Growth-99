# scrape_screenshot.py — Python equivalent of scrape_screenshot.ts (POST handler)
# - No web framework; callable directly from main_app.py
# - Uses LangChain RunnableLambda + a minimal LangGraph node to call Firecrawl
# - Preserves request/response shapes and error messages

from typing import Any, Dict, Optional
import os
import json

# Third-party HTTP client for async calls
import httpx

# LangChain / LangGraph
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END


async def _firecrawl_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single step that calls Firecrawl API to capture a regular viewport screenshot.
    Returns the parsed JSON from Firecrawl (same as TS 'await response.json()').
    """
    url: Optional[str] = payload.get("url")
    if not url:
        # Mirror TS early validation in POST; this branch is usually not hit
        raise ValueError("URL is required")

    headers = {
        "Authorization": f"Bearer {os.getenv('FIRECRAWL_API_KEY')}",
        "Content-Type": "application/json",
    }

    body = {
        "url": url,
        "formats": ["screenshot@fullPage"],   # Regular viewport screenshot, not full page
        "waitFor": 3000,             # Wait for page to fully load (ms)
        "timeout": 30000,            # Request timeout (ms)
        "blockAds": True,
        "actions": [
            {"type": "wait", "milliseconds": 2000}  # Additional wait for dynamic content
        ],
    }

    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers=headers,
            content=json.dumps(body),
        )

    if resp.status_code < 200 or resp.status_code >= 300:
        # Keep same error style: include Firecrawl's response text
        raise RuntimeError(f"Firecrawl API error: {resp.text}")

    try:
        return resp.json()
    except Exception as e:
        # If JSON parsing fails, reflect a clear error
        raise RuntimeError(f"Firecrawl API error: invalid JSON response ({e})")


def _compile_graph():
    """
    Minimal LangGraph: START -> fetch -> END
    """
    graph = StateGraph(dict)

    async def fetch_node(state: Dict[str, Any]) -> Dict[str, Any]:
        chain = RunnableLambda(_firecrawl_fetch)
        data = await chain.ainvoke(state)
        # Pass through Firecrawl's JSON as 'data' for the caller
        return {"data": data}

    graph.add_node("fetch", fetch_node)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", END)
    return graph.compile()


# Cache compiled graph
_GRAPH = _compile_graph()


async def POST(req: Any) -> Dict[str, Any]:
    """
    Python equivalent of the Next.js POST handler.
    Accepts either:
      - an object with async .json() method, or
      - a plain dict (already-parsed JSON body).
    Returns a dict mirroring NextResponse.json payload.
    """
    try:
        # Parse JSON body (mirror NextRequest.json())
        if hasattr(req, "json"):
            body = await req.json()  # supports async .json()
        elif isinstance(req, dict):
            body = req
        else:
            body = {}

        url = body.get("url")
        if not url:
            return {"error": "URL is required", "status": 400}

        # Run the Firecrawl request via the one-node graph
        result = await _GRAPH.ainvoke({"url": url})
        data = result.get("data", {})

        # Mirror TS checks: require success and data.screenshot
        if not data.get("success") or not (data.get("data") or {}).get("screenshot"):
            raise RuntimeError("Failed to capture screenshot")

        return {
            "success": True,
            "screenshot": data["data"]["screenshot"],
            "metadata": data["data"].get("metadata"),
        }

    except Exception as error:
        # Keep messages aligned with TS
        msg = str(error) if getattr(error, "args", None) else "Failed to capture screenshot"
        # Log analog
        print("Screenshot capture error:", error)
        return {"error": msg or "Failed to capture screenshot", "status": 500}
