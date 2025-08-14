# route.py — Python equivalent of route.ts (GET handler)
# - No web framework; callable directly from main_app.py
# - Mirrors global.activeSandbox usage via module-level `active_sandbox`
# - Uses LangChain RunnableLambda and a tiny LangGraph to execute the sandbox code
# - Preserves messages, flow, and JSON structures exactly

from typing import Any, Dict, Optional
import json
import inspect

# LangChain / LangGraph (used where possible, without changing external behavior)
try:
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
except Exception as _e:
    raise

# Mirror the TS global: `global.activeSandbox`
active_sandbox: Optional[Any] = None


async def _run_in_sandbox(code: str) -> Dict[str, Any]:
    """
    Helper that calls the sandbox's run_code / runCode method.
    Returns whatever the sandbox returns (commonly a dict with 'output').
    Wrapped in LangChain + a one-node LangGraph to satisfy the requirement.
    """
    async def _call_runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        the_code = payload.get("code", "")
        if not active_sandbox:
            return {"output": ""}

        runner = getattr(active_sandbox, "run_code", None) or getattr(active_sandbox, "runCode", None)
        if runner is None:
            return {"output": ""}

        if inspect.iscoroutinefunction(runner):
            return await runner(the_code)
        else:
            return runner(the_code)

    chain = RunnableLambda(_call_runner)

    def _compile_graph():
        graph = StateGraph(dict)

        async def exec_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return await chain.ainvoke(state)

        graph.add_node("exec", exec_node)
        graph.add_edge(START, "exec")
        graph.add_edge("exec", END)
        return graph.compile()

    if not hasattr(_run_in_sandbox, "_compiled_graph"):
        _run_in_sandbox._compiled_graph = _compile_graph()

    graph = _run_in_sandbox._compiled_graph
    return await graph.ainvoke({"code": code})


async def GET(request: Any) -> Dict[str, Any]:
    """
    Python equivalent of the Next.js route GET handler.
    Returns plain dicts (no HTTP layer), preserving the original JSON payload shapes.
    """
    try:
        if not active_sandbox:
            # TS used 400 status; we include 'status' field for parity
            return {
                "success": False,
                "error": "No active sandbox",
                "status": 400,
            }

        print("[sandbox-logs] Fetching Vite dev server logs...")

        # === Exact code string preserved from the original route.ts ===
        sandbox_code = """
import subprocess
import os

# Try to get the Vite process output
try:
    # Read the last 100 lines of any log files
    log_content = []
    
    # Check if there are any node processes running
    ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    vite_processes = [line for line in ps_result.stdout.split('\\n') if 'vite' in line.lower()]
    
    if vite_processes:
        log_content.append("Vite is running")
    else:
        log_content.append("Vite process not found")
    
    # Try to capture recent console output (this is a simplified approach)
    # In a real implementation, you'd want to capture the Vite process output directly
    print(json.dumps({
        "hasErrors": False,
        "logs": log_content,
        "status": "running" if vite_processes else "stopped"
    }))
except Exception as e:
    print(json.dumps({
        "hasErrors": True,
        "logs": [str(e)],
        "status": "error"
    }))
""".lstrip("\n")
        # =============================================================

        result = await _run_in_sandbox(sandbox_code)

        # Mirror the TS JSON.parse(result.output || '{}') behavior
        output_text = (result.get("output") if isinstance(result, dict) else None) or ""
        try:
            log_data = json.loads(output_text or "{}")
            # Success path mirrors NextResponse.json({ success: true, ...logData })
            response = {"success": True}
            if isinstance(log_data, dict):
                response.update(log_data)
            else:
                # If parsed JSON isn’t a dict, just wrap it as a log
                response.update({
                    "hasErrors": False,
                    "logs": [log_data],
                    "status": "unknown",
                })
            return response
        except Exception:
            # Fallback when JSON.parse fails in TS
            return {
                "success": True,
                "hasErrors": False,
                "logs": [output_text],
                "status": "unknown",
            }

    except Exception as error:
        print("[sandbox-logs] Error:", error)
        # TS used 500 status with error.message; keep the same keys.
        return {
            "success": False,
            "error": str(error),
            "status": 500,
        }
