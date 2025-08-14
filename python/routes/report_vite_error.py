from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda
import re
from datetime import datetime

vite_errors: List[Dict[str, Any]] = []

class GraphState(TypedDict, total=False):
    payload: Dict[str, Any]
    response: Dict[str, Any]

def _compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        error_msg = payload.get("error")
        file = payload.get("file")
        type_ = payload.get("type", "runtime-error")
        if not error_msg:
            return {"success": False, "error": "Error message is required"}
        error_obj: Dict[str, Any] = {
            "type": type_,
            "message": error_msg,
            "file": file or "unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        m = re.search(r"Failed to resolve import ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]", error_msg or "")
        if m:
            error_obj["type"] = "import-error"
            error_obj["import"] = m.group(1)
            error_obj["file"] = m.group(2)
        vite_errors.append(error_obj)
        if len(vite_errors) > 50:
            del vite_errors[:-50]
        print("[report-vite-error] Error reported:", error_obj)
        return {"success": True, "message": "Error reported successfully", "error": error_obj}
    except Exception as e:
        print("[report-vite-error] Error:", e)
        return {"success": False, "error": str(e)}

_processor = RunnableLambda(_compute)

def _node(state: GraphState) -> GraphState:
    payload = state.get("payload", {})
    resp = _processor.invoke(payload)
    return {"response": resp}

_sg = StateGraph(GraphState)
_sg.add_node("process", _node)
_sg.set_entry_point("process")
_sg.add_edge("process", END)
_graph = _sg.compile()

def POST(body: Dict[str, Any]) -> Dict[str, Any]:
    result = _graph.invoke({"payload": body})
    return result["response"]
