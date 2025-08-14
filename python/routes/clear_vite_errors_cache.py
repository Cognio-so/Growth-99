from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda

vite_errors_cache: Optional[Dict[str, Any]] = None

class GraphState(TypedDict, total=False):
    payload: Dict[str, Any]
    response: Dict[str, Any]

def _compute(_: Dict[str, Any]) -> Dict[str, Any]:
    global vite_errors_cache
    vite_errors_cache = None
    print("[clear-vite-errors-cache] Cache cleared")
    return {
        "success": True,
        "message": "Vite errors cache cleared",
    }

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

def POST() -> Dict[str, Any]:
    try:
        result = _graph.invoke({})
        return result["response"]
    except Exception as e:
        print("[clear-vite-errors-cache] Error:", e)
        return {"success": False, "error": str(e)}
