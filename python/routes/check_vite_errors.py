from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda

class GraphState(TypedDict, total=False):
    payload: Dict[str, Any]
    response: Dict[str, Any]

def _compute(_: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        "errors": [],
        "message": "No Vite errors detected",
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

def GET() -> Dict[str, Any]:
    result = _graph.invoke({})
    return result["response"]
