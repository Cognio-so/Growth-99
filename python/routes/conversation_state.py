from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda
from pydantic import BaseModel, Field
import time

class ProjectEvolution(BaseModel):
    majorChanges: List[Any] = Field(default_factory=list)

class Context(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    edits: List[Any] = Field(default_factory=list)
    projectEvolution: ProjectEvolution = Field(default_factory=ProjectEvolution)
    userPreferences: Dict[str, Any] = Field(default_factory=dict)
    currentTopic: Optional[str] = None

class ConversationStateModel(BaseModel):
    conversationId: str
    startedAt: int
    lastUpdated: int
    context: Context

conversation_state: Optional[ConversationStateModel] = None

class GraphState(TypedDict, total=False):
    payload: Dict[str, Any]
    response: Dict[str, Any]

def _get_compute(_: Dict[str, Any]) -> Dict[str, Any]:
    if conversation_state is None:
        return {"success": True, "state": None, "message": "No active conversation"}
    return {"success": True, "state": conversation_state.model_dump()}

def _post_compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    global conversation_state
    action = payload.get("action")
    data = payload.get("data")
    try:
        if action == "reset":
            now = int(time.time() * 1000)
            conversation_state = ConversationStateModel(
                conversationId=f"conv-{int(time.time() * 1000)}",
                startedAt=now,
                lastUpdated=now,
                context=Context(),
            )
            print("[conversation-state] Reset conversation state")
            return {"success": True, "message": "Conversation state reset", "state": conversation_state.model_dump()}
        elif action == "clear-old":
            if conversation_state is None:
                return {"success": False, "error": "No active conversation to clear"}
            ctx = conversation_state.context
            ctx.messages = ctx.messages[-5:]
            ctx.edits = ctx.edits[-3:]
            ctx.projectEvolution.majorChanges = ctx.projectEvolution.majorChanges[-2:]
            print("[conversation-state] Cleared old conversation data")
            return {"success": True, "message": "Old conversation data cleared", "state": conversation_state.model_dump()}
        elif action == "update":
            if conversation_state is None:
                return {"success": False, "error": "No active conversation to update"}
            if data:
                if "currentTopic" in data:
                    conversation_state.context.currentTopic = data["currentTopic"]
                if "userPreferences" in data:
                    conversation_state.context.userPreferences = {
                        **conversation_state.context.userPreferences,
                        **data["userPreferences"],
                    }
                conversation_state.lastUpdated = int(time.time() * 1000)
            return {"success": True, "message": "Conversation state updated", "state": conversation_state.model_dump()}
        else:
            return {"success": False, "error": 'Invalid action. Use "reset" or "update"'}
    except Exception as e:
        print("[conversation-state] Error:", e)
        return {"success": False, "error": str(e)}

def _delete_compute(_: Dict[str, Any]) -> Dict[str, Any]:
    global conversation_state
    try:
        conversation_state = None
        print("[conversation-state] Cleared conversation state")
        return {"success": True, "message": "Conversation state cleared"}
    except Exception as e:
        print("[conversation-state] Error clearing state:", e)
        return {"success": False, "error": str(e)}

_get_processor = RunnableLambda(_get_compute)
_post_processor = RunnableLambda(_post_compute)
_delete_processor = RunnableLambda(_delete_compute)

def _node_factory(processor: RunnableLambda):
    def _node(state: GraphState) -> GraphState:
        payload = state.get("payload", {})
        resp = processor.invoke(payload)
        return {"response": resp}
    return _node

# GET graph
_get_sg = StateGraph(GraphState)
_get_sg.add_node("process", _node_factory(_get_processor))
_get_sg.set_entry_point("process")
_get_sg.add_edge("process", END)
_get_graph = _get_sg.compile()

# POST graph
_post_sg = StateGraph(GraphState)
_post_sg.add_node("process", _node_factory(_post_processor))
_post_sg.set_entry_point("process")
_post_sg.add_edge("process", END)
_post_graph = _post_sg.compile()

# DELETE graph
_delete_sg = StateGraph(GraphState)
_delete_sg.add_node("process", _node_factory(_delete_processor))
_delete_sg.set_entry_point("process")
_delete_sg.add_edge("process", END)
_delete_graph = _delete_sg.compile()

def GET() -> Dict[str, Any]:
    result = _get_graph.invoke({})
    return result["response"]

def POST(body: Dict[str, Any]) -> Dict[str, Any]:
    result = _post_graph.invoke({"payload": body})
    return result["response"]

def DELETE() -> Dict[str, Any]:
    result = _delete_graph.invoke({})
    return result["response"]
