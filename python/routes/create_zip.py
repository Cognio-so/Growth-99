from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda

active_sandbox: Optional[Any] = None  # expected to expose run_code(code: str, timeout: Optional[int] = None)

class GraphState(TypedDict, total=False):
    payload: Dict[str, Any]
    response: Dict[str, Any]

_CREATE_ZIP_CODE = """
import zipfile
import os
import json

os.chdir('/home/user/app')

# Create zip file
with zipfile.ZipFile('/tmp/project.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('.'):
        # Skip node_modules and .git
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '.next', 'dist']]
        
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '.')
            zipf.write(file_path, arcname)

# Get file size
file_size = os.path.getsize('/tmp/project.zip')
print(f" Created project.zip ({file_size} bytes)")
"""

_READ_AND_B64_CODE = """
import base64

with open('/tmp/project.zip', 'rb') as f:
    content = f.read()
    encoded = base64.b64encode(content).decode('utf-8')
    print(encoded)
"""

def _compute(_: Dict[str, Any]) -> Dict[str, Any]:
    if active_sandbox is None:
        return {"success": False, "error": "No active sandbox"}
    print("[create-zip] Creating project zip...")
    try:
        active_sandbox.run_code(_CREATE_ZIP_CODE)
        read_result = active_sandbox.run_code(_READ_AND_B64_CODE)
        base64_content = ""
        if isinstance(read_result, dict):
            logs = read_result.get("logs", {})
            stdout = logs.get("stdout") if isinstance(logs, dict) else None
            if isinstance(stdout, list):
                base64_content = "".join(stdout).strip()
            if not base64_content:
                base64_content = (read_result.get("output") or "").strip()
        elif isinstance(read_result, str):
            base64_content = read_result.strip()

        data_url = f"data:application/zip;base64,{base64_content}"
        return {
            "success": True,
            "dataUrl": data_url,
            "fileName": "e2b-project.zip",
            "message": "Zip file created successfully",
        }
    except Exception as e:
        print("[create-zip] Error:", e)
        return {"success": False, "error": str(e)}

_processor = RunnableLambda(_compute)

def _node(state: GraphState) -> GraphState:
    resp = _processor.invoke(state.get("payload", {}))
    return {"response": resp}

_sg = StateGraph(GraphState)
_sg.add_node("process", _node)
_sg.set_entry_point("process")
_sg.add_edge("process", END)
_graph = _sg.compile()

def POST() -> Dict[str, Any]:
    result = _graph.invoke({})
    return result["response"]
