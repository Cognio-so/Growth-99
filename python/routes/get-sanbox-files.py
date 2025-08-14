# get_sandbox_files.py — Python equivalent of get_sandbox_files.ts (GET handler)
# - No web framework; callable directly from main_app.py
# - Mirrors global.activeSandbox and global.sandboxState usage
# - Uses LangChain RunnableLambda + a minimal LangGraph node to run code in the sandbox
# - Preserves file enumeration logic, manifest construction, and route extraction

from typing import Any, Dict, List, Optional, Tuple
import json
import time
import re
import inspect

# LangChain / LangGraph (used where possible, without changing external behavior)
try:
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
except Exception as _e:
    # If these aren’t available in your env:
    # pip install langchain langgraph
    raise

# ---- Globals mirroring the TS file ----
active_sandbox: Optional[Any] = None
sandbox_state: Optional[Dict[str, Any]] = None
# ---------------------------------------


# ---- Helpers: sandbox execution via LangChain + LangGraph ----
async def _run_in_sandbox(code: str) -> Dict[str, Any]:
    """
    Run arbitrary code inside the sandbox using either .run_code or .runCode.
    Wrapped with LangChain RunnableLambda and dispatched via a one-node LangGraph.
    """
    async def _runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        c = payload.get("code", "")
        if not active_sandbox:
            return {"output": ""}
        run = getattr(active_sandbox, "run_code", None) or getattr(active_sandbox, "runCode", None)
        if run is None:
            return {"output": ""}
        if inspect.iscoroutinefunction(run):
            return await run(c)
        return run(c)

    chain = RunnableLambda(_runner)

    def _compile_graph():
        g = StateGraph(dict)

        async def exec_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return await chain.ainvoke(state)

        g.add_node("exec", exec_node)
        g.add_edge(START, "exec")
        g.add_edge("exec", END)
        return g.compile()

    if not hasattr(_run_in_sandbox, "_graph"):
        _run_in_sandbox._graph = _compile_graph()

    graph = _run_in_sandbox._graph
    return await graph.ainvoke({"code": code})


def _extract_output_text(result: Any) -> str:
    """
    Best-effort extraction of text output from various runner result shapes.
    Mirrors TS behavior (result.logs.stdout.join('')) when available.
    """
    if isinstance(result, dict):
        # Prefer direct 'output'
        out = result.get("output")
        if isinstance(out, str):
            return out

        # E2B-like shape: logs.stdout array
        logs = result.get("logs")
        if isinstance(logs, dict):
            stdout = logs.get("stdout")
            if isinstance(stdout, list) and all(isinstance(x, str) for x in stdout):
                return "".join(stdout)
    return ""


# ---- Minimal counterparts to the TS utilities (parseJavaScriptFile, buildComponentTree) ----
def parse_javascript_file(content: str, full_path: str) -> Dict[str, Any]:
    """
    Lightweight JS/TS/JSX/TSX parser to enrich FileInfo similarly to the TS version.
    We keep it conservative to avoid changing behavior:
      - Detect imports, exports
      - Heuristically tag component-like files
    """
    imports: List[str] = []
    exports: List[str] = []
    for m in re.finditer(r"""import\s+(?:[^'"]+?\s+from\s+)?['"]([^'"]+)['"]""", content):
        imports.append(m.group(1))

    for m in re.finditer(r"""export\s+(?:default\s+)?(?:const|function|class)?\s*([A-Za-z0-9_]+)?""", content):
        name = (m.group(1) or "default").strip()
        if name and name not in exports:
            exports.append(name)

    is_component = bool(re.search(r"""(export\s+default\s+function|function\s+[A-Z][A-Za-z0-9_]*\s*\(|<Route|createBrowserRouter)""", content))

    file_type = "component" if is_component else "utility"
    return {
        "imports": imports,
        "exports": exports,
        "type": file_type,
        "hasRoutes": ("<Route" in content) or ("createBrowserRouter" in content),
        "path": full_path,
    }


def build_component_tree(files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Very small component graph: for each file, list its imported modules (by specifier).
    This mirrors the intent of buildComponentTree without changing caller behavior.
    """
    tree = {}
    for path, info in files.items():
        imports = info.get("imports") if isinstance(info, dict) else None
        tree[path] = {
            "imports": list(imports) if isinstance(imports, list) else [],
            "type": info.get("type") if isinstance(info, dict) else "utility",
        }
    return tree


def extract_routes(files: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Direct translation of the TS `extractRoutes` logic:
      - Look for React Router usage and capture path="..." definitions
      - Handle Next.js-style pages directory layout
    """
    routes: List[Dict[str, str]] = []

    for path, file_info in files.items():
        content = file_info.get("content", "") if isinstance(file_info, dict) else ""
        relative_path = file_info.get("relativePath", "") if isinstance(file_info, dict) else ""

        if ("<Route" in content) or ("createBrowserRouter" in content):
            # Simplified extraction aligned with TS
            for m in re.finditer(r"""path=["']([^"']+)["'].*?(?:element|component)={(.*?)}""", content, flags=re.DOTALL):
                route_path = m.group(1)
                routes.append({
                    "path": route_path,
                    "component": path,
                })

        # Next.js-style pages detection
        if relative_path.startswith("pages/") or relative_path.startswith("src/pages/"):
            route_path = (
                "/" + re.sub(r"^(src/)?pages/", "", relative_path)
                .replace(".jsx", "")
                .replace(".js", "")
                .replace(".tsx", "")
                .replace(".ts", "")
            )
            route_path = re.sub(r"/index$", "", route_path)
            routes.append({
                "path": route_path,
                "component": path,
            })

    return routes


# ---- GET handler ----
async def GET() -> Dict[str, Any]:
    """
    Python equivalent of the Next.js GET handler in get_sandbox_files.ts.
    Returns plain dicts (no HTTP layer), preserving the original JSON payload shapes.
    """
    try:
        if not active_sandbox:
            return {
                "success": False,
                "error": "No active sandbox",
                "status": 404,
            }

        print("[get-sandbox-files] Fetching and analyzing file structure...")

        # === Exact code string preserved from the original TS ===
        sandbox_code = """
import os
import json

def get_files_content(directory='/home/user/app', extensions=['.jsx', '.js', '.tsx', '.ts', '.css', '.json']):
    files_content = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip node_modules and other unwanted directories
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'dist', 'build']]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, '/home/user/app')
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Only include files under 10KB to avoid huge responses
                        if len(content) < 10000:
                            files_content[relative_path] = content
                except:
                    pass
    
    return files_content

# Get the files
files = get_files_content()

# Also get the directory structure
structure = []
for root, dirs, files in os.walk('/home/user/app'):
    level = root.replace('/home/user/app', '').count(os.sep)
    indent = ' ' * 2 * level
    structure.append(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 2 * (level + 1)
    for file in files:
        if not any(skip in root for skip in ['node_modules', '.git', 'dist', 'build']):
            structure.append(f"{sub_indent}{file}")

result = {
    'files': files,
    'structure': '\\n'.join(structure[:50])  # Limit structure to 50 lines
}

print(json.dumps(result))
""".lstrip("\n")
        # ========================================================

        run_result = await _run_in_sandbox(sandbox_code)
        output_text = _extract_output_text(run_result)
        parsed_result = json.loads(output_text or "{}")

        # Build enhanced file manifest
        now_ms = int(time.time() * 1000)
        file_manifest: Dict[str, Any] = {
            "files": {},
            "routes": [],
            "componentTree": {},
            "entryPoint": "",
            "styleFiles": [],
            "timestamp": now_ms,
        }

        files_dict = parsed_result.get("files", {}) or {}
        for relative_path, content in files_dict.items():
            full_path = f"/home/user/app/{relative_path}"

            # Base FileInfo
            file_info: Dict[str, Any] = {
                "content": content,
                "type": "utility",
                "path": full_path,
                "relativePath": relative_path,
                "lastModified": now_ms,
            }

            # Parse JavaScript/JSX/TS/TSX files
            if re.search(r"\.(jsx?|tsx?)$", relative_path):
                parsed = parse_javascript_file(content, full_path)
                file_info.update(parsed)

                # Identify entry point
                if relative_path in ("src/main.jsx", "src/index.jsx"):
                    file_manifest["entryPoint"] = full_path

                # Identify App.jsx (fallback if no entry set)
                if relative_path in ("src/App.jsx", "App.jsx"):
                    file_manifest["entryPoint"] = file_manifest.get("entryPoint") or full_path

            # Track style files
            if relative_path.endswith(".css"):
                file_manifest["styleFiles"].append(full_path)
                file_info["type"] = "style"

            file_manifest["files"][full_path] = file_info

        # Build component tree
        file_manifest["componentTree"] = build_component_tree(file_manifest["files"])

        # Extract routes (simplified, aligned with TS)
        file_manifest["routes"] = extract_routes(file_manifest["files"])

        # Update global file cache with manifest
        if isinstance(sandbox_state, dict):
            fc = sandbox_state.get("fileCache")
            if isinstance(fc, dict):
                fc["manifest"] = file_manifest

        return {
            "success": True,
            "files": files_dict,
            "structure": parsed_result.get("structure", ""),
            "fileCount": len(files_dict),
            "manifest": file_manifest,
        }

    except Exception as error:
        print("[get-sandbox-files] Error:", error)
        return {
            "success": False,
            "error": str(error),
            "status": 500,
        }
