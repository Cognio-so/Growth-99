"""
detect_and_install_packages.py — Python equivalent of detect_and_install_packages.ts (POST handler)
- No web framework; callable directly from main_app.py
- Mirrors global.activeSandbox usage via module-level `active_sandbox`
- Uses LangChain RunnableLambda + a minimal LangGraph node to execute code in the sandbox
- Preserves messages, flow, and JSON structures from the TS version
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import re
import inspect

# LangChain / LangGraph (used where possible, without changing external behavior)
try:
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
except Exception as _e:
    raise

# ---- Mirror TS global: `global.activeSandbox` ----
active_sandbox: Optional[Any] = None


# ---- Helpers: sandbox execution via LangChain + LangGraph ----
async def _run_in_sandbox(code: str, *, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Run arbitrary code inside the sandbox using either .run_code or .runCode.
    Wrapped with LangChain RunnableLambda and dispatched via a one-node LangGraph.
    """

    async def _runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        c = payload.get("code", "")
        t = payload.get("timeout", None)

        if not active_sandbox:
            return {"output": ""}

        run = getattr(active_sandbox, "run_code", None) or getattr(active_sandbox, "runCode", None)
        if run is None:
            return {"output": ""}

        if inspect.iscoroutinefunction(run):
            # Best-effort pass of timeout kwarg if supported
            try:
                return await run(c, timeout=t)
            except TypeError:
                return await run(c)
        else:
            try:
                return run(c, timeout=t)
            except TypeError:
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
    return await graph.ainvoke({"code": code, "timeout": timeout})


def _extract_output_text(result: Any) -> str:
    """
    Best-effort extraction of text output from various possible result shapes,
    mirroring TS behavior (logs.stdout.join('')) when available.
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


# ---- Core logic (translation of TS POST handler) ----
async def POST(request: Any) -> Dict[str, Any]:
    """
    Python equivalent of the Next.js POST handler in detect_and_install_packages.ts.
    Accepts either:
      - an object with async .json() method, or
      - a plain dict (already-parsed JSON body).
    Returns dicts (no HTTP layer) with the same keys as the TS NextResponse.json payloads.
    """
    try:
        # Parse JSON body like NextRequest.json()
        if hasattr(request, "json"):
            body = await request.json()  # supports async .json()
        elif isinstance(request, dict):
            body = request
        else:
            body = {}

        files = body.get("files")

        if not files or not isinstance(files, dict):
            return {
                "success": False,
                "error": "Files object is required",
                "status": 400,
            }

        if not active_sandbox:
            return {
                "success": False,
                "error": "No active sandbox",
                "status": 404,
            }

        print("[detect-and-install-packages] Processing files:", list(files.keys()))

        # Extract all import statements from the files (ESM + CJS)
        imports: set[str] = set()
        import_regex = re.compile(r"""import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s*,?\s*)*(?:from\s+)?['"]([^'"]+)['"]""")
        require_regex = re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""")

        for file_path, content in files.items():
            if not isinstance(content, str):
                continue

            # Skip non-JS/JSX/TS/TSX files
            if not re.search(r"\.(jsx?|tsx?)$", file_path):
                continue

            # Find ES6 imports
            for m in import_regex.finditer(content):
                imports.add(m.group(1))

            # Find CommonJS requires
            for m in require_regex.finditer(content):
                imports.add(m.group(1))

        print("[detect-and-install-packages] Found imports:", list(imports))

        # Log specific heroicons imports
        heroicon_imports = [imp for imp in imports if "heroicons" in imp]
        if heroicon_imports:
            print("[detect-and-install-packages] Heroicon imports:", heroicon_imports)

        # Filter out relative imports and built-in Node modules
        builtins = {'fs', 'path', 'http', 'https', 'crypto', 'stream', 'util', 'os', 'url', 'querystring', 'child_process'}
        packages = []
        for imp in imports:
            # Skip relative imports
            if imp.startswith(".") or imp.startswith("/"):
                continue
            # Skip built-in Node modules
            if imp in builtins:
                continue
            # Keep everything else (scoped or regular)
            packages.append(imp)

        # Extract just the package names (without subpaths)
        package_names: List[str] = []
        for pkg in packages:
            if pkg.startswith("@"):
                # Scoped package: @scope/package or @scope/package/subpath
                parts = pkg.split("/")
                package_names.append("/".join(parts[:2]))
            else:
                # Regular package: package or package/subpath
                package_names.append(pkg.split("/")[0])

        # Remove duplicates (order not guaranteed, which matches TS set behavior)
        unique_packages = list(dict.fromkeys(package_names))

        print("[detect-and-install-packages] Packages to install:", unique_packages)

        if len(unique_packages) == 0:
            return {
                "success": True,
                "packagesInstalled": [],
                "message": "No new packages to install",
            }

        # Check which packages are already installed (via sandbox)
        check_code = f"""
import os
import json

installed = []
missing = []

packages = {json.dumps(unique_packages)}

for package in packages:
    # Handle scoped packages
    if package.startswith('@'):
        package_path = f"/home/user/app/node_modules/{{package}}"
    else:
        package_path = f"/home/user/app/node_modules/{{package}}"
    
    if os.path.exists(package_path):
        installed.append(package)
    else:
        missing.append(package)

result = {{
    'installed': installed,
    'missing': missing
}}

print(json.dumps(result))
""".lstrip("\n")

        check_result = await _run_in_sandbox(check_code)
        check_stdout = _extract_output_text(check_result)
        status = json.loads(check_stdout or "{}")

        print("[detect-and-install-packages] Package status:", status)

        if not status or not isinstance(status, dict):
            return {
                "success": False,
                "error": "Failed to determine package status",
                "status": 500,
            }

        if len(status.get("missing", []) or []) == 0:
            return {
                "success": True,
                "packagesInstalled": [],
                "packagesAlreadyInstalled": status.get("installed", []),
                "message": "All packages already installed",
            }

        # Install missing packages
        print("[detect-and-install-packages] Installing packages:", status["missing"])

        install_code = f"""
import subprocess
import os
import json

os.chdir('/home/user/app')
packages_to_install = {json.dumps(status["missing"])}

# Join packages into a single install command
packages_str = ' '.join(packages_to_install)
cmd = f'npm install {{packages_str}} --save'

print(f"Running: {{cmd}}")

# Run npm install with explicit save flag
result = subprocess.run(['npm', 'install', '--save'] + packages_to_install, 
                       capture_output=True, 
                       text=True, 
                       cwd='/home/user/app',
                       timeout=60)

print("stdout:", result.stdout)
if result.stderr:
    print("stderr:", result.stderr)

# Verify installation
installed = []
failed = []

for package in packages_to_install:
    # Handle scoped packages correctly
    if package.startswith('@'):
        # For scoped packages like @heroicons/react
        package_path = f"/home/user/app/node_modules/{{package}}"
    else:
        package_path = f"/home/user/app/node_modules/{{package}}"
    
    if os.path.exists(package_path):
        installed.append(package)
        print(f"✓ Verified installation of {{package}}")
    else:
        # Check if it's a submodule of an installed package
        base_package = package.split('/')[0]
        if package.startswith('@'):
            # For @scope/package, the base is @scope/package
            base_package = '/'.join(package.split('/')[:2])
        
        base_path = f"/home/user/app/node_modules/{{base_package}}"
        if os.path.exists(base_path):
            installed.append(package)
            print(f"✓ Verified installation of {{package}} (via {{base_package}})")
        else:
            failed.append(package)
            print(f"✗ Failed to verify installation of {{package}}")

result_data = {{
    'installed': installed,
    'failed': failed,
    'returncode': result.returncode
}}

print("\\nResult:", json.dumps(result_data))
""".lstrip("\n")

        install_result = await _run_in_sandbox(install_code, timeout=60000)
        raw_output = _extract_output_text(install_result)

        # Parse the result more safely (mirror TS)
        install_status: Dict[str, Any]
        try:
            m = re.search(r"Result:\s*({.*})", raw_output, flags=re.DOTALL)
            if m:
                install_status = json.loads(m.group(1))
            else:
                # Fallback parsing
                result_line = None
                for line in raw_output.splitlines():
                    if "Result:" in line:
                        result_line = line
                        break
                if result_line:
                    install_status = json.loads(result_line.split("Result:")[1].strip())
                else:
                    raise ValueError("Could not find Result in output")
        except Exception as parse_error:
            print("[detect-and-install-packages] Failed to parse install result:", parse_error)
            print("[detect-and-install-packages] stdout:", raw_output)
            # Fallback to assuming all missing packages were installed
            install_status = {
                "installed": status.get("missing", []),
                "failed": [],
                "returncode": 0,
            }

        if install_status.get("failed"):
            print("[detect-and-install-packages] Failed to install:", install_status["failed"])

        return {
            "success": True,
            "packagesInstalled": install_status.get("installed", []),
            "packagesFailed": install_status.get("failed", []),
            "packagesAlreadyInstalled": status.get("installed", []),
            "message": f"Installed {len(install_status.get('installed', []))} packages",
            "logs": raw_output,
        }

    except Exception as error:
        print("[detect-and-install-packages] Error:", error)
        return {
            "success": False,
            "error": str(error),
            "status": 500,
        }
