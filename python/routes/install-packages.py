# install_packages.py — Python equivalent of install_packages.ts (POST handler with SSE-like streaming)
# - No web framework; callable directly from main_app.py
# - Mirrors global.activeSandbox / global.sandboxData
# - Uses LangChain RunnableLambda and a minimal LangGraph node to execute code in the sandbox
# - Preserves messages, flow, and JSON structures as in the original TS implementation

from typing import Any, Dict, List, Optional, AsyncIterator
import asyncio
import json
import inspect
import re
import os

# LangChain / LangGraph (used where possible, without changing behavior)
try:
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
except Exception as _e:
    # If these aren’t available in your env, install:
    # pip install langchain langgraph
    raise

# Optional E2B Sandbox (only used for reconnect semantics if installed)
try:
    # Common python package name is `e2b` or `e2b_code_interpreter`; we soft-import to avoid hard dependency
    from e2b import Sandbox as E2BSandbox  # type: ignore
except Exception:
    E2BSandbox = None  # We only use if present; otherwise we rely on active_sandbox provided by your app


# ---- Globals to mirror TypeScript `declare global` ----
active_sandbox: Optional[Any] = None
sandbox_data: Optional[Any] = None
# ------------------------------------------------------


# ---- Minimal SSE-style streaming response ----
class EventStreamResponse:
    """
    Lightweight SSE-like response:
      - headers: for compatibility with SSE consumers
      - async iteration yields 'data: <json>\\n\\n' bytes, one event at a time
    """
    def __init__(self) -> None:
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self.headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        self._closed = False

    async def send(self, data: Dict[str, Any]) -> None:
        if self._closed:
            return
        payload = f"data: {json.dumps(data)}\n\n".encode("utf-8")
        await self._queue.put(payload)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
# ------------------------------------------------------


# ---- LangChain / LangGraph helpers to run sandbox code without changing behavior ----
async def _run_in_sandbox(code: str, *, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Helper that calls the sandbox's run_code / runCode method. Returns a dict like {'output': '...'}.
    Wrapped with LangChain RunnableLambda and dispatched through a minimal LangGraph node.
    """
    async def _call_runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        the_code = payload.get("code", "")
        the_timeout = payload.get("timeout", None)

        if not active_sandbox:
            return {"output": ""}

        runner = getattr(active_sandbox, "run_code", None) or getattr(active_sandbox, "runCode", None)
        if runner is None:
            return {"output": ""}

        # If the sandbox runner supports a timeout kwarg, pass it through
        if inspect.iscoroutinefunction(runner):
            try:
                return await runner(the_code, timeout=the_timeout)  # type: ignore[arg-type]
            except TypeError:
                return await runner(the_code)
        else:
            try:
                return runner(the_code, timeout=the_timeout)  # type: ignore[arg-type]
            except TypeError:
                return runner(the_code)

    # LangChain Runnable
    chain = RunnableLambda(_call_runner)

    # Minimal LangGraph: START -> exec -> END
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
    return await graph.ainvoke({"code": code, "timeout": timeout})
# ------------------------------------------------------------------------------


def _extract_output_text(result: Any) -> str:
    """
    Best-effort extraction of text output from various possible result shapes.
    Mirrors TS fallbacks: output OR logs.stdout.join('\\n'), etc.
    """
    if isinstance(result, dict):
        if "output" in result and isinstance(result["output"], str):
            return result["output"]

        # Some runners return logs = {'stdout': [...], 'stderr': [...]}
        logs = result.get("logs")
        if isinstance(logs, dict):
            stdout = logs.get("stdout")
            if isinstance(stdout, list) and all(isinstance(x, str) for x in stdout):
                return "\n".join(stdout)

        # Some return results: [{ text: "..."}]
        results = result.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            text = results[0].get("text")
            if isinstance(text, str):
                return text
    return ""


async def _install_worker(
    sandbox_instance: Any,
    valid_packages: List[str],
    stream: EventStreamResponse,
) -> None:
    """
    Background task that performs the exact steps from the TS version:
      - start event
      - stop dev server
      - check installed packages (parse NEED_INSTALL)
      - npm install only needed ones (stream lines; handle ERESOLVE)
      - verify installed
      - (conditionally) restart dev server
      - complete
    """
    async def send_progress(data: Dict[str, Any]) -> None:
        await stream.send(data)

    try:
        # Start message
        await send_progress({
            "type": "start",
            "message": f"Installing {len(valid_packages)} package{'s' if len(valid_packages) > 1 else ''}...",
            "packages": valid_packages,
        })

        # Stop dev server
        await send_progress({"type": "status", "message": "Stopping development server..."})
        stop_server_code = """
import subprocess
import os
import signal

# Try to kill any existing Vite process
try:
    with open('/tmp/vite-process.pid', 'r') as f:
        pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        print("Stopped existing Vite process")
except:
    print("No existing Vite process found")
""".lstrip("\n")
        await _run_in_sandbox(stop_server_code)

        # Check installed packages
        await send_progress({"type": "status", "message": "Checking installed packages..."})
        check_code = f"""
import os
import json

os.chdir('/home/user/app')

# Read package.json to check installed packages
try:
    with open('package.json', 'r') as f:
        package_json = json.load(f)
    
    dependencies = package_json.get('dependencies', {{}})
    dev_dependencies = package_json.get('devDependencies', {{}})
    all_deps = {{**dependencies, **dev_dependencies}}
    
    # Check which packages need to be installed
    packages_to_check = {json.dumps(valid_packages)}
    already_installed = []
    need_install = []
    
    for pkg in packages_to_check:
        # Handle scoped packages
        if pkg.startswith('@'):
            pkg_name = pkg
        else:
            # Extract package name without version
            pkg_name = pkg.split('@')[0]
        
        if pkg_name in all_deps:
            already_installed.append(pkg_name)
        else:
            need_install.append(pkg)
    
    print(f"Already installed: {{already_installed}}")
    print(f"Need to install: {{need_install}}")
    print(f"NEED_INSTALL:{{json.dumps(need_install)}}")
    
except Exception as e:
    print(f"Error checking packages: {{e}}")
    print(f"NEED_INSTALL:{{json.dumps(packages_to_check)}}")
""".lstrip("\n")
        check_result = await _run_in_sandbox(check_code)

        # Parse NEED_INSTALL from output
        packages_to_install = list(valid_packages)
        output_text = _extract_output_text(check_result)
        for line in output_text.splitlines():
            if line.startswith("NEED_INSTALL:"):
                try:
                    packages_to_install = json.loads(line[len("NEED_INSTALL:"):])
                except Exception as e:
                    print("[install-packages] Failed to parse packages to install:", e)
                break

        if len(packages_to_install) == 0:
            await send_progress({
                "type": "success",
                "message": "All packages are already installed",
                "installedPackages": [],
                "alreadyInstalled": valid_packages,
            })
            await stream.close()
            return

        # Install only packages not already installed
        await send_progress({
            "type": "info",
            "message": f"Installing {len(packages_to_install)} new package(s): {', '.join(packages_to_install)}",
        })

        install_code = f"""
import subprocess
import os

os.chdir('/home/user/app')

# Run npm install with output capture
packages_to_install = {json.dumps(packages_to_install)}
cmd_args = ['npm', 'install', '--legacy-peer-deps'] + packages_to_install

print(f"Running command: {{' '.join(cmd_args)}}")

process = subprocess.Popen(
    cmd_args,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Stream output
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

# Get the return code
rc = process.poll()

# Capture any stderr
stderr = process.stderr.read()
if stderr:
    print("STDERR:", stderr)
    if 'ERESOLVE' in stderr:
        print("ERESOLVE_ERROR: Dependency conflict detected - using --legacy-peer-deps flag")

print(f"\\nInstallation completed with code: {{rc}}")

# Verify packages were installed
import json
with open('/home/user/app/package.json', 'r') as f:
    package_json = json.load(f)
    
installed = []
for pkg in {json.dumps(packages_to_install)}:
    if pkg in package_json.get('dependencies', {{}}):
        installed.append(pkg)
        print(f"✓ Verified {{pkg}}")
    else:
        print(f"✗ Package {{pkg}} not found in dependencies")
        
print(f"\\nVerified installed packages: {{installed}}")
""".lstrip("\n")

        install_result = await _run_in_sandbox(install_code, timeout=60000)
        raw_output = _extract_output_text(install_result)
        for line in [ln for ln in raw_output.splitlines() if ln.strip()]:
            if "STDERR:" in line:
                error_msg = line.replace("STDERR:", "").strip()
                if error_msg and error_msg != "undefined":
                    await send_progress({"type": "error", "message": error_msg})
            elif "ERESOLVE_ERROR:" in line:
                msg = line.replace("ERESOLVE_ERROR:", "").strip()
                await send_progress({
                    "type": "warning",
                    "message": f"Dependency conflict resolved with --legacy-peer-deps: {msg}"
                })
            elif "npm WARN" in line:
                await send_progress({"type": "warning", "message": line})
            elif line.strip() and "undefined" not in line:
                await send_progress({"type": "output", "message": line})

        # Extract verified installed packages
        installed_packages: List[str] = []
        m = re.search(r"Verified installed packages: \[(.*?)\]", raw_output, re.DOTALL)
        if m and m.group(1):
            installed_packages = [
                p.strip().replace("'", "")
                for p in m.group(1).split(",")
                if p.strip().replace("'", "")
            ]

        if installed_packages:
            await send_progress({
                "type": "success",
                "message": f"Successfully installed: {', '.join(installed_packages)}",
                "installedPackages": installed_packages,
            })
        else:
            await send_progress({
                "type": "error",
                "message": "Failed to verify package installation",
            })

        # Restart Vite dev server
        await send_progress({"type": "status", "message": "Restarting development server..."})
        restart_code = """
import subprocess
import os
import time

os.chdir('/home/user/app')

# Kill any existing Vite processes
subprocess.run(['pkill', '-f', 'vite'], capture_output=True)
time.sleep(1)

# Start Vite dev server
env = os.environ.copy()
env['FORCE_COLOR'] = '0'

process = subprocess.Popen(
    ['npm', 'run', 'dev'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env
)

print(f'✓ Vite dev server restarted with PID: {process.pid}')

# Store process info for later
with open('/tmp/vite-process.pid', 'w') as f:
    f.write(str(process.pid))

# Wait a bit for Vite to start up
time.sleep(3)

# Touch files to trigger Vite reload
subprocess.run(['touch', '/home/user/app/package.json'])
subprocess.run(['touch', '/home/user/app/vite.config.js'])

print("Vite restarted and should now recognize all packages")
""".lstrip("\n")
        await _run_in_sandbox(restart_code)

        await send_progress({
            "type": "complete",
            "message": "Package installation complete and dev server restarted!",
            "installedPackages": installed_packages,
        })

    except Exception as e:
        msg = str(e)
        if msg and msg != "undefined":
            await send_progress({"type": "error", "message": msg})
    finally:
        await stream.close()


async def POST(request: Any) -> Any:
    """
    Python equivalent of Next.js POST handler from install_packages.ts.
    Returns an EventStreamResponse (SSE-like). Consume it by iterating `async for chunk in resp`.
    """
    try:
        # Parse JSON body like NextRequest.json()
        if hasattr(request, "json"):
            body = await request.json()  # supports request objects with async .json()
        elif isinstance(request, dict):
            body = request
        else:
            body = {}

        packages = body.get("packages")
        sandbox_id = body.get("sandboxId")

        # Validate packages array
        if not packages or not isinstance(packages, list) or len(packages) == 0:
            # TS: 400 + json({ success:false, error: 'Packages array is required' })
            return {
                "success": False,
                "error": "Packages array is required",
                "status": 400,
            }

        # Validate and deduplicate package names
        seen = set()
        valid_packages: List[str] = []
        for pkg in packages:
            if pkg and isinstance(pkg, str):
                trimmed = pkg.strip()
                if trimmed and trimmed not in seen:
                    seen.add(trimmed)
                    valid_packages.append(trimmed)

        if len(valid_packages) == 0:
            return {
                "success": False,
                "error": "No valid package names provided",
                "status": 400,
            }

        # Log if duplicates were found (stdout logs, like console.log in TS)
        if len(packages) != len(valid_packages):
            print(f"[install-packages] Cleaned packages: removed {len(packages) - len(valid_packages)} invalid/duplicate entries")
            print(f"[install-packages] Original: {packages}")
            print(f"[install-packages] Cleaned: {valid_packages}")

        # Try to get sandbox - either from global or reconnect
        sandbox = active_sandbox

        if not sandbox and sandbox_id:
            print(f"[install-packages] Reconnecting to sandbox {sandbox_id}...")
            if E2BSandbox is None:
                print("[install-packages] e2b Sandbox library not available; cannot reconnect")
                return {
                    "success": False,
                    "error": "Failed to reconnect to sandbox: E2B Sandbox library not available",
                    "status": 500,
                }
            try:
                api_key = os.getenv("E2B_API_KEY")
                sandbox = await E2BSandbox.connect(sandbox_id, api_key=api_key)  # type: ignore[attr-defined]
                globals()["active_sandbox"] = sandbox
                print(f"[install-packages] Successfully reconnected to sandbox {sandbox_id}")
            except Exception as e:
                print(f"[install-packages] Failed to reconnect to sandbox:", e)
                return {
                    "success": False,
                    "error": f"Failed to reconnect to sandbox: {str(e)}",
                    "status": 500,
                }

        if not sandbox:
            return {
                "success": False,
                "error": "No active sandbox available",
                "status": 400,
            }

        print("[install-packages] Installing packages:", packages)

        # Create SSE-like response and start background worker
        stream = EventStreamResponse()
        asyncio.create_task(_install_worker(sandbox, valid_packages, stream))
        return stream  # Caller can: async for chunk in stream: ...

    except Exception as error:
        print("[install-packages] Error:", error)
        return {
            "success": False,
            "error": str(error),
            "status": 500,
        }
