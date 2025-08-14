# routes/run-command.py
from __future__ import annotations
from typing import Any, Dict, Optional
import inspect
import json

# These are set/synced by main.py after /api/create-ai-sandbox
active_sandbox: Optional[Any] = None
sandbox_state: Optional[Dict[str, Any]] = None
sandbox_data: Optional[Dict[str, Any]] = None


async def _maybe_await(value: Any) -> Any:
    """Await if awaitable; otherwise return as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_exec(exec_or_dict: Any) -> Dict[str, Any]:
    """
    Normalize various SDK outputs to a common dict:
      {stdout: str, stderr: str, returncode: int}
    Supports:
      - dict results with 'output'/'stdout'/'logs'
      - Execution-like objects with .wait(), .logs/.stdout/.output, .returncode
    """
    # Dict shape (some SDKs already return a dict)
    if isinstance(exec_or_dict, dict):
        out = exec_or_dict.get("output") or exec_or_dict.get("stdout") or exec_or_dict.get("logs") or ""
        err = exec_or_dict.get("stderr") or exec_or_dict.get("error") or ""
        code = exec_or_dict.get("returncode") or exec_or_dict.get("exitCode") or 0
        try:
            code = int(code)
        except Exception:
            code = 0
        return {"stdout": str(out), "stderr": str(err), "returncode": code}

    # Execution-like object
    obj = exec_or_dict
    wait = getattr(obj, "wait", None) or getattr(obj, "wait_for_done", None)
    if callable(wait):
        try:
            wait()  # usually sync
        except TypeError:
            # rare async wait()
            import asyncio
            asyncio.get_event_loop().run_until_complete(wait())
        except Exception:
            pass

    out = getattr(obj, "output", None) or getattr(obj, "stdout", None) or getattr(obj, "logs", None) or ""
    err = getattr(obj, "stderr", None) or getattr(obj, "error", None) or ""
    code = getattr(obj, "returncode", None) or getattr(obj, "exit_code", None) or getattr(obj, "exitCode", None) or 0
    try:
        code = int(code)
    except Exception:
        code = 0
    return {"stdout": str(out or ""), "stderr": str(err or ""), "returncode": code}


async def POST(request: Any) -> Dict[str, Any]:
    """
    Execute a shell command inside the sandbox.

    Body (JSON):
      {
        "command": "bash -lc 'ls -laR /home/user/app | head -n 200'",
        "cwd": "/home/user/app"   # optional; defaults to /home/user/app
      }

    Returns:
      {
        "success": bool,
        "stdout": str,
        "stderr": str,
        "returncode": int
      }
    """
    # Parse JSON body
    try:
        if hasattr(request, "json"):
            body = await request.json()
        elif isinstance(request, dict):
            body = request
        else:
            body = {}
    except Exception:
        body = {}

    cmd = body.get("command") or ""
    cwd = body.get("cwd") or "/home/user/app"

    if not cmd or not isinstance(cmd, str):
        return {"success": False, "error": "Missing 'command' string", "status": 400}

    if active_sandbox is None:
        return {"success": False, "error": "No active sandbox", "status": 404}

    # Pick the correct sandbox run function across SDK variants
    run_fn = (
        getattr(active_sandbox, "run_code", None)
        or getattr(active_sandbox, "runCode", None)
        or getattr(active_sandbox, "run", None)
        or getattr(active_sandbox, "exec", None)
    )
    if not callable(run_fn):
        return {"success": False, "error": "Sandbox missing run_code/runCode/run/exec", "status": 500}

    # We send a tiny Python script into the sandbox that executes your shell command and prints JSON
    py = f"""
import subprocess, json, os
cmd = {json.dumps(cmd)}
cwd = {json.dumps(cwd)}
try:
    proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(json.dumps({{
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr
    }}))
except Exception as e:
    print(json.dumps({{"returncode": 1, "stdout": "", "stderr": str(e)}}))
"""

    # IMPORTANT: do NOT blindly `await` — some SDKs return Execution objects
    result = run_fn(py)
    result = await _maybe_await(result)

    norm = _normalize_exec(result)

    # If stdout contains the JSON we printed from the sandbox, parse it for exact fields
    parsed = None
    try:
        parsed = json.loads(norm.get("stdout", ""))
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and "returncode" in parsed:
        rc = int(parsed.get("returncode", 0))
        return {
            "success": rc == 0,
            "stdout": parsed.get("stdout", ""),
            "stderr": parsed.get("stderr", ""),
            "returncode": rc,
        }

    # Fallback to normalized fields
    return {
        "success": norm.get("returncode", 0) == 0,
        "stdout": norm.get("stdout", ""),
        "stderr": norm.get("stderr", ""),
        "returncode": norm.get("returncode", 0),
    }
