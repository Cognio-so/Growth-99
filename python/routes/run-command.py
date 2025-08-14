# run_command.py

from typing import Any, Dict, Optional
import json

# Mirror the TS global: `global.activeSandbox`
active_sandbox: Optional[Any] = None


async def POST(request: Any) -> Dict[str, Any]:
    """
    Python equivalent of run-command.ts POST handler.
    - No web framework; callable directly from main_app.py
    - Same validations, logging, sandbox execution, and response shape
    """
    try:
        # Parse JSON body like NextRequest.json()
        if hasattr(request, "json"):
            body = await request.json()  # supports request objects with async .json()
        elif isinstance(request, dict):
            body = request  # allow passing a plain dict
        else:
            body = {}

        command = body.get("command")
        if not command:
            # TS returned 400; here we keep the same payload keys
            return {
                "success": False,
                "error": "Command is required",
            }

        if not active_sandbox:
            # TS returned 400; here we keep the same payload keys
            return {
                "success": False,
                "error": "No active sandbox",
            }

        print(f"[run-command] Executing: {command}")

        # Build the exact embedded Python script (matches TS version)
        # Note: JSON array syntax is also valid Python for literal lists.
        args_literal = json.dumps(command.split(" "))
        embedded_code = f"""
import subprocess
import os

os.chdir('/home/user/app')
result = subprocess.run({args_literal},
                       capture_output=True,
                       text=True,
                       shell=False)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("\\nSTDERR:")
    print(result.stderr)
print(f"\\nReturn code: {{result.returncode}}")
""".lstrip("\n")

        # Use LangChain Runnable if available; otherwise call sandbox directly
        result: Any
        try:
            from langchain_core.runnables import RunnableLambda  # type: ignore

            async def _runner(code: str) -> Any:
                # Prefer a Pythonic method name if provided; fallback to TS-style
                if hasattr(active_sandbox, "run_code"):
                    return await active_sandbox.run_code(code)
                return await active_sandbox.runCode(code)  # type: ignore[attr-defined]

            chain = RunnableLambda(_runner)
            result = await chain.ainvoke(embedded_code)
        except Exception:
            # Fallback: direct sandbox call without LangChain
            if hasattr(active_sandbox, "run_code"):
                result = await active_sandbox.run_code(embedded_code)
            else:
                result = await active_sandbox.runCode(embedded_code)  # type: ignore[attr-defined]

        # Extract output like in TS: result.logs.stdout.join('\n')
        output: Optional[str] = None
        if isinstance(result, dict):
            try:
                stdout_list = result["logs"]["stdout"]
                if isinstance(stdout_list, list):
                    output = "\n".join(stdout_list)
            except Exception:
                # Sometimes providers return a flat 'output'
                output = result.get("output")  # type: ignore[assignment]
        else:
            # Attribute-style access
            logs = getattr(result, "logs", None)
            if logs is not None:
                stdout_list = getattr(logs, "stdout", None)
                if isinstance(stdout_list, list):
                    output = "\n".join(stdout_list)
            if output is None:
                output = getattr(result, "output", None)

        return {
            "success": True,
            "output": output,
            "message": "Command executed successfully",
        }

    except Exception as error:
        print("[run-command] Error:", error)
        return {
            "success": False,
            "error": str(error),
        }
