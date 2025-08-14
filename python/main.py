# main.py — FastAPI launcher for all 20 route modules under ./routes
from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Any, Dict
import importlib.util
import os
import sys
from pathlib import Path
import uvicorn
import inspect

# -------------------------------------------------------------------
# Resolve project paths
# -------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
ROUTES_DIR = ROOT / "routes"
if not ROUTES_DIR.exists():
    # Fallback for notebook/preview environments
    ROUTES_DIR = Path("/mnt/data/routes")

# Make root importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------------
# Dynamic import for files (supports hyphenated filenames)
# -------------------------------------------------------------------
def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# -------------------------------------------------------------------
# Load ALL modules explicitly (works even if routes/ is not a package)
# -------------------------------------------------------------------
MODULES: Dict[str, Any] = {}

def _load_all():
    table = [
        # original & added
        ("appy_ai_code_stream",        "appy-ai-code-stream.py"),
        ("analyze_edit_intent",        "analyze-edit-intent.py"),
        ("create_ai_sandbox",          "create-ai-sandbox.py"),
        ("detect_and_install_packages","detect-and-install-packages.py"),
        ("get_sandbox_files",          "get-sanbox-files.py"),
        ("install_packages",           "install-packages.py"),
        ("kill_sandbox",               "kill-sandbox.py"),
        ("restart_vite",               "restart-vite.py"),
        ("run_command",                "run-command.py"),
        ("sandbox_logs",               "sandbox-logs.py"),
        ("sandbox_status",             "sandbox-status.py"),
        ("scrape_screenshot",          "scrape_screenshot.py"),
        ("scrape_url_enhanced",        "scrape_url_enhanced.py"),
        ("conversation_state",         "conversation_state.py"),
        ("generate_ai_stream",         "generate_ai_stream.py"),
        ("check_vite_errors",          "check_vite_errors.py"),
        ("clear_vite_errors_cache",    "clear_vite_errors_cache.py"),
        ("create_zip",                 "create_zip.py"),
        ("monitor_vite_logs",          "monitor_vite_logs.py"),
        ("report_vite_error",          "report_vite_error.py"),
    ]
    for alias, fname in table:
        MODULES[alias] = import_module_from_path(alias, ROUTES_DIR / fname)

    # Optional: prefer generate_code_ai.py if you dropped it in
    opt = ROUTES_DIR / "generate_code_ai.py"
    if opt.exists():
        MODULES["generate_code_ai"] = import_module_from_path("generate_code_ai", opt)

_load_all()

# Which globals we want to keep in sync across modules
SHARED_ATTRS = ("active_sandbox", "sandbox_state", "sandbox_data", "existing_files")

async def sync_globals():
    """Copy sandbox-related globals from create_ai_sandbox -> all other modules."""
    src = MODULES.get("create_ai_sandbox")
    if not src:
        return
    for attr in SHARED_ATTRS:
        val = getattr(src, attr, None)
        for mod in MODULES.values():
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, val)
                except Exception:
                    pass

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
async def maybe_await(value: Any) -> Any:
    """Await value if it's awaitable; else return as-is."""
    if inspect.isawaitable(value):
        return await value
    return value

async def call0(fn):
    """Call a 0-arg function, sync or async."""
    return await maybe_await(fn())

async def call1(fn, arg):
    """Call a 1-arg function, sync or async."""
    return await maybe_await(fn(arg))

def as_json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(content=data, status_code=status)

# -------------------------------------------------------------------
# Global app + lifespan
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Python Backend starting…")
    yield
    # Best-effort sandbox cleanup
    sb = getattr(MODULES.get("create_ai_sandbox"), "active_sandbox", None)
    if sb:
        try:
            killer = getattr(sb, "kill", None) or getattr(sb, "close", None)
            if killer:
                await maybe_await(killer())
            print("✅ Sandbox closed")
        except Exception as e:
            print(f"⚠️ Sandbox cleanup error: {e}")

app = FastAPI(
    title="Design Assist Python Backend",
    description="Python backend integrating 20 route modules",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://your-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Health / Root
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "routes_loaded": True}

@app.get("/")
async def root():
    return {
        "message": "Design Assist Python Backend",
        "endpoints": {
            "generate_code": "/api/generate-ai-code-stream",
            "create_sandbox": "/api/create-ai-sandbox",
            "get_files": "/api/get-sandbox-files",
            "scrape_url": "/api/scrape-url-enhanced",
            "install_packages": "/api/install-packages",
        },
    }

# -------------------------------------------------------------------
# AI Code Generation (non-stream simple response)
# -------------------------------------------------------------------
@app.post("/api/generate-ai-code-stream")
async def api_generate_ai_code_stream(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        model = body.get("model", "openai/gpt-4.1")
        context = body.get("context", {}) or {}
        is_edit = body.get("isEdit", False)

        # Ensure sandbox_state is synced BEFORE codegen
        await sync_globals()

        # Prefer generate_code_ai if present; else fallback to generate_ai_stream
        gen_mod = MODULES.get("generate_code_ai") or MODULES["generate_ai_stream"]

        # If your gen module expects to access shared globals, copy them in:
        for attr in SHARED_ATTRS:
            if hasattr(gen_mod, attr) and hasattr(MODULES["create_ai_sandbox"], attr):
                try:
                    setattr(gen_mod, attr, getattr(MODULES["create_ai_sandbox"], attr))
                except Exception:
                    pass

        if hasattr(gen_mod, "generate_code"):
            res = gen_mod.generate_code(prompt, model, context, is_edit)
            res = await maybe_await(res)
            return as_json(res)
        elif hasattr(gen_mod, "POST"):
            # Fallback: POST-style API
            res = await call1(gen_mod.POST, {"prompt": prompt, "model": model, "context": context, "isEdit": is_edit})
            return res if isinstance(res, (JSONResponse, StreamingResponse)) else as_json(res)
        else:
            raise RuntimeError("Generator module has no generate_code or POST")
    except Exception as e:
        print(f"[generate-ai-code-stream] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Sandbox Management
# -------------------------------------------------------------------
@app.post("/api/create-ai-sandbox")
async def api_create_ai_sandbox():
    try:
        res = await call0(MODULES["create_ai_sandbox"].POST)
        await sync_globals()
        return as_json(res)
    except Exception as e:
        print(f"[create-ai-sandbox] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-sandbox-files")
async def api_get_sandbox_files():
    try:
        await sync_globals()
        res = await call0(MODULES["get_sandbox_files"].GET)
        return as_json(res)
    except Exception as e:
        print(f"[get-sandbox-files] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sandbox-status")
async def api_sandbox_status():
    try:
        fn = MODULES["sandbox_status"].get_sandbox_status
        res = await maybe_await(fn())
        return as_json(res)
    except Exception as e:
        print(f"[sandbox-status] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kill-sandbox")
async def api_kill_sandbox():
    try:
        res = await call0(MODULES["kill_sandbox"].POST)
        return as_json(res)
    except Exception as e:
        print(f"[kill-sandbox] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Apply Code (placeholder if appy-ai-code-stream is WIP)
# -------------------------------------------------------------------
@app.post("/api/apply-ai-code-stream")
async def api_apply_ai_code_stream(_: Request):
    try:
        return as_json({"success": True, "message": "Code application endpoint working - implementation in progress"})
    except Exception as e:
        print(f"[apply-ai-code-stream] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Edit Intent Analysis
# -------------------------------------------------------------------
@app.post("/api/analyze-edit-intent")
async def api_analyze_edit_intent(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        manifest = body.get("manifest", {})
        model = body.get("model", "openai/gpt-4.1")
        fn = MODULES["analyze_edit_intent"].analyze_edit_intent
        res = await maybe_await(fn(prompt, manifest, model))
        return as_json(res)
    except Exception as e:
        print(f"[analyze-edit-intent] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Vite Error Management
# -------------------------------------------------------------------
@app.get("/api/check-vite-errors")
async def api_check_vite_errors():
    try:
        fn = MODULES["check_vite_errors"].GET
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return as_json(res)
    except Exception as e:
        print(f"[check-vite-errors] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-vite-errors-cache")
async def api_clear_vite_errors_cache():
    try:
        fn = MODULES["clear_vite_errors_cache"].POST
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return as_json(res)
    except Exception as e:
        print(f"[clear-vite-errors-cache] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitor-vite-logs")
async def api_monitor_vite_logs():
    try:
        fn = MODULES["monitor_vite_logs"].GET
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return as_json(res)
    except Exception as e:
        print(f"[monitor-vite-logs] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/report-vite-error")
async def api_report_vite_error(request: Request):
    try:
        body = await request.json()
        res = await call1(MODULES["report_vite_error"].POST, body)
        return as_json(res)
    except Exception as e:
        print(f"[report-vite-error] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Conversation State
# -------------------------------------------------------------------
@app.get("/api/conversation-state")
async def api_conversation_state_get():
    try:
        fn = MODULES["conversation_state"].GET
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return as_json(res)
    except Exception as e:
        print(f"[conversation-state-get] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation-state")
async def api_conversation_state_post(request: Request):
    try:
        body = await request.json()
        res = await call1(MODULES["conversation_state"].POST, body)
        return as_json(res)
    except Exception as e:
        print(f"[conversation-state-post] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversation-state")
async def api_conversation_state_delete():
    try:
        fn = MODULES["conversation_state"].DELETE
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return as_json(res)
    except Exception as e:
        print(f"[conversation-state-delete] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Package Management (streaming + non-stream)
# -------------------------------------------------------------------
@app.post("/api/install-packages")
async def api_install_packages(request: Request):
    try:
        res = await call1(MODULES["install_packages"].POST, request)
        return res  # already a Response (StreamingResponse/JSONResponse)
    except Exception as e:
        print(f"[install-packages] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-and-install-packages")
async def api_detect_and_install_packages(request: Request):
    try:
        res = await call1(MODULES["detect_and_install_packages"].POST, request)
        return res if isinstance(res, (StreamingResponse, JSONResponse)) else as_json(res)
    except Exception as e:
        print(f"[detect-and-install-packages] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Sandbox Ops
# -------------------------------------------------------------------
@app.post("/api/create-zip")
async def api_create_zip():
    try:
        fn = MODULES["create_zip"].POST
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return as_json(res)
    except Exception as e:
        print(f"[create-zip] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/restart-vite")
async def api_restart_vite():
    try:
        res = await call0(MODULES["restart_vite"].POST)
        return as_json(res)
    except Exception as e:
        print(f"[restart-vite] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-command")
async def api_run_command(request: Request):
    try:
        await sync_globals()
        res = await call1(MODULES["run_command"].POST, request)
        return res if isinstance(res, (StreamingResponse, JSONResponse)) else as_json(res)
    except Exception as e:
        print(f"[run-command] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sandbox-logs")
async def api_sandbox_logs():
    try:
        fn = MODULES["sandbox_logs"].GET
        res = await call0(fn) if len(inspect.signature(fn).parameters) == 0 else await call1(fn, {})
        return res if isinstance(res, (StreamingResponse, JSONResponse)) else as_json(res)
    except Exception as e:
        print(f"[sandbox-logs] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Web Scraping
# -------------------------------------------------------------------
@app.post("/api/scrape-screenshot")
async def api_scrape_screenshot(request: Request):
    try:
        res = await call1(MODULES["scrape_screenshot"].POST, request)
        return res if isinstance(res, (StreamingResponse, JSONResponse)) else as_json(res)
    except Exception as e:
        print(f"[scrape-screenshot] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scrape-url-enhanced")
async def api_scrape_url_enhanced(request: Request):
    try:
        res = await call1(MODULES["scrape_url_enhanced"].POST, request)
        return res if isinstance(res, (StreamingResponse, JSONResponse)) else as_json(res)
    except Exception as e:
        print(f"[scrape-url-enhanced] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Global error handler
# -------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[GLOBAL ERROR] {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "path": str(request.url)},
    )

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 Starting on http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")
