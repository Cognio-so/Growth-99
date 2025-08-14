# app.py — Streamlit UI for your routes/main_app.py
# Place in project root (next to main_app.py, .env, requirements.txt)
from __future__ import annotations
import os
import json
import asyncio
import traceback
import streamlit as st

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure we import from project root
import sys
from pathlib import Path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your in-process router
import main_app  # this is the file we built earlier

st.set_page_config(page_title="URL → Scrape → Codegen", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def call_api(method: str, path: str, body: dict | None = None):
    """Sync wrapper around the async main_app.handle_request."""
    try:
        return asyncio.run(main_app.handle_request(method, path, body or {}))
    except Exception as e:
        return {"success": False, "error": str(e), "trace": traceback.format_exc()}

def _ensure_sandbox():
    """Create sandbox if needed and return latest status."""
    resp = call_api("POST", "/api/create-sandbox")
    # if already created earlier, many implementations will still return success or a url
    return resp

def _get_sandbox_files():
    return call_api("GET", "/api/sandbox/files")

def _detect_and_install(files: dict):
    # detect-and-install expects {"files": {...}}
    return call_api("POST", "/api/detect-and-install", {"files": files})

def _restart_vite():
    return call_api("POST", "/api/vite/restart")

def _sandbox_status():
    return call_api("GET", "/api/sandbox/status")

def _sandbox_logs():
    return call_api("GET", "/api/sandbox/logs")

# ----------------------------
# Session state
# ----------------------------
if "sandbox_url" not in st.session_state:
    st.session_state.sandbox_url = None
if "last_files" not in st.session_state:
    st.session_state.last_files = None
if "last_codegen" not in st.session_state:
    st.session_state.last_codegen = None
if "last_scrape" not in st.session_state:
    st.session_state.last_scrape = None

# ----------------------------
# Sidebar: environment & controls
# ----------------------------
st.sidebar.title("Environment")
fc_key = os.getenv("FIRECRAWL_API_KEY")
e2b_key = os.getenv("E2B_API_KEY")

st.sidebar.write("**FIRECRAWL_API_KEY**:", "✅ set" if fc_key else "❌ missing")
st.sidebar.write("**E2B_API_KEY**:", "✅ set" if e2b_key else "❌ missing")
st.sidebar.caption("Set keys in your shell or .env file.")

with st.sidebar.expander("Sandbox controls"):
    if st.button("Create / Ensure Sandbox", use_container_width=True):
        resp = _ensure_sandbox()
        if isinstance(resp, dict) and resp.get("success"):
            st.success("Sandbox ready.")
            st.session_state.sandbox_url = resp.get("url", st.session_state.sandbox_url)
        else:
            st.error(f"Failed: {resp}")

    if st.button("Check Sandbox Status", use_container_width=True):
        st.json(_sandbox_status())

    if st.button("View Sandbox Logs", use_container_width=True):
        st.json(_sandbox_logs())

    if st.button("Restart Vite", use_container_width=True):
        st.json(_restart_vite())

    if st.session_state.sandbox_url:
        st.link_button("Open Hosted App", st.session_state.sandbox_url, use_container_width=True)

# ----------------------------
# Main: Build from URL
# ----------------------------
st.title("Build from URL → Scrape → Codegen → Host")

url = st.text_input("Enter a URL to scrape and generate code from", placeholder="https://example.com")
colA, colB = st.columns([1, 1])

with colA:
    if st.button("Scrape & Generate", type="primary"):
        with st.spinner("Ensuring sandbox, scraping, and generating code..."):
            # Hits our orchestration route: /api/build-from-url
            resp = call_api("POST", "/api/build-from-url", {"url": url})
            if not isinstance(resp, dict) or not resp.get("success"):
                st.error(f"Build failed: {resp}")
            else:
                st.session_state.last_scrape = resp.get("scrape")
                st.session_state.last_codegen = resp.get("codegen")
                # Often the sandbox URL is available after creation; fetch again just in case
                status = _ensure_sandbox()
                st.session_state.sandbox_url = status.get("url", st.session_state.sandbox_url)
                st.success("Done!")

with colB:
    if st.button("Get Sandbox Files"):
        resp = _get_sandbox_files()
        if isinstance(resp, dict) and resp.get("success"):
            st.session_state.last_files = resp
            st.success(f"Fetched {resp.get('fileCount', 0)} files.")
        else:
            st.error(f"Failed to fetch files: {resp}")

# Results
if st.session_state.last_scrape:
    st.subheader("Scrape Summary")
    st.json(st.session_state.last_scrape)

if st.session_state.last_codegen is not None:
    st.subheader("Generated Output")
    # Codegen might be text or dict; display appropriately
    if isinstance(st.session_state.last_codegen, dict):
        st.json(st.session_state.last_codegen)
    else:
        st.code(str(st.session_state.last_codegen), language="markdown")

# ----------------------------
# Files & Package Utilities
# ----------------------------
st.markdown("---")
st.header("Sandbox Files & Packages")

if st.session_state.last_files:
    mf = st.session_state.last_files
    st.subheader("Directory Structure")
    st.code(mf.get("structure") or "", language="text")

    with st.expander("File Manifest (summary)"):
        manifest = mf.get("manifest", {})
        # show a compact view
        compact = {
            "entryPoint": manifest.get("entryPoint"),
            "styleFiles": manifest.get("styleFiles"),
            "routes": manifest.get("routes"),
            "timestamp": manifest.get("timestamp"),
            "fileCount": st.session_state.last_files.get("fileCount"),
        }
        st.json(compact)

    with st.expander("All Files (path → first 400 chars)"):
        files = mf.get("files", {})
        for rel, content in list(files.items()):
            st.write(f"**{rel}**")
            preview = content[:400] + ("..." if len(content) > 400 else "")
            st.code(preview, language="javascript" if rel.endswith((".js", ".jsx", ".ts", ".tsx")) else "text")

    st.write("")
    if st.button("Detect & Install Required Packages"):
        files = mf.get("files") or {}
        resp = _detect_and_install(files)
        st.json(resp)

# ----------------------------
# Hosting Link (Vite Dev)
# ----------------------------
st.markdown("---")
st.header("Hosting")
if st.session_state.sandbox_url:
    st.success("Your Vite dev app should be live at the link below.")
    st.link_button("Open Hosted App", st.session_state.sandbox_url)
else:
    st.info("Create/ensure the sandbox first, then you’ll see the hosted app URL here.")
