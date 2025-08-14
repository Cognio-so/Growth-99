"""
create_ai_sandbox.py — Python equivalent of create_ai_sandbox.ts (POST handler)
- No web framework; callable directly from main_app.py
- Mirrors global.activeSandbox, sandboxData, existingFiles, sandboxState
- Uses LangChain RunnableLambda + a minimal LangGraph node to run code in the sandbox
- Preserves file contents, logging, and overall flow precisely
"""

from typing import Any, Dict, Optional, Set
import os
import asyncio
import inspect
from types import SimpleNamespace
import json

# --- LangChain / LangGraph (used without changing behavior) ---
try:
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
except Exception as _e:
    raise

# --- E2B Sandbox (python SDK) ---
try:
    from e2b import Sandbox as E2BSandbox  # type: ignore
except Exception:
    E2BSandbox = None  # We'll error nicely if it's not installed

# --- App config shim to mirror appConfig.e2b.* ---
# In TS: import { appConfig } from '@/config/app.config';
# Here we try to import similarly; otherwise use conservative defaults.
try:
    # If you have a Python equivalent, expose `appConfig.e2b.timeoutMinutes`, `timeoutMs`, `vitePort`, `viteStartupDelay`
    from config.app_config import appConfig  # type: ignore
except Exception:
    appConfig = SimpleNamespace(
        e2b=SimpleNamespace(
            timeoutMinutes=15,
            timeoutMs=15 * 60 * 1000,    # ms
            vitePort=5173,
            viteStartupDelay=4000,       # ms
        )
    )

# --- Globals mirroring the TS file ---
active_sandbox: Optional[Any] = None
sandbox_data: Optional[Dict[str, Any]] = None
existing_files: Set[str] = set()
sandbox_state: Optional[Dict[str, Any]] = None


# --- Helpers (LangChain + LangGraph) ---
async def _run_in_sandbox(sandbox: Any, code: str) -> Dict[str, Any]:
    """
    Run arbitrary code inside the sandbox using either .run_code or .runCode.
    Wrapped with LangChain RunnableLambda and dispatched via a one-node LangGraph.
    """
    async def _runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        c = payload.get("code", "")
        run = getattr(sandbox, "run_code", None) or getattr(sandbox, "runCode", None)
        if run is None:
            raise RuntimeError("Sandbox missing run_code/runCode")
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


def _get_output_text(result: Any) -> str:
    """Best-effort output extraction (some SDKs return {'output': '...'})"""
    if isinstance(result, dict):
        out = result.get("output")
        if isinstance(out, str):
            return out
    return ""


async def POST() -> Dict[str, Any]:
    """
    Python equivalent of the Next.js POST handler.
    Returns a dict mirroring the JSON structure from the TS version.
    """
    global active_sandbox, sandbox_data, existing_files, sandbox_state

    sandbox: Optional[Any] = None

    try:
        print("[create-ai-sandbox] Creating base sandbox...")

        # Kill existing sandbox if any
        if active_sandbox:
            print("[create-ai-sandbox] Killing existing sandbox...")
            try:
                killer = getattr(active_sandbox, "kill", None)
                if killer:
                    if inspect.iscoroutinefunction(killer):
                        await killer()
                    else:
                        killer()
            except Exception as e:
                print("Failed to close existing sandbox:", e)
            active_sandbox = None

        # Clear existing files tracking
        existing_files.clear()

        # Create base sandbox - we'll set up Vite ourselves for full control
        print(f"[create-ai-sandbox] Creating base E2B sandbox with {appConfig.e2b.timeoutMinutes} minute timeout...")
        if E2BSandbox is None:
            raise RuntimeError("E2B Sandbox library not available; install the 'e2b' Python package.")

        # Try to match TS: Sandbox.create({ apiKey, timeoutMs })
        create_kwargs = {
            "api_key": os.getenv("E2B_API_KEY"),
            "timeout_ms": appConfig.e2b.timeoutMs,
        }
        try:
            sandbox = await E2BSandbox.create(**create_kwargs)  # type: ignore[arg-type]
        except TypeError:
            # Some SDKs might expect 'timeoutMs'
            create_kwargs = {
                "apiKey": os.getenv("E2B_API_KEY"),
                "timeoutMs": appConfig.e2b.timeoutMs,
            }
            sandbox = await E2BSandbox.create(**create_kwargs)  # type: ignore[arg-type]

        sandbox_id = getattr(sandbox, "sandboxId", None) or str(int(asyncio.get_event_loop().time() * 1000))
        get_host = getattr(sandbox, "getHost", None)
        host = get_host(appConfig.e2b.vitePort) if callable(get_host) else f"localhost:{appConfig.e2b.vitePort}"

        print(f"[create-ai-sandbox] Sandbox created: {sandbox_id}")
        print(f"[create-ai-sandbox] Sandbox host: {host}")

        # Set up a basic Vite React app using Python to write files (exact script preserved)
        print("[create-ai-sandbox] Setting up Vite React app...")
        setup_script = '''
import os
import json

print('Setting up React app with Vite and Tailwind...')

# Create directory structure
os.makedirs('/home/user/app/src', exist_ok=True)

# Package.json
package_json = {
    "name": "sandbox-app",
    "version": "1.0.0",
    "type": "module",
    "scripts": {
        "dev": "vite --host",
        "build": "vite build",
        "preview": "vite preview"
    },
    "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0"
    },
    "devDependencies": {
        "@vitejs/plugin-react": "^4.0.0",
        "vite": "^4.3.9",
        "tailwindcss": "^3.3.0",
        "postcss": "^8.4.31",
        "autoprefixer": "^10.4.16"
    }
}

with open('/home/user/app/package.json', 'w') as f:
    json.dump(package_json, f, indent=2)
print('✓ package.json')

# Vite config for E2B - with allowedHosts
vite_config = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// E2B-compatible Vite configuration
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    hmr: false,
    allowedHosts: ['.e2b.app', 'localhost', '127.0.0.1']
  }
})"""

with open('/home/user/app/vite.config.js', 'w') as f:
    f.write(vite_config)
print('✓ vite.config.js')

# Tailwind config - standard without custom design tokens
tailwind_config = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}"""

with open('/home/user/app/tailwind.config.js', 'w') as f:
    f.write(tailwind_config)
print('✓ tailwind.config.js')

# PostCSS config
postcss_config = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}"""

with open('/home/user/app/postcss.config.js', 'w') as f:
    f.write(postcss_config)
print('✓ postcss.config.js')

# Index.html
index_html = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sandbox App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>"""

with open('/home/user/app/index.html', 'w') as f:
    f.write(index_html)
print('✓ index.html')

# Main.jsx
main_jsx = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)"""

with open('/home/user/app/src/main.jsx', 'w') as f:
    f.write(main_jsx)
print('✓ src/main.jsx')

# App.jsx with explicit Tailwind test
app_jsx = """function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center p-4">
      <div className="text-center max-w-2xl">
        <p className="text-lg text-gray-400">
          Sandbox Ready<br/>
          Start building your React app with Vite and Tailwind CSS!
        </p>
      </div>
    </div>
  )
}

export default App"""

with open('/home/user/app/src/App.jsx', 'w') as f:
    f.write(app_jsx)
print('✓ src/App.jsx')

# Index.css with explicit Tailwind directives
index_css = """@tailwind base;
@tailwind components;
@tailwind utilities;

/* Force Tailwind to load */
@layer base {
  :root {
    font-synthesis: none;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    -webkit-text-size-adjust: 100%;
  }
  
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  background-color: rgb(17 24 39);
}"""

with open('/home/user/app/src/index.css', 'w') as f:
  f.write(index_css)
print('✓ src/index.css')

print('\\nAll files created successfully!')
'''

        # Execute the setup script
        await _run_in_sandbox(sandbox, setup_script)

        # Install dependencies (exact block preserved)
        print("[create-ai-sandbox] Installing dependencies...")
        install_block = """
import subprocess

print('Installing npm packages...')
result = subprocess.run(
    ['npm', 'install'],
    cwd='/home/user/app',
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print('✓ Dependencies installed successfully')
else:
    print(f'⚠ Warning: npm install had issues: {result.stderr}')
    # Continue anyway as it might still work
"""
        await _run_in_sandbox(sandbox, install_block)

        # Start Vite dev server (exact block preserved)
        print("[create-ai-sandbox] Starting Vite dev server...")
        start_vite_block = """
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

print(f'✓ Vite dev server started with PID: {process.pid}')
print('Waiting for server to be ready...')
"""
        await _run_in_sandbox(sandbox, start_vite_block)

        # Wait for Vite to be fully ready (convert ms -> seconds)
        await asyncio.sleep(float(appConfig.e2b.viteStartupDelay) / 1000.0)

        # Force Tailwind CSS to rebuild (exact block preserved)
        force_tailwind_block = """
import os
import time

# Touch the CSS file to trigger rebuild
css_file = '/home/user/app/src/index.css'
if os.path.exists(css_file):
    os.utime(css_file, None)
    print('✓ Triggered CSS rebuild')
    
# Also ensure PostCSS processes it
time.sleep(2)
print('✓ Tailwind CSS should be loaded')
"""
        await _run_in_sandbox(sandbox, force_tailwind_block)

        # Store sandbox globally
        active_sandbox = sandbox
        sandbox_data = {
            "sandboxId": sandbox_id,
            "url": f"https://{host}",
        }

        # Set extended timeout if supported
        set_timeout = getattr(sandbox, "setTimeout", None) or getattr(sandbox, "set_timeout", None)
        if callable(set_timeout):
            try:
                set_timeout(appConfig.e2b.timeoutMs)
                print(f"[create-ai-sandbox] Set sandbox timeout to {appConfig.e2b.timeoutMinutes} minutes")
            except Exception:
                pass

        # Initialize sandbox state
        sandbox_state = {
            "fileCache": {
                "files": {},
                "lastSync": int(asyncio.get_event_loop().time() * 1000),
                "sandboxId": sandbox_id,
            },
            "sandbox": sandbox,
            "sandboxData": {
                "sandboxId": sandbox_id,
                "url": f"https://{host}",
            },
        }

        # Track initial files
        for path in [
            "src/App.jsx", "src/main.jsx", "src/index.css",
            "index.html", "package.json", "vite.config.js",
            "tailwind.config.js", "postcss.config.js",
        ]:
            existing_files.add(path)

        print("[create-ai-sandbox] Sandbox ready at:", f"https://{host}")

        return {
            "success": True,
            "sandboxId": sandbox_id,
            "url": f"https://{host}",
            "message": "Sandbox created and Vite React app initialized",
        }

    except Exception as error:
        print("[create-ai-sandbox] Error:", error)

        # Clean up on error
        if sandbox:
            try:
                killer = getattr(sandbox, "kill", None)
                if killer:
                    if inspect.iscoroutinefunction(killer):
                        await killer()
                    else:
                        killer()
            except Exception as e:
                print("Failed to close sandbox on error:", e)

        # Use traceback if available for richer details
        try:
            import traceback
            details = traceback.format_exc()
        except Exception:
            details = None

        return {
            "error": str(error) if isinstance(error, Exception) else "Failed to create sandbox",
            "details": details,
            "status": 500,
        }
