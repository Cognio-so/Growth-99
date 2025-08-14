# route.py

from typing import Any, Optional

# Global variable to mirror `global.activeSandbox`
active_sandbox: Optional[Any] = None


async def POST() -> dict:
    """
    Equivalent of the Next.js route's POST() handler in Python.
    - No HTTP framework used (callable directly from main_app.py).
    - Preserves logic and messages.
    """
    try:
        if not active_sandbox:
            return {
                "success": False,
                "error": "No active sandbox"
            }

        print("[restart-vite] Forcing Vite restart...")

        # The exact Python script that was embedded in the TS file's template string
        embedded_code = """
import subprocess
import os
import signal
import time
import threading
import json
import sys

# Kill existing Vite process
try:
    with open('/tmp/vite-process.pid', 'r') as f:
        pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        print("Killed existing Vite process")
        time.sleep(1)
except:
    print("No existing Vite process found")

os.chdir('/home/user/app')

# Clear error file
error_file = '/tmp/vite-errors.json'
with open(error_file, 'w') as f:
    json.dump({"errors": [], "lastChecked": time.time()}, f)

# Function to monitor Vite output for errors
def monitor_output(proc, error_file):
    while True:
        line = proc.stderr.readline()
        if not line:
            break
        
        sys.stdout.write(line)  # Also print to console
        
        # Check for import resolution errors
        if "Failed to resolve import" in line:
            try:
                # Extract package name from error
                import_match = line.find('"')
                if import_match != -1:
                    end_match = line.find('"', import_match + 1)
                    if end_match != -1:
                        package_name = line[import_match + 1:end_match]
                        # Skip relative imports
                        if not package_name.startswith('.'):
                            with open(error_file, 'r') as f:
                                data = json.load(f)
                            
                            # Handle scoped packages correctly
                            if package_name.startswith('@'):
                                # For @scope/package, keep the scope
                                pkg_parts = package_name.split('/')
                                if len(pkg_parts) >= 2:
                                    final_package = '/'.join(pkg_parts[:2])
                                else:
                                    final_package = package_name
                            else:
                                # For regular packages, just take the first part
                                final_package = package_name.split('/')[0]
                            
                            error_obj = {
                                "type": "npm-missing",
                                "package": final_package,
                                "message": line.strip(),
                                "timestamp": time.time()
                            }
                            
                            # Avoid duplicates
                            if not any(e['package'] == error_obj['package'] for e in data['errors']):
                                data['errors'].append(error_obj)
                                
                            with open(error_file, 'w') as f:
                                json.dump(data, f)
                                
                            print(f"WARNING: Detected missing package: {error_obj['package']}")
            except Exception as e:
                print(f"Error parsing Vite error: {e}")

# Start Vite with error monitoring
process = subprocess.Popen(
    ['npm', 'run', 'dev'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_output, args=(process, error_file))
monitor_thread.daemon = True
monitor_thread.start()

print("Vite restarted successfully!")

# Store process info for later
with open('/tmp/vite-process.pid', 'w') as f:
    f.write(str(process.pid))

# Wait for Vite to fully start
time.sleep(5)
print("Vite is ready")
""".lstrip("\n")

        # Match the TS behavior: call the sandbox's code-execution method
        # Prefer a Pythonic `run_code`, but fall back to `runCode` if that's what's provided.
        if hasattr(active_sandbox, "run_code"):
            result = await active_sandbox.run_code(embedded_code)
        else:
            result = await active_sandbox.runCode(embedded_code)  # type: ignore[attr-defined]

        # Mirror `result.output` access
        output = result.get("output") if isinstance(result, dict) else getattr(result, "output", None)

        return {
            "success": True,
            "message": "Vite restarted successfully",
            "output": output
        }

    except Exception as error:
        print("[restart-vite] Error:", error)
        return {
            "success": False,
            "error": str(error)
        }
