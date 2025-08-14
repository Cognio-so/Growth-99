from datetime import datetime
from e2b_code_interpreter import Sandbox
import json

# Global variables (similar to those in TypeScript)
active_sandbox = None
sandbox_data = None
existing_files = set()

def get_sandbox_status():
    """
    Get the status of the sandbox, similar to the TypeScript GET route functionality.
    Returns JSON-serializable dictionary with sandbox status information.
    """
    try:
        # Check if sandbox exists
        sandbox_exists = active_sandbox is not None
        
        sandbox_healthy = False
        sandbox_info = None
        
        if sandbox_exists and active_sandbox:
            try:
                # Retrieve sandbox information
                info = active_sandbox.get_info()
                
                sandbox_healthy = True
                sandbox_info = {
                    "sandboxId": sandbox_data.get("sandboxId") if sandbox_data else None,
                    "url": sandbox_data.get("url") if sandbox_data else None,
                    "filesTracked": list(existing_files) if existing_files else [],
                    "lastHealthCheck": datetime.now().isoformat()
                }
            except Exception as error:
                print(f"[sandbox-status] Health check failed: {error}")
                sandbox_healthy = False
        
        return {
            "success": True,
            "active": sandbox_exists,
            "healthy": sandbox_healthy,
            "sandboxData": sandbox_info,
            "message": "Sandbox is active and healthy" if sandbox_healthy 
                      else "Sandbox exists but is not responding" if sandbox_exists 
                      else "No active sandbox"
        }
        
    except Exception as error:
        print(f"[sandbox-status] Error: {error}")
        return {
            "success": False,
            "active": False,
            "error": str(error)
        }
from dotenv import load_dotenv
import os
import re
import json
def initialize_sandbox(timeout_seconds=60):
    """
    Initialize a new sandbox instance
    """
    global active_sandbox, sandbox_data, existing_files
    
    try:
        # Create sandbox with specified timeout
        api_key=os.getenv("E2B_API_KEY")
        active_sandbox = Sandbox(api_key=api_key,timeout=timeout_seconds)
        
        # Get sandbox info
        info = active_sandbox.get_info()
        
        # Store sandbox data
        sandbox_data = {
            "sandboxId": info.sandbox_id,
            "templateId": info.template_id,
            "name": info.name,
            "startedAt": info.started_at.isoformat(),
            "endAt": info.end_at.isoformat(),
            "url": None  # The URL field might need to be populated from elsewhere
        }
        
        existing_files = set()
        
        return {
            "success": True,
            "message": "Sandbox initialized successfully",
            "sandboxInfo": sandbox_data
        }
        
    except Exception as error:
        print(f"[initialize-sandbox] Error: {error}")
        return {
            "success": False,
            "error": str(error)
        }

# Example usage
if __name__ == "__main__":
    # Initialize a sandbox
    init_result = initialize_sandbox()
    print(json.dumps(init_result, indent=2))
    
    if init_result["success"]:
        # Get status after initialization
        status = get_sandbox_status()
        print(json.dumps(status, indent=2))
