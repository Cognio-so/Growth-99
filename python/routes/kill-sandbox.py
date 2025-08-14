from typing import Any, Dict, Set, Optional

# Global variables to match TypeScript globals
active_sandbox: Optional[Any] = None
sandbox_data: Optional[Any] = None
existing_files: Set[str] = set()

async def POST() -> Dict[str, Any]:
    """Kill active sandbox - equivalent to POST function from TypeScript"""
    global active_sandbox, sandbox_data, existing_files
    
    try:
        print('[kill-sandbox] Killing active sandbox...')
        
        sandbox_killed = False
        
        # Kill existing sandbox if any
        if active_sandbox is not None:
            try:
                await active_sandbox.close()
                sandbox_killed = True
                print('[kill-sandbox] Sandbox closed successfully')
            except Exception as e:
                print(f'[kill-sandbox] Failed to close sandbox: {e}')
            
            active_sandbox = None
            sandbox_data = None
        
        # Clear existing files tracking
        if existing_files:
            existing_files.clear()
        
        return {
            "success": True,
            "sandboxKilled": sandbox_killed,
            "message": "Sandbox cleaned up successfully"
        }
        
    except Exception as error:
        print(f'[kill-sandbox] Error: {error}')
        return {
            "success": False,
            "error": str(error)
        }