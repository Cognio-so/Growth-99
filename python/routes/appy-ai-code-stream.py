import os
import re
import json
import asyncio
from typing import Dict, List, Set, Optional, Any, Union
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_structured_chat_agent
from langchain.agents.agent import AgentExecutor
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# FastAPI and Streaming
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# LangSmith for logging/tracing
from langsmith import Client
import langsmith

# Global state (equivalent to global variables in the original code)
class GlobalState:
    conversation_state: Optional[Dict] = None
    active_sandbox: Any = None
    existing_files: Set[str] = set()
    sandbox_state: Dict = {}
    sandbox_data: Dict = {}

global_state = GlobalState()

# Models for request/response
class ParsedResponse(BaseModel):
    explanation: str = ""
    template: str = ""
    files: List[Dict[str, str]] = []
    packages: List[str] = []
    commands: List[str] = []
    structure: Optional[str] = None

class ApplyCodeRequest(BaseModel):
    response: str
    isEdit: bool = False
    packages: List[str] = []
    sandboxId: Optional[str] = None

class FileInfo(BaseModel):
    path: str
    content: str

# Sandbox implementation (simplified for this example)
class Sandbox:
    def __init__(self, sandbox_id: Optional[str] = None):
        self.sandbox_id = sandbox_id or f"sandbox-{datetime.now().timestamp()}"
        self.api_key = os.getenv("E2B_API_KEY", "")
    
    @staticmethod
    async def connect(sandbox_id: str, api_key: str):
        sandbox = Sandbox(sandbox_id)
        # In a real implementation, would connect to an existing sandbox
        return sandbox
    
    async def run_code(self, code: str):
        # Simulated code execution (in real implementation, would run in sandbox)
        print(f"[Sandbox] Running code: {code[:100]}...")
        # Extract the path from the code using regex
        path_match = re.search(r'"(.*?)"', code)
        if path_match:
            path = path_match.group(1)
            print(f"[Sandbox] Would write to: {path}")
        return {"success": True}
    
    async def commands_run(self, cmd: str, **kwargs):
        # Simulated command execution
        print(f"[Sandbox] Running command: {cmd}")
        return {"exitCode": 0}

# Functions for AI response parsing
def extract_packages_from_code(content: str) -> List[str]:
    """Extract package names from Python import statements."""
    packages = []
    # Match Python imports (both import X and from X import Y)
    import_regex = r'(?:from|import)\s+([a-zA-Z0-9_]+)(?:\.|.*?)'
    for match in re.finditer(import_regex, content):
        package_name = match.group(1)
        # Skip relative imports and built-ins
        if (
            not package_name.startswith('.') and 
            package_name not in ['os', 'sys', 'typing', 're', 'json', 'datetime'] and
            not package_name.startswith('__')
        ):
            if package_name not in packages:
                packages.append(package_name)
                print(f"[apply-ai-code-stream] 📦 Package detected from imports: {package_name}")
    
    return packages

def parse_ai_response(response: str) -> ParsedResponse:
    """Parse AI response to extract files, packages, commands, etc."""
    sections = ParsedResponse()
    
    # Parse file sections with <file> tags
    file_map = {}
    file_regex = r'<file path="([^"]+)">([\s\S]*?)(?:<\/file>|$)'
    for match in re.finditer(file_regex, response):
        file_path = match.group(1)
        content = match.group(2).strip()
        has_closing_tag = '</file>' in response[match.start():match.start() + len(match.group(0))]
        
        # Handle duplicate/incomplete files similar to TS version
        if file_path not in file_map:
            file_map[file_path] = {"content": content, "is_complete": has_closing_tag}
        elif not file_map[file_path]["is_complete"] and has_closing_tag:
            # Replace incomplete with complete
            print(f"[apply-ai-code-stream] Replacing incomplete {file_path} with complete version")
            file_map[file_path] = {"content": content, "is_complete": has_closing_tag}
        elif (
            file_map[file_path]["is_complete"] and 
            has_closing_tag and 
            len(content) > len(file_map[file_path]["content"])
        ):
            # Replace with longer complete version
            print(f"[apply-ai-code-stream] Replacing {file_path} with longer complete version")
            file_map[file_path] = {"content": content, "is_complete": has_closing_tag}
    
    # Convert map to array for sections.files
    for path, file_info in file_map.items():
        if not file_info["is_complete"]:
            print(f"[apply-ai-code-stream] Warning: File {path} appears to be truncated (no closing tag)")
        
        sections.files.append({"path": path, "content": file_info["content"]})
        
        # Extract packages from file content
        file_packages = extract_packages_from_code(file_info["content"])
        for pkg in file_packages:
            if pkg not in sections.packages:
                sections.packages.append(pkg)
    
    # Parse markdown code blocks with file paths
    markdown_file_regex = r'```(?:file )?path="([^"]+)"\n([\s\S]*?)```'
    for match in re.finditer(markdown_file_regex, response):
        file_path = match.group(1)
        content = match.group(2).strip()
        sections.files.append({"path": file_path, "content": content})
        
        # Extract packages from file content
        file_packages = extract_packages_from_code(content)
        for pkg in file_packages:
            if pkg not in sections.packages:
                sections.packages.append(pkg)
    
    # Parse code blocks
    code_block_regex = r'```(?:python)?\n([\s\S]*?)```'
    for match in re.finditer(code_block_regex, response):
        content = match.group(1).strip()
        # Try to detect file name from comments
        file_name_match = re.search(r'#\s*(?:File:|Component:)\s*([^\n]+)', content)
        if file_name_match:
            file_name = file_name_match.group(1).strip()
            file_path = file_name if '/' in file_name else f"src/{file_name}"
            
            # Don't add duplicate files
            if not any(f["path"] == file_path for f in sections.files):
                sections.files.append({"path": file_path, "content": content})
                
                # Extract packages
                file_packages = extract_packages_from_code(content)
                for pkg in file_packages:
                    if pkg not in sections.packages:
                        sections.packages.append(pkg)
    
    # Parse commands
    cmd_regex = r'<command>(.*?)<\/command>'
    for match in re.finditer(cmd_regex, response):
        sections.commands.append(match.group(1).strip())
    
    # Parse packages - both <package> tags and <packages> block
    pkg_regex = r'<package>(.*?)<\/package>'
    for match in re.finditer(pkg_regex, response):
        sections.packages.append(match.group(1).strip())
    
    packages_regex = r'<packages>([\s\S]*?)<\/packages>'
    packages_match = re.search(packages_regex, response)
    if packages_match:
        packages_content = packages_match.group(1).strip()
        packages_list = re.split(r'[\n,]+', packages_content)
        for pkg in packages_list:
            pkg = pkg.strip()
            if pkg and pkg not in sections.packages:
                sections.packages.append(pkg)
    
    # Parse structure, explanation, and template
    structure_match = re.search(r'<structure>([\s\S]*?)<\/structure>', response)
    if structure_match:
        sections.structure = structure_match.group(1).strip()
    
    explanation_match = re.search(r'<explanation>([\s\S]*?)<\/explanation>', response)
    if explanation_match:
        sections.explanation = explanation_match.group(1).strip()
    
    template_match = re.search(r'<template>(.*?)<\/template>', response)
    if template_match:
        sections.template = template_match.group(1).strip()
    
    return sections

# LangGraph Agent tools
@tool
def create_file(file_path: str, content: str) -> str:
    """Create or update a file with the given content."""
    # In a real implementation, this would write to the filesystem
    # or execute in a sandbox
    is_update = file_path in global_state.existing_files
    
    # Normalize path
    normalized_path = file_path
    if normalized_path.startswith('/'):
        normalized_path = normalized_path[1:]
    if (not normalized_path.startswith('src/') and 
        not normalized_path.startsWith('public/') and
        normalized_path != 'index.html'):
        normalized_path = f"src/{normalized_path}"
    
    # Simulated file creation
    global_state.existing_files.add(normalized_path)
    
    action = "updated" if is_update else "created"
    return f"Successfully {action} file: {normalized_path}"

@tool
def install_packages(packages: List[str]) -> str:
    """Install Python packages."""
    # In a real implementation, would use pip or equivalent
    if not packages:
        return "No packages to install"
    
    # Add standard RAG packages as per user rules
    if any(p in ["langchain", "chromadb", "faiss"] for p in packages):
        if "langchain" not in packages:
            packages.append("langchain")
        if "langsmith" not in packages:
            packages.append("langsmith")
    
    # Filter out standard modules and duplicates
    unique_packages = list(set(packages))
    unique_packages = [p for p in unique_packages if p and p.strip()]
    
    return f"Successfully installed packages: {', '.join(unique_packages)}"

@tool
def run_command(command: str) -> str:
    """Run a command in the sandbox."""
    # In a real implementation, would execute in sandbox
    return f"Command executed successfully: {command}"

# LangGraph agent state
class AgentState(BaseModel):
    parsed_response: ParsedResponse = Field(default_factory=ParsedResponse)
    packages_to_install: List[str] = Field(default_factory=list)
    files_created: List[str] = Field(default_factory=list)
    files_updated: List[str] = Field(default_factory=list)
    commands_executed: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    current_step: str = "parse"
    progress_messages: List[Dict] = Field(default_factory=list)

# LangGraph agent workflow
def create_agent_workflow():
    # Define agent nodes
    tools = [create_file, install_packages, run_command]
    tool_executor = ToolNode (tools)
    
    # Create the agent for executing the plan
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an AI code processing agent. 
Your job is to execute the parsed AI response.
You will be given parsed data containing files, packages, and commands.
Process them in the right order: install packages, create/update files, run commands."""),
        HumanMessage(content="{input}"),
    ])
    
    # LangGraph graph
    workflow = StateGraph(AgentState)
    
    # Define steps
    def should_install_packages(state: AgentState) -> str:
        if state.parsed_response.packages and state.current_step == "parse":
            state.current_step = "install_packages"
            state.progress_messages.append({
                "type": "step",
                "step": 1,
                "message": f"Installing {len(state.parsed_response.packages)} packages..."
            })
            return "install_packages"
        else:
            state.current_step = "create_files"
            state.progress_messages.append({
                "type": "step",
                "step": 2,
                "message": f"Creating {len(state.parsed_response.files)} files..."
            })
            return "create_files"
    
    def install_packages_step(state: AgentState) -> AgentState:
        try:
            packages = state.parsed_response.packages
            # Always include langchain and langsmith for RAG applications
            if any(p.lower() in ["langchain", "chroma", "faiss", "embedding"] for p in packages):
                for pkg in ["langchain", "langsmith"]:
                    if pkg not in packages:
                        packages.append(pkg)
            
            if packages:
                result = install_packages(packages)
                state.progress_messages.append({
                    "type": "package-progress",
                    "status": "complete",
                    "message": result
                })
            
            state.current_step = "create_files"
        except Exception as e:
            state.errors.append(f"Failed to install packages: {str(e)}")
            state.progress_messages.append({
                "type": "warning",
                "message": f"Package installation failed: {str(e)}"
            })
            state.current_step = "create_files"
        
        state.progress_messages.append({
            "type": "step",
            "step": 2,
            "message": f"Creating {len(state.parsed_response.files)} files..."
        })
        return state
    
    def create_files_step(state: AgentState) -> AgentState:
        files = state.parsed_response.files
        # Filter out config files
        config_files = ['tailwind.config.js', 'vite.config.js', 'package.json', 'package-lock.json', 'tsconfig.json', 'postcss.config.js']
        filtered_files = [
            file for file in files 
            if not any(file["path"].endswith(cf) for cf in config_files)
        ]
        
        for i, file in enumerate(filtered_files):
            try:
                state.progress_messages.append({
                    "type": "file-progress",
                    "current": i + 1,
                    "total": len(filtered_files),
                    "fileName": file["path"],
                    "action": "creating"
                })
                
                result = create_file(file["path"], file["content"])
                
                is_update = file["path"] in global_state.existing_files
                if is_update:
                    state.files_updated.append(file["path"])
                else:
                    state.files_created.append(file["path"])
                
                state.progress_messages.append({
                    "type": "file-complete",
                    "fileName": file["path"],
                    "action": "updated" if is_update else "created"
                })
            except Exception as e:
                state.errors.append(f"Failed to create {file['path']}: {str(e)}")
                state.progress_messages.append({
                    "type": "file-error",
                    "fileName": file["path"],
                    "error": str(e)
                })
        
        if state.parsed_response.commands:
            state.current_step = "run_commands"
            state.progress_messages.append({
                "type": "step",
                "step": 3,
                "message": f"Executing {len(state.parsed_response.commands)} commands..."
            })
        else:
            state.current_step = "complete"
            state.progress_messages.append({
                "type": "complete",
                "results": {
                    "filesCreated": state.files_created,
                    "filesUpdated": state.files_updated,
                    "packagesInstalled": state.parsed_response.packages,
                    "commandsExecuted": state.commands_executed,
                    "errors": state.errors
                },
                "explanation": state.parsed_response.explanation,
                "structure": state.parsed_response.structure,
                "message": f"Successfully applied {len(state.files_created)} files"
            })
        
        return state
    
    def run_commands_step(state: AgentState) -> AgentState:
        commands = state.parsed_response.commands
        for i, cmd in enumerate(commands):
            try:
                state.progress_messages.append({
                    "type": "command-progress",
                    "current": i + 1,
                    "total": len(commands),
                    "command": cmd,
                    "action": "executing"
                })
                
                result = run_command(cmd)
                state.commands_executed.append(cmd)
                
                state.progress_messages.append({
                    "type": "command-complete",
                    "command": cmd,
                    "exitCode": 0,
                    "success": True
                })
            except Exception as e:
                state.errors.append(f"Failed to execute {cmd}: {str(e)}")
                state.progress_messages.append({
                    "type": "command-error",
                    "command": cmd,
                    "error": str(e)
                })
        
        state.current_step = "complete"
        state.progress_messages.append({
            "type": "complete",
            "results": {
                "filesCreated": state.files_created,
                "filesUpdated": state.files_updated,
                "packagesInstalled": state.parsed_response.packages,
                "commandsExecuted": state.commands_executed,
                "errors": state.errors
            },
            "explanation": state.parsed_response.explanation,
            "structure": state.parsed_response.structure,
            "message": f"Successfully applied {len(state.files_created)} files"
        })
        
        return state
    
    # Add nodes to the graph
    workflow.add_node("parse", lambda state: state)
    workflow.add_node("install_packages", install_packages_step)
    workflow.add_node("create_files", create_files_step)
    workflow.add_node("run_commands", run_commands_step)
    
    # Add edges
    workflow.add_conditional_edges(
        "parse",
        should_install_packages,
        {
            "install_packages": "install_packages",
            "create_files": "create_files",
        }
    )
    workflow.add_edge("install_packages", "create_files")
    workflow.add_conditional_edges(
        "create_files",
        lambda state: "run_commands" if state.current_step == "run_commands" else "end",
        {
            "run_commands": "run_commands",
            "end": END,
        }
    )
    workflow.add_edge("run_commands", END)
    
    # Set the entry point
    workflow.set_entry_point("parse")
    
    return workflow.compile()

# FastAPI application
app = FastAPI()

# Stream response generator
async def stream_progress(progress_messages):
    for message in progress_messages:
        yield f"data: {json.dumps(message)}\n\n"
        await asyncio.sleep(0.1)  # Small delay to simulate streaming

@app.post("/api/apply-ai-code")
async def apply_ai_code(request: ApplyCodeRequest):
    try:
        response_text = request.response
        
        if not response_text:
            raise HTTPException(status_code=400, detail="response is required")
        
        # Debug log
        print("[apply-ai-code-stream] Received response to parse:")
        print(f"[apply-ai-code-stream] Response length: {len(response_text)}")
        print(f"[apply-ai-code-stream] Response preview: {response_text[:500]}")
        print(f"[apply-ai-code-stream] isEdit: {request.isEdit}")
        print(f"[apply-ai-code-stream] packages: {request.packages}")
        
        # Parse the AI response
        parsed = parse_ai_response(response_text)
        
        # Log what was parsed
        print("[apply-ai-code-stream] Parsed result:")
        print(f"[apply-ai-code-stream] Files found: {len(parsed.files)}")
        if parsed.files:
            for f in parsed.files:
                print(f"[apply-ai-code-stream] - {f['path']} ({len(f['content'])} chars)")
        print(f"[apply-ai-code-stream] Packages found: {parsed.packages}")
        
        # Create and run the agent workflow
        workflow = create_agent_workflow()
        
        # Initialize the agent state with parsed response
        initial_state = AgentState(
            parsed_response=parsed,
            packages_to_install=request.packages,
            progress_messages=[{
                "type": "start",
                "message": "Starting code application...",
                "totalSteps": 3
            }]
        )
        
        # Execute the workflow
        result_state = workflow.invoke(initial_state)
        
        # Return streaming response
        return StreamingResponse(
            stream_progress(result_state.progress_messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception as e:
        print(f"Apply AI code stream error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI code: {str(e)}")

