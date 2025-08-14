import os
import re
import json
import time
import asyncio
from typing import Dict, List, Optional, TypedDict, Union, Any, AsyncGenerator
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.smith import RunEvalConfig, run_on_dataset

# LLM Providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode


# LangSmith Tracing
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="Vamshi-test"

# Global state
load_dotenv()

app = FastAPI()

# Global state storage (equivalent to global variables in TypeScript)
class SandboxState:
    file_cache: Dict = {
        "files": {},
        "manifest": None,
        "lastSync": None
    }

class ConversationState:
    def __init__(self):
        self.conversation_id = f"conv-{int(time.time())}"
        self.started_at = int(time.time())
        self.last_updated = int(time.time())
        self.context = {
            "messages": [],
            "edits": [],
            "project_evolution": {"major_changes": []},
            "user_preferences": {}
        }

# Initialize global state
sandbox_state = SandboxState()
conversation_state = None

# Helper classes for type definitions
class ConversationMessage(TypedDict):
    id: str
    role: str
    content: str
    timestamp: int
    metadata: Dict

class ConversationEdit(TypedDict):
    timestamp: int
    user_request: str
    edit_type: str
    target_files: List[str]
    confidence: float
    outcome: str

class FileManifest(TypedDict):
    files: Dict
    directories: List[str]

class RequestModel(BaseModel):
    prompt: str
    model: str = "openai/gpt-4.1"
    context: Optional[Dict] = None
    is_edit: bool = False

class StreamProgressResponse(BaseModel):
    type: str
    message: Optional[str] = None
    text: Optional[str] = None
    raw: Optional[bool] = None
    name: Optional[str] = None
    path: Optional[str] = None
    index: Optional[int] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    generated_code: Optional[str] = None
    explanation: Optional[str] = None
    files: Optional[int] = None
    components: Optional[int] = None
    model: Optional[str] = None
    packages_to_install: Optional[List[str]] = None

# LLM initialization functions
def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="gpt-4.1"
    )

def get_anthropic():
    return ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="claude-3-opus-20240229"
    )

def get_google():
    return ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="gemini-1.5-pro"
    )

def get_groq():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="llama3-8b-8192"
    )

# Helper function to analyze user preferences from conversation history
def analyze_user_preferences(messages: List[ConversationMessage]) -> Dict:
    user_messages = [m for m in messages if m["role"] == "user"]
    patterns = []
    
    # Count edit-related keywords
    targeted_edit_count = 0
    comprehensive_edit_count = 0
    
    for msg in user_messages:
        content = msg["content"].lower()
        
        # Check for targeted edit patterns
        if re.search(r'\b(update|change|fix|modify|edit|remove|delete)\s+(\w+\s+)?(\w+)\b', content):
            targeted_edit_count += 1
        
        # Check for comprehensive edit patterns
        if re.search(r'\b(rebuild|recreate|redesign|overhaul|refactor)\b', content):
            comprehensive_edit_count += 1
        
        # Extract common request patterns
        if 'hero' in content: patterns.append('hero section edits')
        if 'header' in content: patterns.append('header modifications')
        if 'color' in content or 'style' in content: patterns.append('styling changes')
        if 'button' in content: patterns.append('button updates')
        if 'animation' in content: patterns.append('animation requests')
    
    return {
        "common_patterns": list(set(patterns))[:3],  # Top 3 unique patterns
        "preferred_edit_style": "targeted" if targeted_edit_count > comprehensive_edit_count else "comprehensive"
    }

# LangGraph agent state definition
class AgentState(TypedDict):
    prompt: str
    model: str
    context: Dict
    is_edit: bool
    conversation_history: List[Dict]
    edit_context: Optional[Dict]
    system_prompt: str
    full_prompt: str
    progress_callbacks: List
    generated_code: str
    packages_to_install: List[str]
    components_count: int
    files: List[Dict]
    explanation: str
    warnings: List[str]

# LangGraph agent nodes
def analyze_intent_node(state: AgentState) -> AgentState:
    """
    Analyzes the user's intent and determines what files to edit.
    """
    global sandbox_state
    
    prompt = state["prompt"]
    is_edit = state["is_edit"]
    context = state["context"]
    model_name = state["model"]
    
    # Send progress update
    send_progress(state["progress_callbacks"], {
        "type": "status", 
        "message": "🔍 Creating search plan..."
    })
    
    edit_context = None
    enhanced_system_prompt = ""
    
    if is_edit and sandbox_state.file_cache.get("manifest"):
        manifest = sandbox_state.file_cache["manifest"]
        file_contents = sandbox_state.file_cache["files"]
        
        try:
            from context_selector import select_files_for_edit
            from file_search_executor import execute_search_plan, format_search_results_for_ai, select_target_file
            
            # For this implementation, we'll use LangChain to analyze intent
            llm = get_openai()  # Use OpenAI for this specific task
            
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are an expert code analyst. Your job is to understand user requests for 
                code edits and determine which files need to be modified. Analyze the request carefully.
                
                Output a JSON object with the following structure:
                {
                    "searchTerms": ["term1", "term2"],  // Key terms to search for in the codebase
                    "editType": "UPDATE_COMPONENT",     // One of: UPDATE_COMPONENT, ADD_FEATURE, FIX_BUG, REFACTOR, DELETE_COMPONENT
                    "reasoning": "Explanation of why these files need to be modified"
                }
                """),
                ("human", f"""
                User request: "{prompt}"
                
                File manifest:
                {json.dumps(manifest, indent=2)}
                """)
            ])
            
            search_plan_chain = intent_prompt | llm | StrOutputParser()
            search_plan_result = search_plan_chain.invoke({})
            search_plan = json.loads(search_plan_result)
            
            # Execute the search plan
            search_execution = execute_search_plan(
                search_plan,
                {path: data["content"] for path, data in file_contents.items()}
            )
            
            if search_execution["success"] and len(search_execution["results"]) > 0:
                # Select target file
                target = select_target_file(search_execution["results"], search_plan["editType"])
                
                if target:
                    # Send progress update
                    send_progress(state["progress_callbacks"], {
                        "type": "status",
                        "message": f"✅ Found code in {target['filePath'].split('/')[-1]} at line {target['lineNumber']}"
                    })
                    
                    # Build enhanced context with search results
                    enhanced_system_prompt = f"""
{format_search_results_for_ai(search_execution["results"])}

SURGICAL EDIT INSTRUCTIONS:
You have been given the EXACT location of the code to edit.
- File: {target["filePath"]}
- Line: {target["lineNumber"]}
- Reason: {target["reason"]}

Make ONLY the change requested by the user. Do not modify any other code.
User request: "{prompt}"
"""
                    
                    # Set up edit context
                    edit_context = {
                        "primaryFiles": [target["filePath"]],
                        "contextFiles": [],
                        "systemPrompt": enhanced_system_prompt,
                        "editIntent": {
                            "type": search_plan["editType"],
                            "description": search_plan["reasoning"],
                            "targetFiles": [target["filePath"]],
                            "confidence": 0.95,
                            "searchTerms": search_plan["searchTerms"]
                        }
                    }
            else:
                # Search failed - fall back to old behavior
                print("Search found no results, falling back to broader context")
                send_progress(state["progress_callbacks"], {
                    "type": "status",
                    "message": "⚠️ Could not find exact match, using broader search..."
                })
                
                # Fall back to file selection
                edit_context = select_files_for_edit(prompt, manifest)
        except Exception as e:
            print(f"Error in agentic search workflow: {e}")
            send_progress(state["progress_callbacks"], {
                "type": "status",
                "message": "⚠️ Search workflow error, falling back to keyword method..."
            })
            
            # Fallback
            try:
                from context_selector import select_files_for_edit
                edit_context = select_files_for_edit(prompt, manifest)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
    
    # Update state with edit context
    state["edit_context"] = edit_context
    if edit_context and edit_context.get("systemPrompt"):
        state["system_prompt"] = edit_context["systemPrompt"]
    
    return state

def build_prompts_node(state: AgentState) -> AgentState:
    """
    Builds the system prompt and full prompt for the LLM.
    """
    global conversation_state
    
    prompt = state["prompt"]
    is_edit = state["is_edit"]
    context = state["context"]
    edit_context = state.get("edit_context")
    
    # Build conversation context for system prompt
    conversation_context = ""
    if conversation_state and len(conversation_state.context["messages"]) > 1:
        print("Building conversation context")
        print(f"Total messages: {len(conversation_state.context['messages'])}")
        print(f"Total edits: {len(conversation_state.context['edits'])}")
        
        conversation_context = "\n\n## Conversation History (Recent)\n"
        
        # Include only the last 3 edits to save context
        recent_edits = conversation_state.context["edits"][-3:]
        if len(recent_edits) > 0:
            print(f"Including {len(recent_edits)} recent edits in context")
            conversation_context += "\n### Recent Edits:\n"
            for edit in recent_edits:
                target_files = [f.split('/')[-1] for f in edit["target_files"]]
                conversation_context += f'- "{edit["user_request"]}" → {edit["edit_type"]} ({", ".join(target_files)})\n'
        
        # Include recently created files - CRITICAL for preventing duplicates
        recent_msgs = conversation_state.context["messages"][-5:]
        recently_created_files = []
        for msg in recent_msgs:
            if msg.get("metadata") and msg["metadata"].get("edited_files"):
                recently_created_files.extend(msg["metadata"]["edited_files"])
        
        if recently_created_files:
            unique_files = list(set(recently_created_files))
            conversation_context += "\n### 🚨 RECENTLY CREATED/EDITED FILES (DO NOT RECREATE THESE):\n"
            for file in unique_files:
                conversation_context += f"- {file}\n"
            conversation_context += "\nIf the user mentions any of these components, UPDATE the existing file!\n"
        
        # Include only last 5 messages for context
        if len(recent_msgs) > 2:  # More than just current message
            conversation_context += "\n### Recent Messages:\n"
            for msg in recent_msgs[:-1]:  # Exclude current message
                if msg["role"] == "user":
                    truncated_content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    conversation_context += f'- "{truncated_content}"\n'
        
        # Include only last 2 major changes
        major_changes = conversation_state.context["project_evolution"]["major_changes"][-2:]
        if major_changes:
            conversation_context += "\n### Recent Changes:\n"
            for change in major_changes:
                conversation_context += f"- {change['description']}\n"
        
        # Keep user preferences - they're concise
        user_prefs = analyze_user_preferences(conversation_state.context["messages"])
        if user_prefs["common_patterns"]:
            conversation_context += "\n### User Preferences:\n"
            conversation_context += f"- Edit style: {user_prefs['preferred_edit_style']}\n"
        
        # Limit total conversation context length
        if len(conversation_context) > 2000:
            conversation_context = conversation_context[:2000] + "\n[Context truncated to prevent length errors]"

    # Build the main system prompt
    system_prompt = f"""You are an expert React developer with perfect memory of the conversation. You maintain context across messages and remember scraped websites, generated components, and applied code. Generate clean, modern React code for Vite applications.
{conversation_context}

🚨 CRITICAL RULES - YOUR MOST IMPORTANT INSTRUCTIONS:
1. **DO EXACTLY WHAT IS ASKED - NOTHING MORE, NOTHING LESS**
   - Don't add features not requested
   - Don't fix unrelated issues
   - Don't improve things not mentioned
2. **CHECK App.jsx FIRST** - ALWAYS see what components exist before creating new ones
3. **NAVIGATION LIVES IN Header.jsx** - Don't create Nav.jsx if Header exists with nav
4. **USE STANDARD TAILWIND CLASSES ONLY**:
   - ✅ CORRECT: bg-white, text-black, bg-blue-500, bg-gray-100, text-gray-900
   - ❌ WRONG: bg-background, text-foreground, bg-primary, bg-muted, text-secondary
   - Use ONLY classes from the official Tailwind CSS documentation
5. **FILE COUNT LIMITS**:
   - Simple style/text change = 1 file ONLY
   - New component = 2 files MAX (component + parent)
   - If >3 files, YOU'RE DOING TOO MUCH

COMPONENT RELATIONSHIPS (CHECK THESE FIRST):
- Navigation usually lives INSIDE Header.jsx, not separate Nav.jsx
- Logo is typically in Header, not standalone
- Footer often contains nav links already
- Menu/Hamburger is part of Header, not separate

PACKAGE USAGE RULES:
- DO NOT use react-router-dom unless user explicitly asks for routing
- For simple nav links in a single-page app, use scroll-to-section or href="#"
- Only add routing if building a multi-page application
- Common packages are auto-installed from your imports

WEBSITE CLONING REQUIREMENTS:
When recreating/cloning a website, you MUST include:
1. **Header with Navigation** - Usually Header.jsx containing nav
2. **Hero Section** - The main landing area (Hero.jsx)
3. **Main Content Sections** - Features, Services, About, etc.
4. **Footer** - Contact info, links, copyright (Footer.jsx)
5. **App.jsx** - Main app component that imports and uses all components
"""

    # Add edit-specific instructions if in edit mode
    if is_edit:
        system_prompt += """
CRITICAL: THIS IS AN EDIT TO AN EXISTING APPLICATION

YOU MUST FOLLOW THESE EDIT RULES:
0. NEVER create tailwind.config.js, vite.config.js, package.json, or any other config files - they already exist!
1. DO NOT regenerate the entire application
2. DO NOT create files that already exist (like App.jsx, index.css, tailwind.config.js)
3. ONLY edit the EXACT files needed for the requested change - NO MORE, NO LESS
4. If the user says "update the header", ONLY edit the Header component - DO NOT touch Footer, Hero, or any other components
5. If the user says "change the color", ONLY edit the relevant style or component file - DO NOT "improve" other parts
6. If you're unsure which file to edit, choose the SINGLE most specific one related to the request
7. IMPORTANT: When adding new components or libraries:
   - Create the new component file
   - UPDATE ONLY the parent component that will use it
   - Example: Adding a Newsletter component means:
     * Create Newsletter.jsx
     * Update ONLY the file that will use it (e.g., Footer.jsx OR App.jsx) - NOT both
8. When adding npm packages:
   - Import them ONLY in the files where they're actually used
   - The system will auto-install missing packages

CRITICAL FILE MODIFICATION RULES - VIOLATION = FAILURE:
- **NEVER TRUNCATE FILES** - Always return COMPLETE files with ALL content
- **NO ELLIPSIS (...)** - Include every single line of code, no skipping
- Files MUST be complete and runnable - include ALL imports, functions, JSX, and closing tags
- Count the files you're about to generate
- If the user asked to change ONE thing, you should generate ONE file (or at most two if adding a new component)
- DO NOT "fix" or "improve" files that weren't mentioned in the request
- DO NOT update multiple components when only one was requested
- DO NOT add features the user didn't ask for
- RESIST the urge to be "helpful" by updating related files

CRITICAL: DO NOT REDESIGN OR REIMAGINE COMPONENTS
- "update" means make a small change, NOT redesign the entire component
- "change X to Y" means ONLY change X to Y, nothing else
- "fix" means repair what's broken, NOT rewrite everything
- "remove X" means delete X from the existing file, NOT create a new file
- "delete X" means remove X from where it currently exists
- Preserve ALL existing functionality and design unless explicitly asked to change it

NEVER CREATE NEW FILES WHEN THE USER ASKS TO REMOVE/DELETE SOMETHING
If the user says "remove X", you must:
1. Find which existing file contains X
2. Edit that file to remove X
3. DO NOT create any new files
"""

    # Add targeted edit mode instructions if we have edit context
    if edit_context:
        edit_intent = edit_context.get("editIntent", {})
        primary_files = edit_context.get("primaryFiles", [])
        
        system_prompt += f"""
TARGETED EDIT MODE ACTIVE
- Edit Type: {edit_intent.get("type", "UPDATE_COMPONENT")}
- Confidence: {edit_intent.get("confidence", 0.75)}
- Files to Edit: {', '.join(primary_files)}

🚨 CRITICAL RULE - VIOLATION WILL RESULT IN FAILURE 🚨
YOU MUST ***ONLY*** GENERATE THE FILES LISTED ABOVE!

ABSOLUTE REQUIREMENTS:
1. COUNT the files in "Files to Edit" - that's EXACTLY how many files you must generate
2. If "Files to Edit" shows ONE file, generate ONLY that ONE file
3. DO NOT generate App.jsx unless it's EXPLICITLY listed in "Files to Edit"
4. DO NOT generate ANY components that aren't listed in "Files to Edit"
5. DO NOT "helpfully" update related files
6. DO NOT fix unrelated issues you notice
7. DO NOT improve code quality in files not being edited
8. DO NOT add bonus features

EXAMPLE VIOLATIONS (THESE ARE FAILURES):
❌ User says "update the hero" → You update Hero, Header, Footer, and App.jsx
❌ User says "change header color" → You redesign the entire header
❌ User says "fix the button" → You update multiple components
❌ Files to Edit shows "Hero.jsx" → You also generate App.jsx "to integrate it"
❌ Files to Edit shows "Header.jsx" → You also update Footer.jsx "for consistency"

CORRECT BEHAVIOR (THIS IS SUCCESS):
✅ User says "update the hero" → You ONLY edit Hero.jsx with the requested change
✅ User says "change header color" → You ONLY change the color in Header.jsx
✅ User says "fix the button" → You ONLY fix the specific button issue
✅ Files to Edit shows "Hero.jsx" → You generate ONLY Hero.jsx
✅ Files to Edit shows "Header.jsx, Nav.jsx" → You generate EXACTLY 2 files: Header.jsx and Nav.jsx

THE AI INTENT ANALYZER HAS ALREADY DETERMINED THE FILES.
DO NOT SECOND-GUESS IT.
DO NOT ADD MORE FILES.
ONLY OUTPUT THE EXACT FILES LISTED IN "Files to Edit".
"""

    # Add styling rules and critical completion rules
    system_prompt += """
CRITICAL UI/UX RULES:
- NEVER use emojis in any code, text, console logs, or UI elements
- ALWAYS ensure responsive design using proper Tailwind classes (sm:, md:, lg:, xl:)
- ALWAYS use proper mobile-first responsive design patterns
- NEVER hardcode pixel widths - use relative units and responsive classes
- ALWAYS test that the layout works on mobile devices (320px and up)
- ALWAYS make sections full-width by default - avoid max-w-7xl or similar constraints
- For full-width layouts: use className="w-full" or no width constraint at all
- Only add max-width constraints when explicitly needed for readability (like blog posts)
- Prefer system fonts and clean typography
- Ensure all interactive elements have proper hover/focus states
- Use proper semantic HTML elements for accessibility

CRITICAL STYLING RULES - MUST FOLLOW:
- NEVER use inline styles with style={{ }} in JSX
- NEVER use <style jsx> tags or any CSS-in-JS solutions
- NEVER create App.css, Component.css, or any component-specific CSS files
- NEVER import './App.css' or any CSS files except index.css
- ALWAYS use Tailwind CSS classes for ALL styling
- ONLY create src/index.css with the @tailwind directives
- The ONLY CSS file should be src/index.css with:
  @tailwind base;
  @tailwind components;
  @tailwind utilities;

CRITICAL COMPLETION RULES:
1. NEVER say "I'll continue with the remaining components"
2. NEVER say "Would you like me to proceed?"
3. NEVER use <continue> tags
4. Generate ALL components in ONE response
5. If App.jsx imports 10 components, generate ALL 10
6. Complete EVERYTHING before ending your response

Use this XML format for React components only (DO NOT create tailwind.config.js - it already exists):

<file path="src/index.css">
@tailwind base;
@tailwind components;
@tailwind utilities;
</file>

<file path="src/App.jsx">
// Main App component that imports and uses other components
// Use Tailwind classes: className="min-h-screen bg-gray-50"
</file>

<file path="src/components/Example.jsx">
// Your React component code here
// Use Tailwind classes for ALL styling
</file>

🚨 CRITICAL CODE GENERATION RULES - VIOLATION = FAILURE 🚨:
1. NEVER truncate ANY code - ALWAYS write COMPLETE files
2. NEVER use "..." anywhere in your code - this causes syntax errors
3. NEVER cut off strings mid-sentence - COMPLETE every string
4. NEVER leave incomplete class names or attributes
5. ALWAYS close ALL tags, quotes, brackets, and parentheses
6. If you run out of space, prioritize completing the current file
"""

    # Build full prompt with context
    full_prompt = prompt
    if context:
        context_parts = []
        
        # Add sandbox ID if available
        if context.get("sandboxId"):
            context_parts.append(f"Current sandbox ID: {context['sandboxId']}")
        
        # Add file structure if available
        if context.get("structure"):
            context_parts.append(f"Current file structure:\n{context['structure']}")
        
        # Include current file contents (from backend cache or frontend)
        backend_files = sandbox_state.file_cache.get("files", {})
        has_backend_files = len(backend_files) > 0
        
        if has_backend_files:
            if edit_context and len(edit_context.get("primaryFiles", [])) > 0:
                context_parts.append("\nEXISTING APPLICATION - TARGETED EDIT MODE")
                
                # Format files for AI
                primary_files_content = {}
                context_files_content = {}
                
                # Get contents of primary files
                for file_path in edit_context["primaryFiles"]:
                    normalized_path = file_path.replace('/home/user/app/', '')
                    if normalized_path in backend_files:
                        primary_files_content[normalized_path] = backend_files[normalized_path]["content"]
                
                # Get contents of context files
                for file_path in edit_context.get("contextFiles", []):
                    normalized_path = file_path.replace('/home/user/app/', '')
                    if normalized_path in backend_files:
                        context_files_content[normalized_path] = backend_files[normalized_path]["content"]
                
                # Format files for AI
                formatted_files = ""
                
                if primary_files_content:
                    formatted_files += "\n## FILES TO EDIT:\n"
                    for file_path, content in primary_files_content.items():
                        formatted_files += f"\n<file path=\"{file_path}\">\n{content}\n</file>\n"
                
                if context_files_content:
                    formatted_files += "\n## CONTEXT FILES (For Reference Only):\n"
                    for file_path, content in context_files_content.items():
                        formatted_files += f"\n<file path=\"{file_path}\">\n{content}\n</file>\n"
                
                context_parts.append(formatted_files)
                context_parts.append("\nIMPORTANT: Only modify the files listed under \"Files to Edit\". The context files are provided for reference only.")
            else:
                # Fallback to showing all files if no edit context
                context_parts.append("\nEXISTING APPLICATION - TARGETED EDIT REQUIRED")
                context_parts.append("\nYou MUST analyze the user request and determine which specific file(s) to edit.")
                context_parts.append("\nCurrent project files (DO NOT regenerate all of these):")
                
                # Show file list first for reference
                context_parts.append("\n### File List:")
                for file_path in backend_files:
                    context_parts.append(f"- {file_path}")
                
                # Include top files as context in fallback mode
                context_parts.append("\n### File Contents (ALL FILES FOR CONTEXT):")
                for file_path, file_data in backend_files.items():
                    if "content" in file_data and isinstance(file_data["content"], str):
                        context_parts.append(f"\n<file path=\"{file_path}\">\n{file_data['content']}\n</file>")
        
        # Construct the full prompt
        if context_parts:
            full_prompt = f"CONTEXT:\n{chr(10).join(context_parts)}\n\nUSER REQUEST:\n{prompt}"
    
    # Update state with system and full prompts
    state["system_prompt"] = system_prompt
    state["full_prompt"] = full_prompt
    
    send_progress(state["progress_callbacks"], {
        "type": "status", 
        "message": "Planning application structure..."
    })
    
    return state

def code_generation_node(state: AgentState) -> AgentState:
    """
    Generates code using the LLM based on the system prompt and user request.
    """
    print("\nStarting code generation...")
    
    # Initialize results
    packages_to_install = []
    current_file = ""
    current_file_path = ""
    component_count = 0
    files = []
    generated_code = ""
    
    # Setup model
    model_name = state["model"]
    
    # Determine which provider to use based on model
    is_anthropic = model_name.startswith("anthropic/")
    is_google = model_name.startswith("google/")
    is_openai = model_name.startswith("openai/")
    
    if is_anthropic:
        llm = get_anthropic()
    elif is_google:
        llm = get_google()
    elif is_openai:
        llm = get_openai()
    else:
        llm = get_groq()
    
    # Set up streaming
    cb_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm.streaming = True
    llm.callbacks = cb_manager
    
    # Create the generation prompt
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", state["system_prompt"]),
        ("human", state["full_prompt"])
    ])
    
    # Create the LangChain chain
    chain = generation_prompt | llm | StrOutputParser()
    
    # Stream the response
    buffer = []
    file_buffer = []
    in_file = False
    file_regex = r'<file path="([^"]+)">([\s\S]*?)<\/file>'
    
    # Process the streaming output
    for chunk in chain.stream({}):
        # Add chunk to buffer
        buffer.append(chunk)
        generated_code += chunk
        
        # Check for file boundaries
        if '<file path="' in chunk and not in_file:
            in_file = True
            file_buffer = [chunk]
        elif in_file:
            file_buffer.append(chunk)
            
            # Check if we've completed a file
            if '</file>' in chunk:
                in_file = False
                file_content = ''.join(file_buffer)
                
                # Try to extract file path and content
                file_match = re.search(file_regex, file_content)
                if file_match:
                    file_path = file_match.group(1)
                    content = file_match.group(2)
                    files.append({"path": file_path, "content": content})
                    
                    # Count components
                    if "components/" in file_path:
                        component_count += 1
                    
                    # Send progress update for the file
                    if "components/" in file_path:
                        component_name = file_path.split('/')[-1].replace('.jsx', '').replace('.js', '')
                        send_progress(state["progress_callbacks"], {
                            "type": "component",
                            "name": component_name,
                            "path": file_path,
                            "index": component_count
                        })
                    elif "App.jsx" in file_path:
                        send_progress(state["progress_callbacks"], {
                            "type": "app",
                            "message": "Generated main App.jsx",
                            "path": file_path
                        })
                    
                    # Extract packages from content (in edit mode)
                    if state["is_edit"]:
                        # Package extraction regex for imports
                        import_regex = r'import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))*\s+from\s+)?[\'"]([^\'"]+)[\'"]'
                        for match in re.finditer(import_regex, content):
                            import_path = match.group(1)
                            # Skip relative imports and built-in React
                            if (not import_path.startswith('.') and 
                                not import_path.startswith('/') and
                                import_path != 'react' and 
                                import_path != 'react-dom' and
                                not import_path.startswith('@/')):
                                # Extract package name
                                if import_path.startswith('@'):
                                    package_name = '/'.join(import_path.split('/')[:2])
                                else:
                                    package_name = import_path.split('/')[0]
                                
                                if package_name not in packages_to_install:
                                    packages_to_install.append(package_name)
                                    send_progress(state["progress_callbacks"], {
                                        "type": "package",
                                        "name": package_name,
                                        "message": f"Package detected from imports: {package_name}"
                                    })
                
                # Clear buffer for next file
                file_buffer = []
        
        # Stream the raw text for live preview
        send_progress(state["progress_callbacks"], {
            "type": "stream",
            "text": chunk,
            "raw": True
        })
    
    # Extract explanation if present
    explanation_match = re.search(r'<explanation>([\s\S]*?)<\/explanation>', generated_code)
    explanation = explanation_match.group(1).strip() if explanation_match else "Code generated successfully!"
    
    # Check for truncation issues
    truncation_warnings = []
    
    # Check for unclosed file tags
    file_open_count = len(re.findall(r'<file path="', generated_code))
    file_close_count = len(re.findall(r'<\/file>', generated_code))
    
    if file_open_count != file_close_count:
        truncation_warnings.append(f"Unclosed file tags detected: {file_open_count} open, {file_close_count} closed")
    
    # Process package tags
    if state["is_edit"]:
        # Check for package tags
        package_regex = r'<package>([^<]+)<\/package>'
        for match in re.finditer(package_regex, generated_code):
            package_name = match.group(1).strip()
            if package_name and package_name not in packages_to_install:
                packages_to_install.append(package_name)
                send_progress(state["progress_callbacks"], {
                    "type": "package",
                    "name": package_name,
                    "message": f"Package detected: {package_name}"
                })
                
        # Check for packages tag (multiple packages)
        packages_regex = r'<packages>([\s\S]*?)<\/packages>'
        packages_match = re.search(packages_regex, generated_code)
        if packages_match:
            packages_content = packages_match.group(1).strip()
            packages_list = [pkg.strip() for pkg in re.split(r'[\n,]+', packages_content) if pkg.strip()]
            
            for package_name in packages_list:
                if package_name not in packages_to_install:
                    packages_to_install.append(package_name)
                    send_progress(state["progress_callbacks"], {
                        "type": "package",
                        "name": package_name,
                        "message": f"Package detected: {package_name}"
                    })
    
    # Send completion
    send_progress(state["progress_callbacks"], {
        "type": "complete",
        "generated_code": generated_code,
        "explanation": explanation,
        "files": len(files),
        "components": component_count,
        "model": state["model"],
        "packages_to_install": packages_to_install if packages_to_install else None,
        "warnings": truncation_warnings if truncation_warnings else None
    })
    
    # Update state with results
    state["generated_code"] = generated_code
    state["files"] = files
    state["packages_to_install"] = packages_to_install
    state["components_count"] = component_count
    state["explanation"] = explanation
    state["warnings"] = truncation_warnings
    
    # Track edit in conversation history if it was an edit
    global conversation_state
    edit_context = state.get("edit_context")
    
    if state["is_edit"] and edit_context and conversation_state:
        edit_intent = edit_context.get("editIntent", {})
        edit_record = {
            "timestamp": int(time.time()),
            "user_request": state["prompt"],
            "edit_type": edit_intent.get("type", "UPDATE_COMPONENT"),
            "target_files": edit_context.get("primaryFiles", []),
            "confidence": edit_intent.get("confidence", 0.75),
            "outcome": "success"  # Assuming success if we got here
        }
        
        conversation_state.context["edits"].append(edit_record)
        
        # Track major changes
        if edit_intent.get("type") == "ADD_FEATURE" or len(files) > 3:
            conversation_state.context["project_evolution"]["major_changes"].append({
                "timestamp": int(time.time()),
                "description": edit_intent.get("description", "Major update"),
                "files_affected": edit_context.get("primaryFiles", [])
            })
        
        # Update last updated timestamp
        conversation_state.last_updated = int(time.time())
    
    return state

def send_progress(callbacks, data):
    """
    Helper function to send progress updates through the streaming response.
    """
    for callback in callbacks:
        if asyncio.iscoroutinefunction(callback):
            asyncio.create_task(callback(data))
        else:
            callback(data)

def analyze_edit_intent(prompt, manifest, model="openai/gpt-4.1"):
    """
    Analyzes the user's edit intent to determine which files to modify.
    """
    try:
        # Use LangChain to analyze intent
        llm = get_openai()  # Use OpenAI for this task
        
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert code analyst. Your job is to understand user requests for 
            code edits and determine which files need to be modified. Analyze the request carefully.
            
            Output a JSON object with the following structure:
            {
                "searchTerms": ["term1", "term2"],  // Key terms to search for in the codebase
                "editType": "UPDATE_COMPONENT",     // One of: UPDATE_COMPONENT, ADD_FEATURE, FIX_BUG, REFACTOR, DELETE_COMPONENT
                "reasoning": "Explanation of why these files need to be modified"
            }
            """),
            ("human", f"""
            User request: "{prompt}"
            
            File manifest:
            {json.dumps(manifest, indent=2)}
            """)
        ])
        
        search_plan_chain = intent_prompt | llm | StrOutputParser()
        search_plan_result = search_plan_chain.invoke({})
        
        try:
            search_plan = json.loads(search_plan_result)
            return {"success": True, "searchPlan": search_plan}
        except json.JSONDecodeError:
            print(f"Error parsing search plan JSON: {search_plan_result}")
            return {"success": False, "error": "Failed to parse search plan"}
        
    except Exception as e:
        print(f"Error analyzing edit intent: {e}")
        return {"success": False, "error": str(e)}

def get_sandbox_files():
    """
    Returns the files from the sandbox state.
    """
    global sandbox_state
    
    try:
        if sandbox_state.file_cache.get("files"):
            return {
                "success": True,
                "files": {path: data["content"] for path, data in sandbox_state.file_cache["files"].items()},
                "manifest": sandbox_state.file_cache.get("manifest")
            }
        else:
            return {"success": False, "error": "No files in sandbox state"}
    except Exception as e:
        print(f"Error getting sandbox files: {e}")
        return {"success": False, "error": str(e)}

def generate_code(prompt, model="openai/gpt-4.1", context=None, is_edit=False):
    """
    Main function for generating code with AI.
    Replaces the FastAPI endpoint with a regular Python function.
    """
    global conversation_state
    
    try:
        # Use default empty context if None is provided
        if context is None:
            context = {}
        
        # Log request information
        print("Received request:")
        print(f"- prompt: {prompt}")
        print(f"- is_edit: {is_edit}")
        print(f"- context.sandboxId: {context.get('sandboxId')}")
        print(f"- context.currentFiles: {list(context.get('currentFiles', {}).keys()) if context.get('currentFiles') else 'none'}")
        print(f"- currentFiles count: {len(context.get('currentFiles', {}))}")
        
        # Initialize conversation state if not exists
        if not conversation_state:
            conversation_state = ConversationState()
        
        # Add user message to conversation history
        user_message = {
            "id": f"msg-{int(time.time())}",
            "role": "user",
            "content": prompt,
            "timestamp": int(time.time()),
            "metadata": {
                "sandboxId": context.get("sandboxId")
            }
        }
        conversation_state.context["messages"].append(user_message)
        
        # Clean up old messages to prevent unbounded growth
        if len(conversation_state.context["messages"]) > 20:
            # Keep only the last 15 messages
            conversation_state.context["messages"] = conversation_state.context["messages"][-15:]
            print("Trimmed conversation history to prevent context overflow")
        
        # Clean up old edits
        if len(conversation_state.context["edits"]) > 10:
            conversation_state.context["edits"] = conversation_state.context["edits"][-8:]
        
        # Show a sample of actual file content for debugging
        if context.get("currentFiles") and context["currentFiles"]:
            first_file = list(context["currentFiles"].items())[0]
            print(f"- sample file: {first_file[0]}")
            if isinstance(first_file[1], str):
                print(f"- sample content preview: {first_file[1][:100]}...")
            else:
                print("- sample content preview: not a string")
        
        if not prompt:
            return {"success": False, "error": "Prompt is required"}
        
        # List to store callbacks for progress updates
        progress_callbacks = []
        
        # Create a callback function that prints progress
        def add_progress_callback(data):
            print(f"Progress: {data}")
        
        # Add the callback to our list
        progress_callbacks.append(add_progress_callback)
        
        # Create initial agent state
        state = AgentState(
            prompt=prompt,
            model=model,
            context=context,
            is_edit=is_edit,
            conversation_history=conversation_state.context["messages"],
            edit_context=None,
            system_prompt="",
            full_prompt="",
            progress_callbacks=progress_callbacks,
            generated_code="",
            packages_to_install=[],
            components_count=0,
            files=[],
            explanation="",
            warnings=[]
        )
        
        # Build the agent graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_intent", analyze_intent_node)
        workflow.add_node("build_prompts", build_prompts_node)
        workflow.add_node("generate_code", code_generation_node)
        
        # Define the conditional entry point logic
        def should_analyze_intent(state: AgentState) -> str:
            """Determines whether to analyze intent or skip to code generation."""
            if state["is_edit"] and sandbox_state.file_cache.get("manifest"):
                return "analyze_intent"
            return "build_prompts"
        
        # Set the conditional entry point for the graph
        workflow.add_conditional_edges(
            START,
            should_analyze_intent,
            {
                "analyze_intent": "analyze_intent",
                "build_prompts": "build_prompts",
            }
        )
        
        # Add edges
        workflow.add_edge('analyze_intent', 'build_prompts')
        workflow.add_edge('build_prompts', 'generate_code')
        workflow.add_edge('generate_code', END)
        
        # Compile the graph
        agent = workflow.compile()
        
        # Execute the agent graph and capture the final state
        result = agent.invoke(state)
        return {
            "success": True,
            "generated_code": result["generated_code"],
            "files": result["files"],
            "explanation": result["explanation"],
            "packages_to_install": result["packages_to_install"]
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Example usage of the generate_code function
    result = generate_code(
        prompt="Create a simple React counter component using hooks",
        model="openai/gpt-4.1",
        is_edit=False
    )
    
    if result["success"]:
        print("\nGeneration successful!")
        print(f"Explanation: {result['explanation']}")
        print(f"Generated {len(result['files'])} files")
        
        # Print the generated files
        for file in result["files"]:
            print(f"\n--- {file['path']} ---")
            print(file["content"])
            print("---")
    else:
        print(f"Error: {result['error']}")
