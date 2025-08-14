# routes/generate_code_ai.py
from __future__ import annotations

import os
import re
import json
import time
import asyncio
from typing import Dict, List, Optional, TypedDict, Any

from dotenv import load_dotenv

# LangChain core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LLM Providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# LangGraph
from langgraph.graph import StateGraph, START, END

load_dotenv()

# -------------------------------------------------------------------
# Shared globals (will be filled by your main_app sync after sandbox)
# -------------------------------------------------------------------
# Expectation: another module (create_ai_sandbox) holds a dict-shaped
# sandbox_state like: {"fileCache": {"files": {...}, "manifest": {...}, ...}, ...}
sandbox_state: Optional[Dict[str, Any]] = None
conversation_state: Optional["ConversationState"] = None


# ---- Helpers to read the shared sandbox_state dict safely ----
def _file_cache() -> Dict[str, Any]:
    """Return the dict like {'files': {...}, 'manifest': ..., 'lastSync': ...}"""
    if isinstance(sandbox_state, dict):
        fc = sandbox_state.get("fileCache") or {}
        return fc if isinstance(fc, dict) else {}
    return {}


def _files_map() -> Dict[str, Any]:
    fc = _file_cache()
    files = fc.get("files")
    return files if isinstance(files, dict) else {}


def _manifest() -> Optional[Dict[str, Any]]:
    fc = _file_cache()
    m = fc.get("manifest")
    return m if isinstance(m, dict) else None


# -------------------------------------------------------------------
# Conversation model (kept minimal)
# -------------------------------------------------------------------
class ConversationState:
    def __init__(self):
        self.conversation_id = f"conv-{int(time.time())}"
        self.started_at = int(time.time())
        self.last_updated = int(time.time())
        self.context = {
            "messages": [],
            "edits": [],
            "project_evolution": {"major_changes": []},
            "user_preferences": {},
        }


# -------------------------------------------------------------------
# Type defs for the agent
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# LLM getters (unchanged)
# -------------------------------------------------------------------
def get_openai():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="gpt-4.1",
    )


def get_anthropic():
    return ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="claude-3-opus-20240229",
    )


def get_google():
    return ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="gemini-1.5-pro",
    )


def get_groq():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=20000,
        streaming=True,
        model="llama3-8b-8192",
    )


# -------------------------------------------------------------------
# Helper: analyze user preferences (unchanged)
# -------------------------------------------------------------------
def analyze_user_preferences(messages: List[Dict[str, Any]]) -> Dict:
    user_messages = [m for m in messages if m.get("role") == "user"]
    patterns = []
    targeted_edit_count = 0
    comprehensive_edit_count = 0

    for msg in user_messages:
        content = (msg.get("content") or "").lower()

        if re.search(r"\b(update|change|fix|modify|edit|remove|delete)\s+(\w+\s+)?(\w+)\b", content):
            targeted_edit_count += 1

        if re.search(r"\b(rebuild|recreate|redesign|overhaul|refactor)\b", content):
            comprehensive_edit_count += 1

        if "hero" in content:
            patterns.append("hero section edits")
        if "header" in content:
            patterns.append("header modifications")
        if "color" in content or "style" in content:
            patterns.append("styling changes")
        if "button" in content:
            patterns.append("button updates")
        if "animation" in content:
            patterns.append("animation requests")

    return {
        "common_patterns": list(set(patterns))[:3],
        "preferred_edit_style": "targeted" if targeted_edit_count > comprehensive_edit_count else "comprehensive",
    }


# -------------------------------------------------------------------
# Progress fanout
# -------------------------------------------------------------------
def send_progress(callbacks, data):
    for cb in callbacks:
        try:
            if asyncio.iscoroutinefunction(cb):
                asyncio.create_task(cb(data))
            else:
                cb(data)
        except Exception:
            pass


# -------------------------------------------------------------------
# Node: analyze intent (uses manifest from shared sandbox_state)
# -------------------------------------------------------------------
def analyze_intent_node(state: AgentState) -> AgentState:
    prompt = state["prompt"]
    is_edit = state["is_edit"]

    send_progress(state["progress_callbacks"], {"type": "status", "message": "🔍 Creating search plan..."})

    edit_context = None
    enhanced_system_prompt = ""

    manifest = _manifest()
    files_map = _files_map()

    if is_edit and manifest:
        try:
            # Use OpenAI for this planning step (as in your original)
            llm = get_openai()
            intent_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                You are an expert code analyst. Your job is to understand user requests for 
                code edits and determine which files need to be modified. Analyze the request carefully.
                
                Output a JSON object with the following structure:
                {
                    "searchTerms": ["term1", "term2"],  // Key terms to search for in the codebase
                    "editType": "UPDATE_COMPONENT",     // One of: UPDATE_COMPONENT, ADD_FEATURE, FIX_BUG, REFACTOR, DELETE_COMPONENT
                    "reasoning": "Explanation of why these files need to be modified"
                }
                """,
                    ),
                    (
                        "human",
                        f"""
                User request: "{prompt}"
                
                File manifest:
                {json.dumps(manifest, indent=2)}
                """,
                    ),
                ]
            )
            plan_text = (intent_prompt | llm | StrOutputParser()).invoke({})
            search_plan = json.loads(plan_text)

            # naive file selection by keywords
            terms = set([t.lower() for t in (search_plan.get("searchTerms") or [])])
            hits = []
            for full_path, data in files_map.items():
                rel = data.get("relativePath") or full_path
                content = data.get("content") or ""
                score = sum(1 for t in terms if (t in rel.lower() or t in content.lower()))
                if score > 0:
                    hits.append(
                        {"filePath": full_path, "lineNumber": 1, "reason": "keyword match", "score": score}
                    )

            if hits:
                target = sorted(hits, key=lambda x: -x["score"])[0]
                send_progress(
                    state["progress_callbacks"],
                    {
                        "type": "status",
                        "message": f"✅ Found code in {target['filePath'].split('/')[-1]} at line {target['lineNumber']}",
                    },
                )
                enhanced_system_prompt = f"""
{ " ".join([f"{r.get('filePath','')}" for r in hits[:5]]) }

SURGICAL EDIT INSTRUCTIONS:
You have been given the EXACT location of the code to edit.
- File: {target["filePath"]}
- Line: {target["lineNumber"]}
- Reason: {target["reason"]}

Make ONLY the change requested by the user. Do not modify any other code.
User request: "{prompt}"
"""
                edit_context = {
                    "primaryFiles": [target["filePath"]],
                    "contextFiles": [],
                    "systemPrompt": enhanced_system_prompt,
                    "editIntent": {
                        "type": search_plan.get("editType", "UPDATE_COMPONENT"),
                        "description": search_plan.get("reasoning", ""),
                        "targetFiles": [target["filePath"]],
                        "confidence": 0.9,
                        "searchTerms": list(terms),
                    },
                }
            else:
                # fallback: App.jsx if present, else first jsx/tsx
                primary = None
                for p in files_map.keys():
                    if p.endswith("src/App.jsx"):
                        primary = p
                        break
                if not primary:
                    for p in files_map.keys():
                        if re.search(r"\.(jsx?|tsx?)$", p):
                            primary = p
                            break
                if primary:
                    edit_context = {
                        "primaryFiles": [primary],
                        "contextFiles": [],
                        "systemPrompt": "",
                        "editIntent": {
                            "type": "UPDATE_COMPONENT",
                            "description": "fallback",
                            "targetFiles": [primary],
                            "confidence": 0.6,
                        },
                    }

        except Exception as e:
            print(f"[analyze_intent_node] error: {e}")

    state["edit_context"] = edit_context
    if edit_context and edit_context.get("systemPrompt"):
        state["system_prompt"] = edit_context["systemPrompt"]
    return state


# -------------------------------------------------------------------
# Node: build prompts (KEEPS YOUR ORIGINAL PROMPT EXACTLY)
# -------------------------------------------------------------------
def build_prompts_node(state: AgentState) -> AgentState:
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
                target_files = [f.split("/")[-1] for f in edit["target_files"]]
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
                    truncated_content = (
                        msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    )
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

    # ----------------------------- ORIGINAL PROMPT (UNCHANGED) -----------------------------
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
    # --------------------------- END ORIGINAL PROMPT ---------------------------

    # Build full prompt with context
    files_map = _files_map()
    full_prompt = prompt
    if context:
        parts = []

        if context.get("sandboxId"):
            parts.append(f"Current sandbox ID: {context['sandboxId']}")

        if context.get("structure"):
            parts.append(f"Current file structure:\n{context['structure']}")

        if files_map:
            if edit_context and edit_context.get("primaryFiles"):
                parts.append("\nEXISTING APPLICATION - TARGETED EDIT MODE")
                primary_files_content = {}
                context_files_content = {}

                for file_path in edit_context["primaryFiles"]:
                    if file_path in files_map:
                        normalized = file_path.replace("/home/user/app/", "")
                        primary_files_content[normalized] = files_map[file_path].get("content", "")

                for file_path in edit_context.get("contextFiles", []):
                    if file_path in files_map:
                        normalized = file_path.replace("/home/user/app/", "")
                        context_files_content[normalized] = files_map[file_path].get("content", "")

                formatted_files = ""
                if primary_files_content:
                    formatted_files += "\n## FILES TO EDIT:\n"
                    for pth, txt in primary_files_content.items():
                        formatted_files += f'\n<file path="{pth}">\n{txt}\n</file>\n'

                if context_files_content:
                    formatted_files += "\n## CONTEXT FILES (For Reference Only):\n"
                    for pth, txt in context_files_content.items():
                        formatted_files += f'\n<file path="{pth}">\n{txt}\n</file>\n'

                parts.append(formatted_files)
                parts.append('\nIMPORTANT: Only modify the files listed under "Files to Edit". The context files are provided for reference only.')
            else:
                parts.append("\nEXISTING APPLICATION - TARGETED EDIT REQUIRED")
                parts.append("\nYou MUST analyze the user request and determine which specific file(s) to edit.")
                parts.append("\nCurrent project files (DO NOT regenerate all of these):")

                parts.append("\n### File List:")
                for full_path in files_map.keys():
                    parts.append(f"- {full_path}")

                parts.append("\n### File Contents (ALL FILES FOR CONTEXT):")
                for full_path, data in files_map.items():
                    txt = data.get("content", "")
                    parts.append(f'\n<file path="{full_path}">\n{txt}\n</file>')

        if parts:
            full_prompt = f"CONTEXT:\n{chr(10).join(parts)}\n\nUSER REQUEST:\n{prompt}"

    # Update state with system and full prompts
    state["system_prompt"] = system_prompt
    state["full_prompt"] = full_prompt

    send_progress(state["progress_callbacks"], {"type": "status", "message": "Planning application structure..."})
    return state


# -------------------------------------------------------------------
# Node: generate code (unchanged logic)
# -------------------------------------------------------------------
def code_generation_node(state: AgentState) -> AgentState:
    packages_to_install: List[str] = []
    files: List[Dict[str, str]] = []
    component_count = 0
    generated_code = ""

    # Choose provider
    model_name = state["model"]
    if model_name.startswith("anthropic/"):
        llm = get_anthropic()
    elif model_name.startswith("google/"):
        llm = get_google()
    elif model_name.startswith("openai/"):
        llm = get_openai()
    else:
        llm = get_groq()

    # Streaming callbacks
    cb_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm.streaming = True
    llm.callbacks = cb_manager

    generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", state["system_prompt"]),
            ("human", state["full_prompt"]),
        ]
    )
    chain = generation_prompt | llm | StrOutputParser()

    buffer: List[str] = []
    file_buffer: List[str] = []
    in_file = False
    file_regex = r'<file path="([^"]+)">([\s\S]*?)<\/file>'

    for chunk in chain.stream({}):
        buffer.append(chunk)
        generated_code += chunk

        if '<file path="' in chunk and not in_file:
            in_file = True
            file_buffer = [chunk]
        elif in_file:
            file_buffer.append(chunk)
            if "</file>" in chunk:
                in_file = False
                file_content = "".join(file_buffer)
                m = re.search(file_regex, file_content)
                if m:
                    path = m.group(1)
                    content = m.group(2)
                    files.append({"path": path, "content": content})

                    if "components/" in path:
                        component_count += 1

                    # package detection on the fly (only in edit mode)
                    if state["is_edit"]:
                        for im in re.finditer(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content):
                            imp = im.group(1)
                            if (
                                not imp.startswith((".", "/", "@/"))
                                and imp not in ("react", "react-dom")
                            ):
                                pkg = "/".join(imp.split("/")[:2]) if imp.startswith("@") else imp.split("/")[0]
                                if pkg not in packages_to_install:
                                    packages_to_install.append(pkg)
                file_buffer = []

        # live preview stream
        send_progress(state["progress_callbacks"], {"type": "stream", "text": chunk, "raw": True})

    # explanation (optional)
    explanation_match = re.search(r"<explanation>([\s\S]*?)</explanation>", generated_code)
    explanation = explanation_match.group(1).strip() if explanation_match else "Code generated successfully!"

    # parse explicit <package> and <packages> tags (only in edit mode)
    if state["is_edit"]:
        for pk in re.finditer(r"<package>([^<]+)</package>", generated_code):
            name = pk.group(1).strip()
            if name and name not in packages_to_install:
                packages_to_install.append(name)
        multi = re.search(r"<packages>([\s\S]*?)</packages>", generated_code)
        if multi:
            for name in [p.strip() for p in re.split(r"[\n,]+", multi.group(1)) if p.strip()]:
                if name not in packages_to_install:
                    packages_to_install.append(name)

    send_progress(
        state["progress_callbacks"],
        {
            "type": "complete",
            "generated_code": generated_code,
            "explanation": explanation,
            "files": len(files),
            "components": component_count,
            "model": state["model"],
            "packages_to_install": packages_to_install or None,
        },
    )

    # update state
    state["generated_code"] = generated_code
    state["files"] = files
    state["packages_to_install"] = packages_to_install
    state["components_count"] = component_count
    state["explanation"] = explanation
    state["warnings"] = []
    return state


# -------------------------------------------------------------------
# Public API (called by main_app)
# -------------------------------------------------------------------
def generate_code(
    prompt: str, model: str = "openai/gpt-4.1", context: Optional[Dict] = None, is_edit: bool = False
) -> Dict[str, Any]:
    """Main function for generating code with AI (no FastAPI)."""
    global conversation_state

    if context is None:
        context = {}

    if conversation_state is None:
        conversation_state = ConversationState()

    # Add user message to conversation history
    conversation_state.context["messages"].append(
        {
            "id": f"msg-{int(time.time())}",
            "role": "user",
            "content": prompt,
            "timestamp": int(time.time()),
            "metadata": {"sandboxId": context.get("sandboxId")},
        }
    )
    if len(conversation_state.context["messages"]) > 20:
        conversation_state.context["messages"] = conversation_state.context["messages"][-15:]

    # Progress sink
    progress_callbacks = [lambda data: print(f"[gen] {data}")]

    # Initial agent state
    state: AgentState = AgentState(
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

    # Build the graph
    graph = StateGraph(AgentState)

    def _entry(s: AgentState) -> str:
        return "analyze_intent" if (s["is_edit"] and _manifest()) else "build_prompts"

    graph.add_node("analyze_intent", analyze_intent_node)
    graph.add_node("build_prompts", build_prompts_node)
    graph.add_node("generate_code", code_generation_node)

    graph.add_conditional_edges(START, _entry, {"analyze_intent": "analyze_intent", "build_prompts": "build_prompts"})
    graph.add_edge("analyze_intent", "build_prompts")
    graph.add_edge("build_prompts", "generate_code")
    graph.add_edge("generate_code", END)

    agent = graph.compile()
    result = agent.invoke(state)

    return {
        "success": True,
        "generated_code": result["generated_code"],
        "files": result["files"],
        "explanation": result["explanation"],
        "packages_to_install": result["packages_to_install"],
    }
