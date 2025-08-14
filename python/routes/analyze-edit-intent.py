import os
import json
import re
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

# Prefer modern provider packages; they’re what you’re already importing.
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------
# Config & helpers
# -------------------------

# Env-driven defaults (override via ENV if you like)
ANTHROPIC_MODEL_DEFAULT = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL_DEFAULT = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
GOOGLE_MODEL_DEFAULT = os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")

def _clean_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    # Avoid double /v1 in some client versions
    return url[:-3] if url.endswith("/v1") else url

# Pseudo/alias models → real provider models
OSS_ALIASES = {
    # your default incoming
    "openai/gpt-oss-20b": ("groq", GROQ_MODEL_DEFAULT),
    "gpt-oss-20b": ("groq", GROQ_MODEL_DEFAULT),
}

def _build_anthropic(model_name: Optional[str] = None) -> ChatAnthropic:
    base_url = _clean_base_url(os.environ.get("ANTHROPIC_BASE_URL"))
    kwargs: Dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model=model_name or ANTHROPIC_MODEL_DEFAULT,
        **kwargs,
    )

def _build_openai(model_name: Optional[str] = None) -> ChatOpenAI:
    base_url = os.environ.get("OPENAI_BASE_URL")  # OpenAI-style clients expect raw base
    kwargs: Dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=model_name or OPENAI_MODEL_DEFAULT,
        **kwargs,
    )

def _build_groq(model_name: Optional[str] = None) -> ChatGroq:
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=model_name or GROQ_MODEL_DEFAULT,  # NOTE: ChatGroq expects `model=`
    )

def _build_google(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model=model_name or GOOGLE_MODEL_DEFAULT,
    )

def _select_model(model_str: str):
    """
    Resolve a user-supplied `model` string like:
      - 'anthropic/claude-3-5-sonnet-20240620'
      - 'openai/gpt-4o-mini'
      - 'google/gemini-1.5-pro'
      - 'groq/llama-3.1-70b-versatile'
      - 'openai/gpt-oss-20b' (alias -> groq)
    """
    # Handle aliases first
    if model_str in OSS_ALIASES:
        provider, real_model = OSS_ALIASES[model_str]
        if provider == "groq":
            return _build_groq(real_model)

    # Provider-prefixed
    if "/" in model_str:
        provider, name = model_str.split("/", 1)
        provider = provider.lower()
        if provider == "anthropic":
            return _build_anthropic(name)
        if provider == "openai":
            return _build_openai(name)
        if provider == "google":
            return _build_google(name)
        if provider == "groq":
            return _build_groq(name)

    # Fallback heuristics
    # Try to infer by recognizable names
    ms = model_str.lower()
    if ms.startswith("claude"):
        return _build_anthropic(model_str)
    if ms.startswith("gpt"):
        return _build_openai(model_str)
    if "gemini" in ms:
        return _build_google(model_str)
    if "llama" in ms or "mixtral" in ms:
        return _build_groq(model_str)

    # Final fallback → Groq
    return _build_groq(model_str)

# -------------------------
# Domain models
# -------------------------

class EditType(str, Enum):
    UPDATE_COMPONENT = 'UPDATE_COMPONENT'
    ADD_FEATURE = 'ADD_FEATURE'
    FIX_ISSUE = 'FIX_ISSUE'
    UPDATE_STYLE = 'UPDATE_STYLE'
    REFACTOR = 'REFACTOR'
    ADD_DEPENDENCY = 'ADD_DEPENDENCY'
    REMOVE_ELEMENT = 'REMOVE_ELEMENT'

class FallbackSearch(BaseModel):
    terms: List[str]
    patterns: Optional[List[str]] = None

class SearchPlanSchema(BaseModel):
    editType: EditType = Field(description='The type of edit being requested')
    reasoning: str = Field(description='Explanation of the search strategy')
    searchTerms: List[str] = Field(description='Specific text to search for (case-insensitive). Be VERY specific - exact button text, class names, etc.')
    regexPatterns: Optional[List[str]] = Field(default=None, description='Regex patterns for finding code structures (e.g., "className=[\\"\\\'].*header.*[\\"\\\']")')
    fileTypesToSearch: List[str] = Field(default=['.jsx', '.tsx', '.js', '.ts'], description='File extensions to search')
    expectedMatches: int = Field(default=1, ge=1, le=10, description='Expected number of matches (helps validate search worked)')
    fallbackSearch: Optional[FallbackSearch] = Field(default=None, description='Backup search if primary fails')

# -------------------------
# Main function
# -------------------------

def analyze_edit_intent(prompt: str, manifest: Dict[str, Any], model: str = 'openai/gpt-oss-20b') -> Dict[str, Any]:
    try:
        print('[analyze-edit-intent] Request received')
        print('[analyze-edit-intent] Prompt:', prompt)
        print('[analyze-edit-intent] Model:', model)
        print('[analyze-edit-intent] Manifest files count:', len(manifest.get('files', {})) if manifest and manifest.get('files') else 0)

        if not prompt or not manifest:
            return {'error': 'prompt and manifest are required'}

        # Collect valid files from manifest
        valid_files = []
        for path, info in (manifest.get('files') or {}).items():
            # Filter out invalid paths (mirrors your TS logic)
            if '.' in path and not re.search(r'/\d+$', path):
                valid_files.append((path, info))

        file_summary_lines = []
        for path, info in valid_files:
            component_name = (info.get('componentInfo') or {}).get('name') or path.split('/')[-1]
            child_components = ', '.join((info.get('componentInfo') or {}).get('childComponents') or []) or 'none'
            file_summary_lines.append(f"- {path} ({component_name}, renders: {child_components})")

        file_summary = '\n'.join(file_summary_lines)

        print('[analyze-edit-intent] Valid files found:', len(valid_files))
        if len(valid_files) == 0:
            print('[analyze-edit-intent] No valid files found in manifest')
            return {'success': False, 'error': 'No valid files found in manifest'}

        print('[analyze-edit-intent] Analyzing prompt:', prompt)
        print('[analyze-edit-intent] File summary preview:', '\n'.join(file_summary.split('\n')[:5]))

        # Select model robustly
        ai_model = _select_model(model)
        print('[analyze-edit-intent] Using AI model object:', type(ai_model).__name__)

        # Structured output parser
        parser = PydanticOutputParser(pydantic_object=SearchPlanSchema)

        system_message = SystemMessage(content=f"""You are an expert at planning code searches. Your job is to create a search strategy to find the exact code that needs to be edited.

DO NOT GUESS which files to edit. Instead, provide specific search terms that will locate the code.

SEARCH STRATEGY RULES:
1. For text changes (e.g., "change 'Start Deploying' to 'Go Now'"):
   - Search for the EXACT text: "Start Deploying"
   
2. For style changes (e.g., "make header black"):
   - Search for component names: "Header", "<header"
   - Search for class names: "header", "navbar"
   - Search for className attributes containing relevant words
   
3. For removing elements (e.g., "remove the deploy button"):
   - Search for the button text or aria-label
   - Search for relevant IDs or data-testids
   
4. For navigation/header issues:
   - Search for: "navigation", "nav", "Header", "navbar"
   - Look for Link components or href attributes
   
5. Be SPECIFIC:
   - Use exact capitalization for user-visible text
   - Include multiple search terms for redundancy
   - Add regex patterns for structural searches

Current project structure for context:
{file_summary}""")

        user_message = HumanMessage(content=f"""User request: "{prompt}"

Create a search plan to find the exact code that needs to be modified. Include specific search terms and patterns.

{parser.get_format_instructions()}""")

        messages = [system_message, user_message]

        result_response = ai_model.invoke(messages)
        result_object = parser.parse(result_response.content)

        print('[analyze-edit-intent] Search plan created:', {
            'editType': result_object.editType,
            'searchTerms': result_object.searchTerms,
            'patterns': len(result_object.regexPatterns) if result_object.regexPatterns else 0,
            'reasoning': result_object.reasoning
        })

        return {'success': True, 'searchPlan': result_object.dict()}

    except Exception as error:
        print('[analyze-edit-intent] Error:', error)
        return {'success': False, 'error': str(error)}
