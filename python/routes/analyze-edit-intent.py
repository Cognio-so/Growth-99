import os
import json
import re
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

# Exact enum match with TypeScript
class EditType(str, Enum):
    UPDATE_COMPONENT = 'UPDATE_COMPONENT'
    ADD_FEATURE = 'ADD_FEATURE'
    FIX_ISSUE = 'FIX_ISSUE'
    UPDATE_STYLE = 'UPDATE_STYLE'
    REFACTOR = 'REFACTOR'
    ADD_DEPENDENCY = 'ADD_DEPENDENCY'
    REMOVE_ELEMENT = 'REMOVE_ELEMENT'

# Create AI model instances exactly like TypeScript
groq = ChatGroq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

anthropic = ChatAnthropic(
    api_key=os.environ.get('ANTHROPIC_API_KEY'),
    base_url=os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com/v1'),
)

openai = ChatOpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
    base_url=os.environ.get('OPENAI_BASE_URL'),
)

def create_google_generative_ai(model_name: str):
    return ChatGoogleGenerativeAI(
        api_key=os.environ.get('GOOGLE_API_KEY'),
        model=model_name
    )

# Schema for the AI's search plan - not file selection!
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

def analyze_edit_intent(prompt: str, manifest: Dict[str, Any], model: str = 'openai/gpt-oss-20b') -> Dict[str, Any]:
    try:
        print('[analyze-edit-intent] Request received')
        print('[analyze-edit-intent] Prompt:', prompt)
        print('[analyze-edit-intent] Model:', model)
        print('[analyze-edit-intent] Manifest files count:', len(manifest.get('files', {})) if manifest and manifest.get('files') else 0)
        
        if not prompt or not manifest:
            return {
                'error': 'prompt and manifest are required'
            }
        
        # Create a summary of available files for the AI
        valid_files = []
        for path, info in (manifest.get('files') or {}).items():
            # Filter out invalid paths
            if '.' in path and not re.search(r'/\d+$', path):
                valid_files.append((path, info))
        
        file_summary_lines = []
        for path, info in valid_files:
            component_name = (info.get('componentInfo') or {}).get('name') or path.split('/')[-1]
            has_imports = len(info.get('imports') or []) > 0
            child_components = ', '.join((info.get('componentInfo') or {}).get('childComponents') or []) or 'none'
            file_summary_lines.append(f"- {path} ({component_name}, renders: {child_components})")
        
        file_summary = '\n'.join(file_summary_lines)
        
        print('[analyze-edit-intent] Valid files found:', len(valid_files))
        
        if len(valid_files) == 0:
            print('[analyze-edit-intent] No valid files found in manifest')
            return {
                'success': False,
                'error': 'No valid files found in manifest'
            }
        
        print('[analyze-edit-intent] Analyzing prompt:', prompt)
        print('[analyze-edit-intent] File summary preview:', '\n'.join(file_summary.split('\n')[:5]))
        
        # Select the appropriate AI model based on the request
        ai_model = None
        if model.startswith('anthropic/'):
            ai_model = anthropic
            # Set model name by creating new instance with specific model
            model_name = model.replace('anthropic/', '')
            ai_model = ChatAnthropic(
                api_key=os.environ.get('ANTHROPIC_API_KEY'),
                base_url=os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com/v1'),
                model=model_name
            )
        elif model.startswith('openai/'):
            if 'gpt-oss' in model:
                ai_model = ChatGroq(
                    api_key=os.environ.get('GROQ_API_KEY'),
                    model_name=model
                )
            else:
                model_name = model.replace('openai/', '')
                ai_model = ChatOpenAI(
                    api_key=os.environ.get('OPENAI_API_KEY'),
                    base_url=os.environ.get('OPENAI_BASE_URL'),
                    model=model_name
                )
        elif model.startswith('google/'):
            model_name = model.replace('google/', '')
            ai_model = create_google_generative_ai(model_name)
        else:
            # Default to groq if model format is unclear
            ai_model = ChatGroq(
                api_key=os.environ.get('GROQ_API_KEY'),
                model_name=model
            )
        
        print('[analyze-edit-intent] Using AI model:', model)
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=SearchPlanSchema)
        
        # Use AI to create a search plan
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
        
        # Return the search plan, not file matches
        return {
            'success': True,
            'searchPlan': result_object.dict()
        }
        
    except Exception as error:
        print('[analyze-edit-intent] Error:', error)
        return {
            'success': False,
            'error': str(error)
        }