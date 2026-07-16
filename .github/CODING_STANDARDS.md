# Python Coding Standards for GenAI Starter

This document outlines the coding standards and best practices for this project, maintained for consistency across all development work.

## Code Style & Linting

### Flake8 Compliance
- **Max line length**: 120 characters
- **Command**: `flake8 src tests --max-line-length=120`
- All code must pass flake8 before submission
- Address violations in this order: errors → style issues → warnings

### Common Flake8 Rules
- **E501**: Line too long - break lines at logical points (after operators, in parameter lists)
- **W293**: Blank line contains whitespace - remove trailing spaces from empty lines
- **W291**: Trailing whitespace - strip all trailing spaces from code lines
- **F401**: Unused import - remove imports that aren't referenced in code
- **F841**: Unused variable - prefix with `_` if intentionally unused, or remove
- **E306**: Expected blank line before nested definition - add blank lines between methods
- **E303**: Too many blank lines - limit to max 2 consecutive blank lines

### Line Breaking Strategy
For long lines over 120 characters:
```python
# Bad - single long line
result = response.get('output') or (response.get('messages', [])[-1].content if response.get('messages') else "No response")

# Good - logical breaks with continuation
result = (response.get('output') or
          (response.get('messages', [])[-1].content
           if response.get('messages') else "No response"))
```

## Dependency Management

### Import Organization
1. Standard library imports first
2. Third-party imports second (alphabetical)
3. Local imports third (relative imports with `from .module import`)

```python
import os
import json
from typing import List

import httpx
from langchain_core.tools import tool

from .logger import logger
```

### Removing Transitive Dependencies
- Only include direct dependencies in `requirements.txt`
- Verify no imports in codebase use transitive dependencies before removing
- Use grep to check: `grep -r "from transitive_module" src tests --include="*.py"`

## Pydantic v2 Compatibility

### Custom BaseRetriever Subclasses
All fields must be explicitly declared with `Field()`:

```python
from pydantic import Field
from langchain_core.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    """Custom retriever implementation."""
    
    # Proper v2 style - declare all fields
    top_k: int = Field(default=4, description="Number of top results")
    custom_param: str = Field(description="Custom parameter description")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents."""
        # Implementation
```

### OllamaEmbeddings Usage
- Never pass `show_progress=True` parameter (causes "Extra inputs not permitted" error)
- The defaults work fine without it:

```python
# Bad
embeddings = OllamaEmbeddings(model="model-name", show_progress=True)

# Good
embeddings = OllamaEmbeddings(model="model-name")
```

## Exception Handling

### Unused Exception Variables
Remove `as e` if the exception variable isn't used:

```python
# Bad
except Exception as e:
    return None

# Good
except Exception:
    return None

# Good if you use it
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    return None
```

### Specific Exception Ordering
Catch specific exceptions before generic ones:

```python
# Good - specific first, generic last
try:
    result = wikipedia.search(query)
except wikipedia.exceptions.PageError:
    return None
except wikipedia.exceptions.DisambiguationError:
    return None
except Exception:
    return None
```

## Documentation Standards

### Docstring Format
Use Google-style docstrings with Args, Returns sections:

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief one-line description.
    
    Longer description if needed, explaining the purpose
    and key behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    pass
```

### Type Hints
- Always include type hints for function parameters and return values
- Use `Optional[Type]` for nullable returns
- Use `List[Type]` from typing module for collections

```python
from typing import List, Optional

def search_documents(query: str, limit: int = 10) -> List[Document]:
    """Search and return documents."""
    pass

def get_config(key: str) -> Optional[str]:
    """Get configuration value or None if not found."""
    pass
```

## Testing Standards

### Unit Test Mocking Patterns
Mock external dependencies to isolate business logic:

```python
from unittest.mock import Mock, patch

def test_my_function():
    with patch('module.external_dependency') as mock_dep:
        mock_dep.return_value = {"expected": "result"}
        result = my_function()
        assert result == expected_value

# Or use Mock() for simple object stubbing
mock_streamlit = Mock()
mock_streamlit.text_input.return_value = "user input"
```

### Mock Class Pattern (for complex state)
For Streamlit-like session objects:

```python
class MockSessionState:
    def __init__(self):
        self._data = {}
    
    def __contains__(self, key):
        return key in self._data
    
    def __getattr__(self, key):
        return self._data.get(key)
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._data[key] = value

session_state = MockSessionState()
```

### Test Organization
- Keep business logic separate from UI logic for easier testing
- Use fixtures in `conftest.py` for shared test setup
- Test both success and error paths

## Code Organization

### Function Separation Pattern
Follow this pattern for handler functions:

```python
def create_chain(model_name: str) -> Chain:
    """Create and return the processing chain."""
    # Setup and chain creation
    return chain

def process_query(chain: Chain, query: str) -> dict:
    """Process a query through the chain."""
    # Business logic
    return result

def handle_ui(st, model_name: str):
    """Streamlit UI handler."""
    # UI setup
    chain = create_chain(model_name)
    result = process_query(chain, input_data)
    # Display results
```

## Common Patterns

### Custom HTTP Client Pattern
For MCP or external service integration:

```python
class MyClient:
    def __init__(self, host: str):
        self.host = host
        self.client = httpx.Client(timeout=30.0)
    
    def call_service(self, method: str, params: dict) -> Optional[dict]:
        """Call external service with error handling."""
        try:
            response = self.client.post(self.host, json={"method": method, "params": params})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Service call failed: {str(e)}", exc_info=True)
            return None
```

### Tool Definition Pattern (LangChain)
For creating tools for agents:

```python
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Tool description for agent documentation.
    
    Args:
        query: What the user is searching for
        
    Returns:
        Formatted result string
    """
    result = perform_search(query)
    return format_result(result)
```

## Pre-Submission Checklist

Before submitting code:

- [ ] Run `pytest tests/ -q` - all tests pass
- [ ] Run `flake8 src tests --max-line-length=120` - zero violations
- [ ] Check for unused imports - `grep -r "^import\|^from" src tests`
- [ ] Verify no trailing whitespace - `grep -n "[[:space:]]$" src/**/*.py`
- [ ] Run type checking if available
- [ ] All docstrings present and formatted correctly
- [ ] Exception handling is specific and meaningful
- [ ] No unused variables or commented-out code

## References

- **Flake8 Documentation**: https://flake8.pycqa.org/
- **PEP 8**: https://www.python.org/dev/peps/pep-0008/
- **Google Python Style Guide**: https://google.github.io/styleguide/pyguide.html
- **Pydantic v2 Migration**: https://docs.pydantic.dev/latest/concepts/models/
