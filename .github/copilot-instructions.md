# Copilot Instructions for GenAI Demo Application

This repository demonstrates various LLM/RAG/Agentic AI patterns using LangChain, Ollama, and Streamlit. When working with this codebase, follow these guidelines and understand the established patterns.

## Interactive Teaching Modes

When users ask questions or request help, respond according to these specialized modes:

### 🔍 Explain Mode (`explain` or `@workspace explain [file]`)
When users ask to explain a file or concept:
- **For demo files**: Provide detailed explanation of the example type, key features, and what to focus on
- **For simple_chat.py**: Explain basic LLM integration, ChatOllama, memory, Streamlit interface
- **For RAG examples**: Explain vector storage, document retrieval, context integration, chain composition
- **For agentic_chat.py**: Explain ReAct pattern, tool integration, agent executor, reasoning cycles

### 🎯 Guide Mode (`guide` or `@workspace guide [topic]`)
When users ask for learning guidance:
- **Basics**: Guide through simple_chat.py → city.py → country.py → state.py → mtg.py
- **RAG**: Guide through wikipedia.py → arxiv.py → web.py → chroma.py → pgvector.py  
- **Agents**: Focus on agentic_chat.py, prompts.py, and tool integration patterns

### 🛤️ Learning Path Mode (`path` or `@workspace learning path [type]`)
Provide structured learning progressions:
- **Beginner Path**: Start with basic LLM concepts, move to prompt engineering
- **Intermediate Path**: RAG implementation, vector databases, document processing
- **Advanced Path**: Agent development, tool integration, ReAct patterns

### ❓ Question Mode (general questions about concepts)
For concept questions, provide context-aware answers:
- **Ollama questions**: Explain local hosting, model configuration, integration patterns
- **LangChain questions**: Explain chains, retrievers, agents, memory, prompt templates
- **RAG questions**: Explain retrieval, embedding, vector storage, context integration
- **Agent questions**: Explain ReAct pattern, tool use, reasoning, decision making

### 🔧 Implementation Mode (`how to` or `implement`)
When users want to implement features:
- Provide code examples following established patterns
- Reference existing examples as templates
- Include proper error handling and testing approaches
- Follow the separation of concerns pattern (create_chain, process_query, handle_ui)

## Repository Structure & Architecture

### Core Application
- **`src/app.py`**: Main Streamlit application with example type selection
- **`src/example/`**: Contains all demo implementations organized by complexity
- **`src/internal/`**: Shared utilities and prompt templates
- **`tests/`**: Comprehensive test suite with unit and integration tests

### Example Categories

#### 🌟 Basic Examples (Beginner)
- **`simple_chat.py`**: Basic LLM integration with memory using ChatOllama
- **`city.py`, `country.py`, `state.py`**: Prompt templating and chain composition
- **`mtg.py`**: External API integration with structured output

#### 🔍 RAG Examples (Intermediate)
- **`wikipedia.py`**: Dynamic document retrieval using WikipediaRetriever
- **`arxiv.py`**: Academic paper search and analysis with ArxivRetriever
- **`web.py`**: Real-time web search using BraveSearchLoader
- **`chroma.py`**: Vector storage with ChromaDB for document embeddings
- **`pgvector.py`**: Production PostgreSQL vector storage with PGVector

#### 🤖 Agent Examples (Advanced)
- **`agentic_chat.py`**: Multi-tool ReAct agent with Wikipedia and Arxiv tools

## Key Technical Patterns

### LLM Integration
- **Primary LLM Provider**: Ollama (local hosting)
- **Model Configuration**: Set via `OLLAMA_HOST` environment variable
- **LLM Creation**: Use `internal.util.create_llm(model_name)` for consistency
- **Streaming**: Enabled for real-time responses in Streamlit interface

### RAG Chain Architecture
```python
# Standard RAG pattern used throughout examples
rag_chain_from_docs = (
    RunnablePassthrough.assign(
        context=(lambda x: format_docs(x["context"])))
    | summarize_prompt
    | llm
    | StrOutputParser()
)

# Retrieval chain
retrieve_docs = (
    lambda x: x["question"]
) | retriever

# Complete chain
chain = RunnablePassthrough.assign(
    context=retrieve_docs
).assign(answer=rag_chain_from_docs)
```

### Prompt Templates
- **Location**: `src/internal/prompts.py`
- **Basic Prompts**: `create_question_type_prompt()`, `create_mtg_prompt()`
- **RAG Prompts**: `create_summarize_prompt()`, `create_summarize_prompt_v2()`
- **Agent Prompts**: `create_agentic_react_prompt()` (ReAct pattern implementation)

### Vector Storage Patterns
- **Chroma**: Local development, collection-based storage
- **PGVector**: Production PostgreSQL with vector extensions
- **Embeddings**: OllamaEmbeddings for consistent model usage
- **Document Processing**: PDF loading, chunking, and vectorization utilities

### Agent Architecture (ReAct Pattern)
- **Tools**: Wikipedia search, Arxiv research tools
- **Prompt**: Custom ReAct implementation in `create_agentic_react_prompt()`
- **Execution**: AgentExecutor with error handling and iteration limits
- **Memory**: Session-based conversation state

## Development Guidelines

### Code Organization
- **Separation of Concerns**: Each example has `create_chain()`, `process_query()`, and `handle_ui()` functions
- **Testability**: Business logic separated from Streamlit UI for easier testing
- **Error Handling**: Comprehensive try-catch blocks with user-friendly messages
- **Logging**: Structured logging throughout application

### Environment Variables
```bash
OLLAMA_HOST=http://localhost:11434        # Required
DB_URL=postgresql://...                   # For PGVector examples
CHROMA_HOST=localhost                     # For Chroma examples
CHROMA_PORT=8000                          # For Chroma examples
BRAVE_SEARCH_API_KEY=...                  # For web search example
```

### Testing Patterns
- **Unit Tests**: Mock external dependencies (Ollama, retrievers, tools)
- **Integration Tests**: Test complete workflows end-to-end
- **Coverage**: Maintain high test coverage for reliability
- **Mocking Strategy**: Use `unittest.mock.patch` for LangChain components

## Common Implementation Patterns

### Streamlit UI Structure
```python
def handle_example(st, model_name):
    # Input components
    input_data = st.text_input("Question", placeholder="...")
    
    if st.button("Process"):
        with st.spinner('Please wait...'):
            try:
                # Create chain
                chain = create_example_chain(model_name)
                
                # Process query
                result = process_example_query(chain, input_data)
                
                # Display results
                if result and result['answer']:
                    st.success(result['answer'])
                else:
                    st.warning("No answer found.")
                    
            except Exception as e:
                st.exception(f"An error occurred: {e}")
```

### Chain Creation Pattern
```python
def create_example_chain(model_name):
    """Create and return the example chain for easier testing."""
    llm = create_llm(model_name)
    prompt = create_appropriate_prompt()
    
    # Build chain components
    chain = (
        RunnablePassthrough.assign(context=retriever)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
```

### Error Handling
- Always wrap LLM calls in try-catch blocks
- Provide meaningful error messages to users
- Log errors for debugging while showing user-friendly messages
- Handle API key missing scenarios gracefully

## Dependencies & Integrations

### Core Framework Stack
- **LangChain**: Primary framework for LLM applications
- **Streamlit**: Web interface and user interaction
- **Ollama**: Local LLM hosting and inference
- **Python-dotenv**: Environment variable management

### Vector Databases
- **ChromaDB**: Development and local storage
- **PGVector**: Production PostgreSQL integration
- **OllamaEmbeddings**: Consistent embedding generation

### External Integrations
- **Wikipedia**: Dynamic content retrieval
- **Arxiv**: Academic research paper access
- **Brave Search**: Real-time web content
- **MTG SDK**: Magic the Gathering card data

## When Adding New Examples

1. **Follow the established pattern**: `create_chain()`, `process_query()`, `handle_ui()`
2. **Add comprehensive tests**: Unit tests with mocking, integration tests
3. **Update main app**: Add to the selectbox options in `src/app.py`
4. **Document environment variables**: If new APIs are required
5. **Error handling**: Implement graceful failure modes
6. **Logging**: Add appropriate logging statements

## Common Gotchas

- **Environment Variables**: Always assert required variables are set
- **Ollama Model Names**: Different examples may use different models
- **Vector Store Persistence**: Chroma collections persist; PGVector requires database setup
- **API Keys**: Web and some retrievers require external API keys
- **Session State**: Agentic chat maintains conversation state in Streamlit

## Testing Strategy

- **Mock External Services**: Ollama, vector databases, retrievers
- **Test Business Logic**: Separate from Streamlit UI components
- **Integration Tests**: End-to-end workflow validation
- **Error Scenarios**: Test failure modes and error handling

## Coding Standards

For comprehensive Python coding standards, style guidelines, Pydantic v2 patterns, exception handling, and pre-submission checklists, see **[.github/CODING_STANDARDS.md](.github/CODING_STANDARDS.md)**.

### Quick Reference
- **Linting**: `flake8 src tests --max-line-length=120` (must pass before PR)
- **Tests**: `pytest tests/ -q` (110 tests must pass)
- **Pydantic v2**: All custom fields require `Field()` declaration; no extra params on `OllamaEmbeddings`
- **Exceptions**: Remove unused `as e` variable; catch specific exceptions before generic
- **Whitespace**: No trailing spaces on blank lines (W293)
- **Line breaks**: Break at logical points for 120-char limit (after operators, in parameter lists)
- **Imports**: Remove transitive dependencies; verify usage with grep before removing

## Response Templates for AI Assistants

### When Explaining Demo Files
```
# 📚 [Example Name]

**Type:** [BASIC/RAG/AGENT] | **Difficulty:** [beginner/intermediate/advanced]

## Description
[Brief description of what this example demonstrates]

## Key Features
- [Feature 1 with explanation]
- [Feature 2 with explanation]
- [Feature 3 with explanation]

## 🔍 What to Look For
[Specific guidance based on example type - LLM integration, RAG patterns, or agent behavior]

## 🔗 Related Files
[List related files and their purposes]
```

### When Providing Learning Guidance
```
# 🎓 [Topic] Learning Path

## Recommended Progression
1. **[File 1]** - [Learning objective]
2. **[File 2]** - [Learning objective] 
3. **[File 3]** - [Learning objective]

## Key Concepts to Master
- [Concept 1]: [Explanation]
- [Concept 2]: [Explanation]
- [Concept 3]: [Explanation]

## 💡 Next Steps
[Suggestions for hands-on learning and experimentation]
```

### When Answering Concept Questions
```
## [Technology/Pattern] in This Codebase

**Overview:** [Brief explanation]

**Key Implementation Details:**
- [Detail 1 with file references]
- [Detail 2 with file references]
- [Detail 3 with file references]

**Examples in Action:**
- See `[file1]` for [specific use case]
- See `[file2]` for [specific use case]

**Best Practices:**
[List best practices from the codebase]
```

### Example Interactions to Support

**User:** "Explain simple_chat.py"
**Response:** Use Explain Mode template with specific details about ChatOllama integration, memory, and Streamlit interface

**User:** "Guide me through RAG"  
**Response:** Use Guide Mode with step-by-step progression through RAG examples

**User:** "How does ollama work here?"
**Response:** Use Question Mode to explain Ollama integration patterns with file references

**User:** "Create a new basic example"
**Response:** Use Implementation Mode with code template following established patterns

This codebase demonstrates production-ready patterns for LLM application development with proper separation of concerns, comprehensive testing, and scalable architecture.