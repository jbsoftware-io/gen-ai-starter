# Generative AI Starter

[![Tests](https://github.com/jbsoftware-io/gen-ai-demo-app/actions/workflows/pr-build.yml/badge.svg?branch=main)](https://github.com/jbsoftware-io/gen-ai-demo-app/actions/workflows/pr-build.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://jbsoftware-io.github.io/gen-ai-demo-app/coverage-badge.json)](https://github.com/jbsoftware-io/gen-ai-demo-app/actions/workflows/pr-build.yml)

## Purpose
This repository can be used as a starting point for building custom LLM applications using Open Source tooling and models.  It incorporates Ollama, Open WebUI, Langchain, Streamlit, Chroma, and PGVector using docker to containerize the application and docker compose to run the various service dependencies.

## Pre-requisites

* Docker Engine installed
  * Run Option 1
      * Docker Engine configured with >= 12 GB memory
  * Run Option 2
      * Docker Engine configured with >= 8 GB memory
      * Ollama Executable Installed
```
brew install ollama
```

## Optional Pre-requisites (required for "Web" example)

* Register to get a free Brave Search API key [here](https://api-dashboard.search.brave.com/register).
  * The free key gives you 2000 calls per month, and if you need to scale they are affordable.
  * To learn more about Brave Search API click [here](https://brave.com/search/api/)
* Create a `.env` file in the root of the project and add your key as follows:
```
 BRAVE_SEARCH_API_KEY={yourKeyHere}
 ```

## Running the Backing Services and LLM

#### Option 1 (Easiest but Slower, only CPU)

```
docker compose --profile=cpu up -d
```

#### Option 2 (Fastest, uses GPU)
```
docker compose up -d
./etc/ollama_entrypoint.sh
```

## Access the Demo App

```
http://localhost:8501/
```

### Open WebUI and Ollama Links

To check out the Open Web UI interface (for manual chats and more) go here and sign up for an admin account.

Open WebUI:

```
http://localhost:3000/
```

List OLama Models:

```
curl http://localhost:11434/api/tags
```

Ollama API Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#api

## Testing

This application includes a comprehensive test suite that runs in Docker for consistency.

### Quick Start
```bash
# Run all tests
docker compose run --rm app pytest tests/ -v

# Run just unit tests
docker compose run --rm app pytest tests/unit/ -v

# Run with coverage
docker compose run --rm app pytest tests/ --cov=src --cov-report=term
```

### Specific Tests
```bash
# Test a specific file
docker compose run --rm app pytest tests/unit/test_app.py -v

# Test a specific class
docker compose run --rm app pytest tests/unit/example/test_simple_chat.py::TestSimpleChat -v

# Test a specific method
docker compose run --rm app pytest tests/unit/test_app.py::TestApp::test_model_selection_logic -v
```

### Before Updating Dependencies
1. **Run the full test suite**: `docker compose run --rm app pytest tests/ -v`
2. **Ensure all tests pass** before making any changes
3. **Update packages incrementally** and test after each change

## 🤖 GenAI Teacher Chat Mode

This repository includes an integrated AI teaching assistant accessible through GitHub Copilot Chat. The GenAI Teacher provides personalized guidance for learning LLM, RAG, and Agentic AI patterns.

### Activation

In any GitHub interface (VS Code, GitHub.com, or GitHub Mobile), use the `#genai-teacher` chat mode:

```
#genai-teacher [your question or request]
```

### Available Teaching Modes

#### 🔍 Explain Mode
Get detailed explanations of demo files and concepts:
```
#genai-teacher explain simple_chat.py
#genai-teacher explain RAG patterns
#genai-teacher explain the ReAct agent pattern
```

#### 🎯 Guide Mode  
Receive structured learning guidance through topics:
```
#genai-teacher guide me through RAG
#genai-teacher guide me through the basics
#genai-teacher guide me through agent development
```

#### 🛤️ Learning Path Mode
Get personalized learning progressions:
```
#genai-teacher learning path for beginners
#genai-teacher learning path for RAG
#genai-teacher learning path for agents
```

#### 💡 Implementation Mode
Get help implementing new features:
```
#genai-teacher how to create a new RAG example
#genai-teacher implement a custom retriever
#genai-teacher add error handling to my chain
```

### Example Interactions

**Beginner Starting Point:**
```
#genai-teacher guide me through the basics
```

**Understanding a Specific File:**
```
#genai-teacher explain src/example/agentic_chat.py
```

**Getting Implementation Help:**
```
#genai-teacher how to add a new vector database example
```

The GenAI Teacher understands the repository structure, coding patterns, and can provide context-aware guidance tailored to your current learning needs.

## Further Reading

- Ollama - Open source app allowing interactions with various LLM models, prompts, tools, and functions
  - Website: https://ollama.com/
  - Github: https://github.com/ollama/ollama
- Open WebUI - Open source Web UI wrapper to interact with local or remote Ollama instances
  - Website: https://openwebui.com/
  - Github: https://github.com/open-webui/open-webui
- RAG
  - Retrieval Augmented Generation Article: https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/
- LangChain - Tool to help build custom prompts and embed using vectorDB
  - Website: https://python.langchain.com/docs/get_started/introduction
- Hugging Face - Open Source ML/AI Community Hub with tons of models for various use cases
  - Website: https://huggingface.co/
  - LangChain Hugging Face Support: https://python.langchain.com/v0.1/docs/integrations/platforms/huggingface/
- Chroma - Vector database
  - Website: https://www.trychroma.com/
  - LangChain Chroma Support: https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/
- PGVector - Tool allowing storage of vectors in postgresdb
  - Website: https://github.com/pgvector/pgvector
  - LangChain PGVector Support: https://python.langchain.com/docs/integrations/vectorstores/pgvector
- Wikipedia - Retriever allowing document retrieval and usage in LLM
  - Website: https://wikipedia.org
  - LangChain Wikipedia Support: https://python.langchain.com/docs/integrations/retrievers/wikipedia/
- Web - Brave API Search Loader - Document loader supporting Brave API Website lookups
  - Website: https://brave.com/search/api/
  - LangChain BraveSearchLoader Docs: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.brave_search.BraveSearchLoader.html
- Arxiv - open-access archive for nearly 2.4 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics
  - Website: https://arxiv.org/
  - LangChain ArxivRetriever Support: https://python.langchain.com/docs/integrations/retrievers/arxiv/
