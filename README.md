# Generative AI Starter

[![Tests](https://github.com/jbsoftware-io/gen-ai-starter/actions/workflows/pr-build.yml/badge.svg?branch=main)](https://github.com/jbsoftware-io/gen-ai-starter/actions/workflows/pr-build.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://jbsoftware-io.github.io/gen-ai-starter/coverage-badge.json)](https://github.com/jbsoftware-io/gen-ai-starter/actions/workflows/pr-build.yml)

## Table of Contents
1. [Purpose & Features](#purpose--features)
2. [Architecture & Repository Structure](#architecture--repository-structure)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Testing](#testing)
6. [Optional: Langfuse Support](#optional-langfuse-support)
7. [GenAI Teacher Chat Mode](#genai-teacher-chat-mode)
8. [Troubleshooting & Codespaces Notes](#troubleshooting--codespaces-notes)
9. [Further Reading](#further-reading)

---

## Purpose & Features
This repository can be used as a starting point for building custom LLM applications using Open Source tooling and models. It incorporates Ollama, Open WebUI, Langchain, Streamlit, Chroma, and PGVector using Docker to containerize the application and Docker Compose to run the various service dependencies.

**Key Features:**
- LLM integration (Ollama, LangChain)
- RAG (Retrieval Augmented Generation) examples
- Agentic AI patterns (ReAct, multi-tool agents)
- Streamlit web interface
- Vector database support (Chroma, PGVector)
- Integrated teaching assistant (GenAI Teacher)

---

## Architecture & Repository Structure

**Main Components:**
- src - Application code and examples
- etc - Scripts and configuration
- tests - Unit and integration tests
- Dockerfile, compose.yml - Containerization and orchestration

---

## Getting Started

### Prerequisites

- **Docker Engine** installed
  - Option 1: Docker Engine configured with >= 12 GB memory
  - Option 2: Docker Engine configured with >= 8 GB memory and Ollama Executable installed
    ```bash
    brew install ollama
    ```

### Optional Prerequisites (for "Web" example)

- Register for a free Brave Search API key [here](https://api-dashboard.search.brave.com/register)
  - 2000 calls/month free, affordable scaling
  - [Brave Search API info](https://brave.com/search/api/)
- Add your key to a .env file in the project root:
    ```bash
    echo "BRAVE_SEARCH_API_KEY=your_key_here" >> .env
    ```

---

## Usage

### Running the Backing Services and LLM

**Option 1 (CPU-only, easiest):**
```bash
docker compose --profile=cpu up -d
```

**Option 2 (GPU, fastest):**
```bash
docker compose up -d
./etc/ollama_entrypoint.sh
```

**Option 3 (GitHub Codespaces):**
1. Open in Codespaces: Click "Code" → "Codespaces" → "Create codespace"
2. Setup environment:
    ```bash
    docker compose --profile=cpu up -d
    ```
3. Access the application:
    - Streamlit App: Auto-opens or check "Ports" tab
    - Open WebUI: Available on port 3000

### Common Docker Compose Commands
```bash
docker compose --profile=cpu up -d      # Start all services
docker compose down                     # Stop all services
docker compose logs -f                  # View service logs
docker compose ps                       # Check service status
docker compose run --rm app pytest tests/ -v   # Run tests
```

### Accessing the Demo App
- Streamlit: [http://localhost:8501/](http://localhost:8501/)
- Open WebUI: [http://localhost:3000/](http://localhost:3000/)

---

## MCP Servers & Standardized Tool Loading

This project includes support for **Model Context Protocol (MCP)** servers, enabling standardized tool loading from OpenAPI specifications.

### PokéAPI MCP Example

The project includes a complete MCP server generated from the PokéAPI OpenAPI specification, demonstrating how to:
- Convert REST APIs to standardized MCP tools
- Deploy tool servers independently via Docker
- Load tools dynamically in LangChain agents
- Test tools through standardized MCP protocols

**What's Included:**
- `mcp-pokemon/` - Generated MCP server for PokéAPI
- `pokeapi-openapi.json` - OpenAPI specification
- `src/example/pokemon_mcp.py` - LangChain integration example
- `tests/integration/test_pokemon_mcp.py` - Integration tests

**Try It:**
1. Run the Streamlit app: `docker compose --profile=cpu up -d`
2. Navigate to [http://localhost:8501/](http://localhost:8501/)
3. Select "Pokemon_MCP" from the example selector
4. Ask questions like "Tell me about Pikachu" or "What are Electric type weaknesses?"

### Regenerating MCP Servers for Other APIs

To create an MCP server for a different OpenAPI-compliant API:

**Prerequisites:**
```bash
# Install Node 20+ and the generator
nvm install 20
npm install -g openapi-mcp-generator
```

**Generate:**
```bash
# From an OpenAPI spec file
openapi-mcp-generator \
  --input path/to/openapi.json \
  --output my-mcp-server \
  --transport streamable-http \
  --port 3002

# From a URL
openapi-mcp-generator \
  --input https://api.example.com/openapi.json \
  --output my-mcp-server \
  --transport streamable-http
```
---

### Ollama Utilities
- List Ollama Models:
    ```bash
    curl http://localhost:11434/api/tags
    ```
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md#api)

### n8n Workflow Automation

n8n is a lightweight workflow automation platform that integrates seamlessly with local services. It's included in the Docker Compose setup for building automation workflows and testing integrations with your LLM infrastructure.

**Getting Started:**
1. Access n8n at [http://localhost:5678/](http://localhost:5678/)
2. Create an account and log in
3. Build your first workflow by combining nodes and connecting services

**Connecting to Ollama:**
To set up an Ollama credential in n8n workflows:
1. In the n8n editor, create a new credential of type "HTTP Request"
2. Use the URL: `http://host.docker.internal:11434`
3. This allows n8n containers to communicate with Ollama across the Docker bridge network

For more details, see the [n8n documentation](https://docs.n8n.io/hosting/installation/server-setups/docker-compose/).

---

## Testing

This application includes a comprehensive test suite that runs in Docker for consistency. Unit tests use mocks, while integration tests connect to real services.

### Quick Start
```bash
# Run all tests
docker compose run --rm app pytest tests/ -v

# Run just unit tests
docker compose run --rm app pytest tests/unit/ -v

# Run integration tests (requires services running)
docker compose run --rm app pytest tests/integration/ -m integration -v

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

---

## Optional: Langfuse Support

This project supports [Langfuse](https://langfuse.com/) for LLM tracing and analytics. **Langfuse is fully optional**—you can run locally, use your cloud credentials, or skip it entirely.

### How to Enable Langfuse (Local or Cloud)

1. **Opt-in with Docker Compose Profile:**
   - To run Langfuse locally, use the `langfuse` profile:
     ```bash
     docker compose --profile=langfuse up -d
     ```
   - If you do not use this profile, Langfuse services will not run.

2. **Configure Environment Variables:**
   - Add the following to your .env file (see below for example):
     - `LANGFUSE_SECRET_KEY`
     - `LANGFUSE_PUBLIC_KEY`
     - `LANGFUSE_BASE_URL`
   - These can point to your local Langfuse instance or your cloud account.

#### Example .env for Local Langfuse
```env
# Langfuse (optional)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=http://host.docker.internal:3001
```

#### Example .env for Cloud Langfuse
```env
# Langfuse (optional)
LANGFUSE_SECRET_KEY=sk-lf-...   # from your cloud dashboard
LANGFUSE_PUBLIC_KEY=pk-lf-...   # from your cloud dashboard
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

> **Note:** If you do not set these variables, Langfuse will be disabled and the app will run without tracing.

### How to Get API and Secret Keys (Local)
1. Start Langfuse locally with the Docker Compose profile above.
2. Visit [http://localhost:3001](http://localhost:3001) in your browser.
3. Register a user and create a project.
4. Copy your **Public Key** and **Secret Key** from the Langfuse dashboard.
5. Add them to your .env file as shown above.

### How to Use Cloud Credentials
1. Sign up at [https://cloud.langfuse.com](https://cloud.langfuse.com).
2. Create a project and get your API keys.
3. Set the .env variables as shown above, using the cloud base URL.

### Disabling Langfuse
If you do not want to use Langfuse, simply leave the variables unset in your .env file. The application will run without tracing or analytics.

---

## GenAI Teacher Chat Mode

This repository includes an integrated AI teaching assistant accessible through GitHub Copilot Chat. The GenAI Teacher provides personalized guidance for learning LLM, RAG, and Agentic AI patterns.

### Activation
In any GitHub interface (VS Code, GitHub.com, or GitHub Mobile), use the `#genai-teacher` chat mode:
```bash
#genai-teacher [your question or request]
```

### Available Teaching Modes

#### 🔍 Explain Mode
Get detailed explanations of demo files and concepts:
```bash
#genai-teacher explain simple_chat.py
#genai-teacher explain RAG patterns
#genai-teacher explain the ReAct agent pattern
```

#### 🎯 Guide Mode
Receive structured learning guidance through topics:
```bash
#genai-teacher guide me through RAG
#genai-teacher guide me through the basics
#genai-teacher guide me through agent development
```

#### 🛤️ Learning Path Mode
Get personalized learning progressions:
```bash
#genai-teacher learning path for beginners
#genai-teacher learning path for RAG
#genai-teacher learning path for agents
```

#### 💡 Implementation Mode
Get help implementing new features:
```bash
#genai-teacher how to create a new RAG example
#genai-teacher implement a custom retriever
#genai-teacher add error handling to my chain
```

### Example Interactions
**Beginner Starting Point:**
```bash
#genai-teacher guide me through the basics
```

**Understanding a Specific File:**
```bash
#genai-teacher explain src/example/agentic_chat.py
```

**Getting Implementation Help:**
```bash
#genai-teacher how to add a new vector database example
```

The GenAI Teacher understands the repository structure, coding patterns, and can provide context-aware guidance tailored to your current learning needs.

---

## Troubleshooting & Codespaces Notes

### Codespaces Notes
- **CPU-Only**: Codespaces uses CPU-only mode (no GPU acceleration)
- **Model Loading**: Initial model downloads may take 5-10 minutes
- **Port Forwarding**: All ports are automatically forwarded and accessible via HTTPS
- **Persistence**: Your workspace persists across sessions

### Adding Brave Search API Key
For the web search example, add your API key to the .env file:
```bash
echo "BRAVE_SEARCH_API_KEY=your_key_here" >> .env
docker compose up -d
./etc/ollama_entrypoint.sh
```

---

## Further Reading

- **Ollama** - Open source app for LLM models, prompts, tools, and functions
  - Website: https://ollama.com/
  - Github: https://github.com/ollama/ollama
- **Open WebUI** - Web UI wrapper for Ollama
  - Website: https://openwebui.com/
  - Github: https://github.com/open-webui/open-webui
- **RAG**
  - [Retrieval Augmented Generation Article](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)
- **LangChain**
  - Website: https://python.langchain.com/docs/get_started/introduction
- **Hugging Face**
  - Website: https://huggingface.co/
  - [LangChain Hugging Face Support](https://python.langchain.com/v0.1/docs/integrations/platforms/huggingface/)
- **Chroma**
  - Website: https://www.trychroma.com/
  - [LangChain Chroma Support](https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/)
- **PGVector**
  - Website: https://github.com/pgvector/pgvector
  - [LangChain PGVector Support](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
- **Wikipedia**
  - Website: https://wikipedia.org
  - [LangChain Wikipedia Support](https://python.langchain.com/docs/integrations/retrievers/wikipedia/)
- **Web - Brave API Search Loader**
  - Website: https://brave.com/search/api/
  - [LangChain BraveSearchLoader Docs](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.brave_search.BraveSearchLoader.html)
- **Arxiv**
  - Website: https://arxiv.org/
  - [LangChain ArxivRetriever Support](https://python.langchain.com/docs/integrations/retrievers/arxiv/)
