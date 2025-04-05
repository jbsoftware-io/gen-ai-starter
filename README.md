# Generative AI Starter

## Purpose
This repository can be used as a starting point for building custom LLM applications.  It incorporates Ollama, Open WebUI, Langchain, Streamlit, Chroma, and PGVector using docker to containerize the application and docker compose to run the various service dependencies.

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

## Example Screenshots

### Static Context:

![Static Context Example](/etc/Static_Context_Example.png)

### JSON Context:

![JSON Context Example](/etc/JSON_Context_Example.png)

### Chroma DB Context:

![Chroma_DB_Example](/etc/Chroma_PDF_Example.png)

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
  - LangChain GPT4All Support: https://python.langchain.com/docs/integrations/providers/gpt4all
  - Example using GPT4All and LangChain on a folder of pdf files for context: https://medium.com/@vikastiwari708409/how-to-use-gpt4all-llms-with-langchain-to-work-with-pdf-files-f0f0becadcb6
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
