# Generative AI Demo App

## Pre-requisites

- Docker Engine installed and configured to have > 12 GB memory
- ollama* - Only if using run option 2
```
brew install ollama
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

- RAG - Retrieval Augmented Generation: https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/
- LangChain - Tool to help build custom prompts and embed using vectorDB: https://python.langchain.com/docs/get_started/introduction
  - LangChain GPT4All Support: https://python.langchain.com/docs/integrations/providers/gpt4all
  - Example using GPT4All and LangChain on a folder of pdf files for context: https://medium.com/@vikastiwari708409/how-to-use-gpt4all-llms-with-langchain-to-work-with-pdf-files-f0f0becadcb6
- Hugging Face - Open Source ML/AI Community Hub with tons of models for various use cases: https://huggingface.co/
  - LangChain Hugging Face Support: https://python.langchain.com/v0.1/docs/integrations/platforms/huggingface/
- Chroma - Vector database: https://www.trychroma.com/
  - LangChain Chroma Support: https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/
- PGVector - Tool allowing storage of vectors in postgresdb: https://github.com/pgvector/pgvector
  - LangChain PGVector Support: https://python.langchain.com/docs/integrations/vectorstores/pgvector
