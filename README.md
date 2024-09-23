# Generative AI Demo App

## Pre-requisites

- python3 (3.12+)
- brew install postgresql libpq
- Docker Compose installed and configured to have > 12 GB memory (16 GB recommended)

## Setting up Virtual Env and Installing Dependencies

```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Running the Demo

### Docker Compose

This application now uses docker container to run ollama, downloads mistral:7b and llama3.1 models, then starts up. The startup can take > 5 minutes depending on your internet speed. Also, the first invocation of any model will take some time, but should be fairly quick after that.

```
docker compose up -d
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

## Running the app

```
streamlit run src/app.py
```

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
- Cerebrium - Run Custom LLM code and models on-demand in the cloud: https://www.cerebrium.ai/
