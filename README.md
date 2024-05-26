# Generative AI Demo App

## Pre-requisites
* GPT4All - Engine to run LLM models locally or deployed linux against CPU or GPU configurations: https://gpt4all.io/index.html
* At least one model downloaded (tested with mistral-7b-openorca.Q4_0.gguf)
* python3 (3.12+)

## Setting up Virtual Env and Installing Dependencies
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Running app
```
streamlit run src/app.py
```

## Further Reading

* Langchain - Tool to help build custom prompts and embed using vectorDB: https://python.langchain.com/docs/get_started/introduction
    * Langchain GPT4All Support: https://python.langchain.com/docs/integrations/providers/gpt4all
    * Example using GPT4All and LangChain on a folder of pdf files for context: https://medium.com/@vikastiwari708409/how-to-use-gpt4all-llms-with-langchain-to-work-with-pdf-files-f0f0becadcb6
 * Chroma - Vector database: https://www.trychroma.com/
   * Langchain Chroma Support: https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/ 
 * PGVector - Tool allowing storage of vectors in postgresdb: https://github.com/pgvector/pgvector
    * Langchain PGVector Support: https://python.langchain.com/docs/integrations/vectorstores/pgvector
 * Cerebrium - Run Custom LLM code and models on-demand in the cloud: https://www.cerebrium.ai/
