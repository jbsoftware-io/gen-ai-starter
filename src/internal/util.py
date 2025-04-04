import os
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"

def create_llm(model_name: str):
    # callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # verbose is required to pass to the callback manager
    llm = Ollama(
        base_url=OLLAMA_HOST,
        model=model_name,
        callbacks=callbacks)
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)