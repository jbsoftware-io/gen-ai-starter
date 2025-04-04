import os
import tempfile
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def strip_non_alphanumeric(s):
    return ''.join(c for c in s if c.isalnum())


def writeToTempFile(source_doc):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, source_doc.name)
    with open(path, "wb") as f:
        f.write(source_doc.getvalue())
    return path


def loadPDF(path):
    loader = PyPDFLoader(
        file_path=path
    )

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )
    return text_splitter.split_documents(data)


def getCollectionName(path, model_name, max_length=63):
    clean_model_name = strip_non_alphanumeric(model_name)
    clean_basename = strip_non_alphanumeric(os.path.basename(path))
    return f"{clean_model_name}{clean_basename}"[:max_length]
