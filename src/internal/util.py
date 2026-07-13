import os
import tempfile

from dotenv import load_dotenv
from langchain_classic.callbacks.streaming_stdout import \
    StreamingStdOutCallbackHandler
from pypdf import PdfReader
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


def create_llm(model_name: str):
    # callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # verbose is required to pass to the callback manager
    llm = OllamaLLM(
        base_url=OLLAMA_HOST,
        model=model_name,
        callbacks=callbacks)
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def getCollectionName(path, model_name, max_length=63):
    clean_model_name = strip_non_alphanumeric(model_name)
    clean_basename = strip_non_alphanumeric(os.path.basename(path))
    return f"{clean_model_name}{clean_basename}"[:max_length]


def loadPDF(path):
    """Load and split PDF documents using pypdf."""
    reader = PdfReader(path)
    
    # Extract text from all pages
    text_content = ""
    for page_num, page in enumerate(reader.pages):
        text_content += f"\n--- Page {page_num + 1} ---\n"
        text_content += page.extract_text()
    
    # Create a Document object
    doc = Document(
        page_content=text_content,
        metadata={"source": path}
    )
    
    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )
    return text_splitter.split_documents([doc])


def print_context(st, result):
    if 'context' in result:
        st.markdown("### Context")
        context_docs = result['context']
        for doc in context_docs:
            st.markdown(f"```{doc.metadata}```")


def strip_non_alphanumeric(s):
    return ''.join(c for c in s if c.isalnum())


def writeToTempFile(source_doc):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, source_doc.name)
    with open(path, "wb") as f:
        f.write(source_doc.getvalue())
    return path
