import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from internal.logger import logger
from internal.prompts import create_summarize_prompt
from internal.util import (create_llm, format_docs, getCollectionName, loadPDF,
                           writeToTempFile)

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"
assert CHROMA_HOST, "CHROMA_HOST is not set"
assert CHROMA_PORT, "CHROMA_PORT is not set"


def create_chroma_chain(vectorstore, model_name):
    """
    Create and return the Chroma RAG chain.
    This function is separated to make testing easier.
    """
    llm = create_llm(model_name)
    summarize_prompt = create_summarize_prompt()

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"])))
        | summarize_prompt
        | llm
        | StrOutputParser()
    )

    retrieve_docs = (
        lambda x: x["question"]
    ) | vectorstore.as_retriever()

    chain = RunnablePassthrough.assign(
        context=retrieve_docs
    ).assign(answer=rag_chain_from_docs)

    return chain


def process_chroma_query(chain, search_query, langfuse_handler=None):
    """
    Execute the Chroma query and return the result.
    This function is separated to make testing easier.
    """
    logger.info("Invoking chain")
    config = {
        "callbacks": [langfuse_handler] if langfuse_handler else None,
    }
    result = chain.invoke({"question": search_query}, config=config)
    logger.info("Result")
    logger.info(result)
    logger.info('-'*30)

    return result


def handle_chroma(st, model_name, langfuse_handler=None):
    """
    Handle Chroma UI and orchestrate the query processing.
    This function now has simpler logic that's easier to test.
    """
    source_doc = st.file_uploader(
        "Source PDF Document",
        label_visibility="collapsed",
        type="pdf"
    )
    search_query = st.text_input(
        "Question",
        placeholder="Ask a question about the uploaded document."
    )

    if st.button("Summarize"):
        with st.spinner('Please wait...'):
            if source_doc:
                try:
                    # Vectorize PDF (this can be mocked easily)
                    vectorstore = vectorizePDF(source_doc, model_name)

                    # Create chain (this can be mocked easily)
                    chain = create_chroma_chain(vectorstore, model_name)

                    # Process query (this can be mocked easily)
                    result = process_chroma_query(
                        chain, search_query, langfuse_handler=langfuse_handler)

                    # Handle result (this is simple business logic)
                    if not result or not result['answer']:
                        st.warning("No answer was found.")
                    else:
                        st.success(result['answer'])

                except Exception as e:
                    st.exception(f"An error occurred: {e}")


def vectorizePDF(source_doc, model_name):
    path = writeToTempFile(source_doc)
    docs = loadPDF(path)
    collection_name = getCollectionName(path, model_name)
    vectorstore = None

    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=True, anonymized_telemetry=False))
    # chroma_client.reset()  # resets the database

    collection = chroma_client.create_collection(
        collection_name, get_or_create=True)

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=model_name,
        show_progress=True)

    # tell LangChain to use our client and collection name
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # if the collection is empty, add the documents again
    if collection.count() == 0:
        logger.info("Adding documents")
        vectorstore.add_documents(docs)

    return vectorstore
