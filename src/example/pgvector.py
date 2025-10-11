import logging
from dotenv import load_dotenv
import os
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from internal.prompts import create_summarize_prompt_v2
from internal.util import (
    create_llm, format_docs, getCollectionName, loadPDF, writeToTempFile,
    print_context
)


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
DB_URL = os.getenv("DB_URL")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"
assert DB_URL, "DB_URL is not set"


def create_pgvector_chain(model_name, retrievers):
    """
    Create and return the PGVector RAG chain.
    This function is separated to make testing easier.
    """
    llm = create_llm(model_name)
    summarize_prompt = create_summarize_prompt_v2()

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"])))
        | summarize_prompt
        | llm
        | StrOutputParser()
    )

    retrieve_docs = (
        lambda x: x["question"]
    ) | MergerRetriever(retrievers=retrievers)

    chain = RunnablePassthrough.assign(
        context=retrieve_docs
    ).assign(answer=rag_chain_from_docs)

    return chain


def process_pgvector_query(chain, search_query):
    """
    Execute the PGVector query and return the result.
    This function is separated to make testing easier.
    """
    logging.info("Invoking chain")
    result = chain.invoke({"question": search_query})

    logging.info("Result")
    logging.info(result)
    logging.info('-'*30)

    return result


def handle_pgvector(st, model_name):
    """
    Handle PGVector UI and orchestrate the query processing.
    This function now has simpler logic that's easier to test.
    """
    source_docs = st.file_uploader(
        "Source PDF Document",
        label_visibility="collapsed",
        type="pdf",
        accept_multiple_files=True
    )
    search_query = st.text_input(
        "Question",
        placeholder="Ask a question about the uploaded document."
    )

    if st.button("Summarize"):
        with st.spinner('Please wait...'):
            try:
                if len(source_docs) == 0:
                    st.warning("Please upload at least one PDF document.")
                    return

                if not search_query:
                    st.warning("Please enter a question.")
                    return

                # Create retrievers from documents
                retrievers = []
                for source_doc in source_docs:
                    vector_store = vectorizePDF(source_doc, model_name)
                    retrievers.append(vector_store.as_retriever())

                # Create chain (this can be mocked easily)
                chain = create_pgvector_chain(model_name, retrievers)

                # Process query (this can be mocked easily)
                result = process_pgvector_query(chain, search_query)

                # Handle result (this is simple business logic)
                if not result or not result['answer']:
                    st.warning("No answer was found.")
                else:
                    st.success(result['answer'])
                    print_context(st, result)

            except Exception as e:
                st.exception(f"An error occurred: {e}")


def vectorizePDF(source_doc, model_name):
    path = writeToTempFile(source_doc)
    docs = loadPDF(path)
    col_name = getCollectionName(path, model_name)
    vector_store = None

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=model_name,
        show_progress=True)

    general_store = PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        use_jsonb=True,
    )

    with general_store.session_maker() as session:
        # if the collection is empty, add the documents
        collection_store = general_store.get_collection(session)  # noqa: E501
        _, created = collection_store.get_or_create(session, col_name)  # noqa: E501

        logging.info(f"Collection {col_name} created: {created}")

        vector_store = PGVector(
            embeddings=embeddings,
            connection=DB_URL,
            collection_name=col_name,
            use_jsonb=True,
        )
        if created:
            logging.info("Adding documents")
            vector_store.add_documents(docs)

    return vector_store
