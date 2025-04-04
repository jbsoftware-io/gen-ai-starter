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
    create_llm, format_docs, getCollectionName, loadPDF, writeToTempFile
)


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
DB_URL = os.getenv("DB_URL")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"
assert DB_URL, "DB_URL is not set"


def handle_pgvector(st, model_name):
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

    retrievers = []

    if st.button("Summarize"):
        with st.spinner('Please wait...'):
            if len(source_docs) > 0:
                try:
                    for source_doc in source_docs:
                        vector_store = vectorizePDF(source_doc, model_name)
                        retrievers.append(vector_store.as_retriever())

                    llm = create_llm(model_name)

                    # summarize_prompt = create_summarize_prompt()
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

                    logging.info("Invoking chain")
                    chain = RunnablePassthrough.assign(
                        context=retrieve_docs
                    ).assign(answer=rag_chain_from_docs)

                    result = chain.invoke({"question": search_query})

                    logging.info("Result")
                    logging.info(result)
                    logging.info('-'*30)

                    if not result or not result['answer']:
                        st.warning("No answer was found.")
                    else:
                        st.success(result['answer'])

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
