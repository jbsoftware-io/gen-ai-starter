from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, tempfile, streamlit as st  # noqa: E401
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from internal.prompts import create_summaryize_prompt_v2
from internal.util import create_llm, format_docs


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
                        temp_dir = tempfile.mkdtemp()
                        path = os.path.join(temp_dir, source_doc.name)
                        with open(path, "wb") as f:
                            f.write(source_doc.getvalue())

                        loader = PyPDFLoader(
                            file_path=path
                        )

                        data = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=2000,
                            chunk_overlap=0
                        )
                        all_splits = text_splitter.split_documents(data)

                        col_name = f"${os.path.basename(path)} {model_name.replace(".", "").replace(":", "")}"

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
                            collection_store = general_store.get_collection(
                                session)
                            collection, created = collection_store.get_or_create(  # noqa: E501
                                session, col_name)

                            print(f"Collection {col_name} created: {created}")

                            vector_store = PGVector(
                                embeddings=embeddings,
                                connection=DB_URL,
                                collection_name=col_name,
                                use_jsonb=True,
                            )
                            if created:
                                print("Adding documents")
                                vector_store.add_documents(all_splits)

                            retrievers.append(vector_store.as_retriever())

                    llm = create_llm(model_name)

                    # summarize_prompt = create_summarize_prompt()
                    summarize_prompt = create_summaryize_prompt_v2()

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

                    print("Invoking chain")
                    chain = RunnablePassthrough.assign(
                        context=retrieve_docs
                    ).assign(answer=rag_chain_from_docs)

                    result = chain.invoke({"question": search_query})

                    print("Result")
                    print(result)
                    print('-'*30)

                    if not result or not result['answer']:
                        st.warning("No answer was found.")
                    else:
                        st.success(result['answer'])

                except Exception as e:
                    st.exception(f"An error occurred: {e}")