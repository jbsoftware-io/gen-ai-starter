from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb, os, tempfile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from internal.prompts import create_summarize_prompt
from internal.util import create_llm, format_docs


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"
assert CHROMA_HOST, "CHROMA_HOST is not set"
assert CHROMA_PORT, "CHROMA_PORT is not set"

def handle_chroma(st, model_name):
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=True, anonymized_telemetry=False))
    # chroma_client.reset()  # resets the database
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
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, source_doc.name)
                    with open(path, "wb") as f:
                        f.write(source_doc.getvalue())

                    loader = PyPDFLoader(
                        file_path=path
                    )

                    data = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter()
                    all_splits = text_splitter.split_documents(data)

                    collection_name = os.path.basename(path)

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
                        print("Adding documents")
                        vectorstore.add_documents(all_splits)

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