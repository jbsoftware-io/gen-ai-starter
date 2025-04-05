import logging
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import BraveSearchLoader
from internal.prompts import create_summarize_prompt_v2
from internal.util import create_llm, format_docs, print_context


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
DB_URL = os.getenv("DB_URL")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"
assert DB_URL, "DB_URL is not set"


def handle_web(st, model_name):
    if not BRAVE_SEARCH_API_KEY:
        st.warning("BRAVE_SEARCH_API_KEY not set, refer to README Optional Pre-requisites for instructions.")  # noqa: E501
        return

    search_query = st.text_input(
        "Question",
        placeholder="Ask a question about any public information."
    )

    if st.button("Summarize"):
        with st.spinner('Please wait...'):
            try:
                if not search_query:
                    st.warning("Please enter a question.")
                    return

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

                loader = BraveSearchLoader(
                    query=search_query,
                    api_key=BRAVE_SEARCH_API_KEY,
                    search_kwargs={"count": 3}
                )
                docs = loader.load()
                logging.info(f"Loaded {len(docs)} documents")

                logging.info("Invoking chain")
                chain = RunnablePassthrough.assign(
                    context=lambda x: docs
                ).assign(answer=rag_chain_from_docs)

                result = chain.invoke({"question": search_query})

                logging.info("Result")
                logging.info(result)
                logging.info('-'*30)

                if not result or not result['answer']:
                    st.warning("No answer was found.")
                else:
                    st.success(result['answer'])
                    print_context(st, result)
            except Exception as e:
                st.exception(f"An error occurred: {e}")
