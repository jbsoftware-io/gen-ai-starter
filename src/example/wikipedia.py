import logging

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from internal.prompts import create_summarize_prompt_v2
from internal.util import create_llm, format_docs, print_context


def create_wikipedia_chain(model_name):
    """
    Create and return the Wikipedia RAG chain.
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
    ) | WikipediaRetriever()
    chain = RunnablePassthrough.assign(
        context=retrieve_docs
    ).assign(answer=rag_chain_from_docs)

    return chain


def process_wikipedia_query(chain, search_query, langfuse_handler=None):
    """
    Execute the Wikipedia query and return the result.
    This function is separated to make testing easier.
    """
    logging.info("Invoking chain")
    config = {
        "callbacks": [langfuse_handler] if langfuse_handler else None,
    }
    result = chain.invoke({"question": search_query}, config=config)

    logging.info("Result")
    logging.info(result)
    logging.info('-'*30)

    return result


def handle_wikipedia(st, model_name, langfuse_handler=None):
    """
    Handle Wikipedia UI and orchestrate the query processing.
    This function now has simpler logic that's easier to test.
    """
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

                # Create chain (this can be mocked easily)
                chain = create_wikipedia_chain(model_name)

                # Process query (this can be mocked easily)
                result = process_wikipedia_query(
                    chain, search_query, langfuse_handler=langfuse_handler)

                # Handle result (this is simple business logic)
                if not result or not result['answer']:
                    st.warning("No answer was found.")
                else:
                    st.success(result['answer'])
                    print_context(st, result)

            except Exception as e:
                st.exception(f"An error occurred: {e}")
