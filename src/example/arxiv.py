import logging

from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from internal.prompts import create_summarize_prompt_v2
from internal.util import create_llm, format_docs, print_context


def handle_arxiv(st, model_name, langfuse_handler=None):
    search_query = st.text_input(
        "Question",
        placeholder="Ask a question about scientific or engineering research."
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

                retriever = ArxivRetriever(
                    load_max_docs=3,
                    get_full_documents=True,
                )

                retrieve_docs = (
                    lambda x: x["question"]
                ) | retriever

                logging.info("Invoking chain")
                chain = RunnablePassthrough.assign(
                    context=retrieve_docs
                ).assign(answer=rag_chain_from_docs)
                config = {
                    "callbacks": [
                        langfuse_handler
                    ] if langfuse_handler else None,
                }

                result = chain.invoke({"question": search_query},
                                      config=config)

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
