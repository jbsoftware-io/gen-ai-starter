import os
import uuid

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama

from internal.custom_retrievers import ArxivRetriever, WikipediaAPIWrapper, wikipedia_query_run
from internal.prompts import create_agentic_react_prompt

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


def get_tools():
    tools = [
        ArxivRetriever(load_max_docs=3, get_full_documents=True),
        wikipedia_query_run,
    ]
    return tools


def get_wikipedia_search_tool(top_k_results=1, doc_content_chars_max=500):
    # Return the wikipedia_query_run tool directly
    return wikipedia_query_run


def handle_agentic_chat(st, model_name, langfuse_handler=None):
    prompt = create_agentic_react_prompt()
    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_HOST)
    tools = get_tools()
    agent = create_react_agent(llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=100,
        max_iterations_per_tool=5,
        return_intermediate_steps=True,
    )

    # start a new chat session
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", "content": "How can I help you?"})
        st.session_state.session_id = str(uuid.uuid4())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if input := st.chat_input("Type here..."):
        st.chat_message("user").markdown(input)
        st.session_state.messages.append({
            "role": "user", "content": input})

        with st.spinner("Processing, please wait..."):
            config = {
                "callbacks": [langfuse_handler] if langfuse_handler else None,
            }
            response = agent_executor.invoke({
                "input": input,
                "chat_history": st.session_state.messages,
            }, config=config)

        st.chat_message("assistant").markdown(response['output'])
        st.chat_message("assistant").markdown(
            f"Intermediate steps:\n{response['intermediate_steps']}")
        st.session_state.messages.append({
            "role": "assistant", "content": response['output']})
