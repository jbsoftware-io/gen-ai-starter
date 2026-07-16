import os
import uuid

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

from internal.custom_retrievers import arxiv_query_run, wikipedia_query_run


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


def get_tools():
    tools = [
        arxiv_query_run,
        wikipedia_query_run,
    ]
    return tools


def get_wikipedia_search_tool(top_k_results=1, doc_content_chars_max=500):
    # Return the wikipedia_query_run tool directly
    return wikipedia_query_run


def handle_agentic_chat(st, model_name, langfuse_handler=None):
    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_HOST)
    tools = get_tools()
    agent = create_react_agent(llm, tools=tools)

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
            response = agent.invoke({
                "messages": st.session_state.messages,
                "input": input,
            }, config=config)

        # Extract output from response
        output = (response.get('output') or
                  (response.get('messages', [])[-1].content
                   if response.get('messages') else "No response"))

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(output)

            # Try to show tool calls and reasoning from the agent
            if isinstance(response, dict) and response.get('messages'):
                # Extract intermediate messages which show tool calls
                tool_info = []
                for msg in response.get('messages', []):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_info.append(f"📌 Called tool: **{tool_name}**")
                    # Check for tool results in content
                    if (hasattr(msg, 'content') and
                            isinstance(msg.content, str)):
                        if (len(msg.content) > 50 and
                                any(x in msg.content.lower() for x in
                                    ['wikipedia', 'arxiv', 'brave', 'search'])):
                            tool_info.append(
                                f"📋 Retrieved context: {msg.content[:200]}...")

                if tool_info:
                    with st.expander("🔍 Tool Calls & Context"):
                        for info in tool_info:
                            st.markdown(info)

        st.session_state.messages.append({
            "role": "assistant", "content": output})
