import os
import uuid

from dotenv import load_dotenv
from deepagents import create_deep_agent

from internal.brave_client import brave_search
from internal.custom_retrievers import ArxivRetriever, wikipedia_query_run
from internal.prompts import create_deep_agents_system_prompt

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


def get_tools():
    """Get tools for the deep agent."""
    tools = [
        ArxivRetriever(load_max_docs=3, get_full_documents=True),
        wikipedia_query_run,
        brave_search,
    ]
    return tools


def get_wikipedia_search_tool(top_k_results=1, doc_content_chars_max=500):
    """Create and return the Wikipedia search tool."""
    return wikipedia_query_run


def create_deep_agents_chain(model_name):
    """Create and return the deep agents chain for easier testing."""
    system_prompt = create_deep_agents_system_prompt()
    tools = get_tools()

    # Create deep agent with Ollama model prefix
    # Deep Agents requires the format: provider:model-name
    agent = create_deep_agent(
        model=f"ollama:{model_name}",
        tools=tools,
        system_prompt=system_prompt
    )

    return agent


def process_deep_agents_query(agent, user_input, langfuse_handler=None):
    """Process a query with the deep agents chain.

    Args:
        agent: The compiled deep agents graph
        user_input: The user's query string
        langfuse_handler: Optional Langfuse callback handler for tracing

    Returns:
        dict: Response containing the final answer
    """
    config = {
        "callbacks": [langfuse_handler] if langfuse_handler else None,
    }

    response = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    }, config=config)

    return response


def handle_deep_agents(st, model_name, langfuse_handler=None):
    """Handle Deep Agents interaction with Streamlit UI and session state.

    Args:
        st: Streamlit module
        model_name: Name of the Ollama model to use
        langfuse_handler: Optional Langfuse callback handler for tracing
    """
    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "How can I help you with research today?"
        })
        st.session_state.session_id = str(uuid.uuid4())

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_input := st.chat_input("Type here..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Processing, please wait..."):
            try:
                agent = create_deep_agents_chain(model_name)
                response = process_deep_agents_query(
                    agent,
                    user_input,
                    langfuse_handler=langfuse_handler
                )

                # Extract the final message from the response
                if "messages" in response and response["messages"]:
                    final_message = response["messages"][-1].content
                else:
                    final_message = str(response)

                # Display assistant response
                st.chat_message("assistant").markdown(final_message)

                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_message
                })

                # Display intermediate steps if available
                if "intermediate_steps" in response:
                    with st.expander("Intermediate Steps"):
                        st.markdown(str(response["intermediate_steps"]))

            except Exception as e:
                st.error(f"An error occurred: {e}")
