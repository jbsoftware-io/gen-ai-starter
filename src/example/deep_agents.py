import os
import uuid

from dotenv import load_dotenv
from deepagents import create_deep_agent

from internal.brave_client import brave_search
from internal.custom_retrievers import arxiv_query_run, wikipedia_query_run
from internal.prompts import create_deep_agents_system_prompt
from internal.logger import logger

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


def get_tools():
    """Get tools for the deep agent."""
    tools = [
        arxiv_query_run,
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
                with st.chat_message("assistant"):
                    st.markdown(final_message)

                    # Show tool calls and reasoning steps
                    tool_info = []
                    step_info = []

                    # Try to extract intermediate steps (main approach)
                    if isinstance(response, dict):
                        if "intermediate_steps" in response and response["intermediate_steps"]:
                            steps = response["intermediate_steps"]

                            for i, step in enumerate(steps):
                                try:
                                    # intermediate_steps can be tuples of (action, result)
                                    if isinstance(step, (list, tuple)) and len(step) >= 2:
                                        action, result = step[0], step[1]
                                    else:
                                        action = step
                                        result = None

                                    # Extract tool name
                                    if hasattr(action, 'tool'):
                                        tool_name = action.tool
                                        tool_input = getattr(action, 'tool_input', "")
                                        tool_info.append(f"📌 **Tool**: {tool_name}")
                                        if tool_input:
                                            tool_info.append(f"   Query: {str(tool_input)[:100]}")

                                    # Extract result
                                    if result:
                                        result_str = str(result)
                                        if len(result_str) > 150:
                                            step_info.append(f"📋 **Result {i+1}**: {result_str[:150]}...")
                                        else:
                                            step_info.append(f"📋 **Result {i+1}**: {result_str}")
                                except Exception as e:
                                    logger.debug(f"Error extracting step {i}: {e}")

                        # Try extracting from messages if intermediate_steps not available
                        if not tool_info and "messages" in response:
                            for msg in response["messages"]:
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get('name', 'unknown')
                                        tool_info.append(f"📌 **Tool**: {tool_name}")

                    # Display tools and steps in expander
                    if tool_info or step_info:
                        with st.expander("🔍 Tool Calls & Reasoning"):
                            if tool_info:
                                st.markdown("**Tools Used:**")
                                for info in tool_info:
                                    st.markdown(info)
                            else:
                                st.markdown("*No tool details extracted*")

                            if step_info:
                                st.markdown("**Retrieved Context:**")
                                for info in step_info:
                                    st.markdown(info)

                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_message
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")
