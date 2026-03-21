"""
PokéAPI MCP Example using Deep Agents

Demonstrates how to use tools generated from OpenAPI specs via MCP servers
with Deep Agents for robust tool use and reasoning.
"""

import asyncio
import json
import os

import streamlit as st
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
POKEMON_MCP_SERVER_HOST = os.getenv("POKEMON_MCP_SERVER_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"
assert POKEMON_MCP_SERVER_HOST, "POKEMON_MCP_SERVER_HOST is not set"


async def get_mcp_tools():
    """Auto-discover tools from MCP server."""
    client = MultiServerMCPClient(
        {
            "pokemon": {
                "transport": "streamable_http",
                "url": f"{POKEMON_MCP_SERVER_HOST}/mcp",
            }
        }
    )
    try:
        mcp_tools = await client.get_tools()

        # Wrap async MCP tools as sync Tool objects for Deep Agents
        wrapped_tools = []
        for mcp_tool in mcp_tools:
            def make_wrapper(tool):
                def sync_wrapper(tool_input):
                    """Call async tool synchronously."""
                    # Handle both string and dict inputs
                    if isinstance(tool_input, str):
                        try:
                            if tool_input.strip().startswith('{'):
                                input_dict = json.loads(tool_input)
                            else:
                                # Wrap plain string in first schema field
                                has_schema = (hasattr(tool, 'args_schema') and
                                              tool.args_schema)
                                if has_schema:
                                    try:
                                        # Try Pydantic model fields
                                        has_fields = hasattr(tool.args_schema,
                                                             '__fields__')
                                        if has_fields:
                                            schema = tool.args_schema
                                            field_names = list(
                                                schema.__fields__.keys())
                                        # Try dict keys
                                        elif isinstance(tool.args_schema,
                                                        dict):
                                            schema = tool.args_schema
                                            field_names = list(
                                                schema.get(
                                                    'properties',
                                                    {}).keys())
                                        else:
                                            field_names = []
                                        if field_names:
                                            input_dict = {
                                                field_names[0]: tool_input}
                                        else:
                                            input_dict = {"input":
                                                          tool_input}
                                    except (AttributeError, TypeError):
                                        input_dict = {"input": tool_input}
                                else:
                                    input_dict = {"input": tool_input}
                        except json.JSONDecodeError:
                            # Fallback: wrap as input
                            has_schema = (hasattr(tool, 'args_schema') and
                                          tool.args_schema)
                            if has_schema:
                                try:
                                    has_fields = hasattr(tool.args_schema,
                                                         '__fields__')
                                    if has_fields:
                                        schema = tool.args_schema
                                        field_names = list(
                                            schema.__fields__.keys())
                                    elif isinstance(tool.args_schema, dict):
                                        schema = tool.args_schema
                                        field_names = list(
                                            schema.get(
                                                'properties',
                                                {}).keys())
                                    else:
                                        field_names = []
                                    if field_names:
                                        input_dict = {field_names[0]:
                                                      tool_input}
                                    else:
                                        input_dict = {"input": tool_input}
                                except (AttributeError, TypeError):
                                    input_dict = {"input": tool_input}
                            else:
                                input_dict = {"input": tool_input}
                    else:
                        input_dict = tool_input

                    # Call async tool synchronously
                    result = asyncio.run(tool.ainvoke(input_dict))
                    return str(result)
                return sync_wrapper

            # Create sync Tool wrapper
            wrapped = Tool(
                name=mcp_tool.name,
                description=mcp_tool.description,
                func=make_wrapper(mcp_tool),
            )
            wrapped_tools.append(wrapped)

        return wrapped_tools, client
    except Exception as e:
        st.error(f"Failed to load MCP tools: {e}")
        raise


def create_chain(model_name: str, tools):
    """Create a Deep Agents agent with auto-discovered MCP tools."""
    if not tools:
        st.error("No tools available from MCP server")
        return None

    system_prompt = (
        "You are a Pokemon information assistant. "
        "You have access to tools to look up Pokemon information.\n\n"
        "When a user asks about Pokemon, use the available tools "
        "to find accurate information.\n"
        "- Use getPokemon to get detailed Pokemon information\n"
        "- Use getPokemonSpecies for species-specific data\n"
        "- Use getType for type information\n"
        "- Use getAbility for ability details\n"
        "- Use getMove for move information\n\n"
        "Answer questions thoroughly using the tool results."
    )

    # Create deep agent with Ollama model
    agent = create_deep_agent(
        model=f"ollama:{model_name}",
        tools=tools,
        system_prompt=system_prompt
    )

    return agent


def process_query(agent, query: str):
    """Process a user query through the deep agent."""
    try:
        # Deep Agents expects messages format
        response = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        # Extract the final answer from the last message
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            # Handle both dict and message object
            if isinstance(last_message, dict):
                answer = last_message.get("content", "No response")
            else:
                # It's a message object (AIMessage, etc.)
                answer = getattr(last_message, 'content',
                                 str(last_message))
        else:
            answer = "No response"

        return {
            "answer": answer,
            "intermediate_steps": response.get("intermediate_steps", []),
        }
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error processing query: {error_msg}")
        return {"answer": f"Error: {error_msg}",
                "intermediate_steps": []}


def handle_pokemon_mcp(st, model_name: str, langfuse_handler=None):
    """Handle the Pokémon MCP example UI."""
    st.header("🎮 PokéAPI via MCP Server (Deep Agents)")
    st.markdown(
        """
This example demonstrates using tools auto-discovered from an MCP server
with Deep Agents. The MCP server exposes PokéAPI endpoints as
standardized tools that Deep Agents can use intelligently.

**Available Tools:** Auto-discovered from the MCP server:
- Get Pokémon info (stats, abilities, moves)
- Look up Pokémon species details
- Query type information and effectiveness
- Find abilities and their effects
- Get move data and power
"""
    )

    # Load tools from MCP server (async)
    with st.spinner("Connecting to MCP server and discovering tools..."):
        try:
            tools, _ = asyncio.run(get_mcp_tools())
            if tools:
                st.success(
                    f"✅ Connected! Auto-discovered {len(tools)} tools")

                st.divider()
            else:
                st.warning("⚠️ Connected but no tools found")
                return
        except Exception as e:
            st.error(f"❌ Failed to connect to MCP server: {e}")
            st.info(
                f"Make sure MCP is running: {POKEMON_MCP_SERVER_HOST}")
            return

    # Create the agent
    agent = create_chain(model_name, tools)
    if not agent:
        return

    # Initialize session state
    if "pokemon_messages" not in st.session_state:
        st.session_state.pokemon_messages = []
        st.session_state.pokemon_messages.append({
            "role": "assistant",
            "content": (
                "Ask me anything about Pokémon! I can look up "
                "Pokémon info, stats, abilities, moves, types, "
                "and more.")
        })

    # Display chat history
    for message in st.session_state.pokemon_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if (user_input := st.chat_input(
            "Ask about Pokémon (e.g., 'Tell me about Pikachu')")):
        st.chat_message("user").markdown(user_input)
        st.session_state.pokemon_messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Querying PokéAPI via MCP tools..."):
            result = process_query(agent, user_input)

            # Display response
            st.chat_message("assistant").markdown(result["answer"])
            st.session_state.pokemon_messages.append({
                "role": "assistant",
                "content": result["answer"]
            })
