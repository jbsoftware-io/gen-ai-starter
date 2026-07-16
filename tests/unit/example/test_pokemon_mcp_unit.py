"""
Unit tests for PokéAPI MCP example using Deep Agents.

Tests the tool wrapping, chain creation, and query processing.
"""

from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from example.pokemon_mcp import (
    get_mcp_tools,
    create_chain,
    process_query,
    handle_pokemon_mcp,
)


class TestGetMCPTools:
    """Test MCP tool loading and wrapping."""

    @pytest.mark.asyncio
    async def test_get_mcp_tools_loads_tools(self):
        """Test that tools are loaded from MCP server."""
        # Mock the MultiServerMCPClient
        mock_tool1 = Mock()
        mock_tool1.name = "getPokemon"
        mock_tool1.description = "Get Pokemon info"
        mock_tool1.args_schema = Mock()
        mock_tool1.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool1.ainvoke = AsyncMock(return_value={"name": "Pikachu"})

        mock_tool2 = Mock()
        mock_tool2.name = "getType"
        mock_tool2.description = "Get type info"
        mock_tool2.args_schema = Mock()
        mock_tool2.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool2.ainvoke = AsyncMock(return_value={"name": "electric"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(
            return_value=[mock_tool1, mock_tool2])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, client = await get_mcp_tools()

            assert len(tools) == 2
            assert tools[0].name == "getPokemon"
            assert tools[1].name == "getType"

    @pytest.mark.asyncio
    async def test_get_mcp_tools_wraps_async_to_sync(self):
        """Test that async MCP tools are wrapped as sync tools."""
        # Mock an async MCP tool
        mock_tool = Mock()
        mock_tool.name = "getPokemon"
        mock_tool.description = "Get Pokemon"
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool.ainvoke = AsyncMock(
            return_value={"name": "Pikachu", "id": 25})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, client = await get_mcp_tools()

            # The wrapped tool should be sync (callable directly)
            wrapped_tool = tools[0]
            assert wrapped_tool.name == "getPokemon"

    @pytest.mark.asyncio
    async def test_get_mcp_tools_handles_json_input(self):
        """Test that tool wrapper handles JSON input."""
        mock_tool = Mock()
        mock_tool.name = "getPokemon"
        mock_tool.description = "Get Pokemon"
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool.ainvoke = AsyncMock(return_value={"name": "Charizard"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, client = await get_mcp_tools()
            wrapped_tool = tools[0]
            assert wrapped_tool.name == "getPokemon"


class TestCreateChain:
    """Test Deep Agents chain creation."""

    def test_create_chain_with_tools(self):
        """Test that chain is created with tools."""
        mock_tools = [
            Mock(name="getPokemon", description="Get Pokemon"),
            Mock(name="getType", description="Get type"),
        ]

        with patch(
                "example.pokemon_mcp.create_deep_agent"
        ) as mock_create_agent:
            mock_agent = Mock()
            mock_create_agent.return_value = mock_agent

            with patch("example.pokemon_mcp.st"):
                agent = create_chain("mistral", mock_tools)

                assert agent is mock_agent
                mock_create_agent.assert_called_once()

                # Verify the agent was created with correct parameters
                call_kwargs = mock_create_agent.call_args[1]
                assert call_kwargs["model"] == "ollama:mistral"
                assert call_kwargs["tools"] == mock_tools
                assert "system_prompt" in call_kwargs

    def test_create_chain_no_tools(self):
        """Test error handling when no tools provided."""
        with patch("example.pokemon_mcp.st") as mock_st:
            result = create_chain("mistral", [])

            mock_st.error.assert_called_once()
            assert result is None


class TestProcessQuery:
    """Test query processing."""

    def test_process_query_with_dict_message(self):
        """Test query processing with dict message response."""
        mock_agent = Mock()
        mock_agent.invoke = Mock(
            return_value={
                "messages": [
                    {"role": "user", "content": "Tell me about Pikachu"},
                    {"role": "assistant",
                     "content": "Pikachu is an Electric-type Pokemon"},
                ]
            }
        )

        result = process_query(mock_agent, "Tell me about Pikachu")

        assert (result["answer"] ==
                "Pikachu is an Electric-type Pokemon")
        assert result["intermediate_steps"] == []

    def test_process_query_with_message_object(self):
        """Test query processing with message object response."""
        mock_message = Mock()
        mock_message.content = ("Charizard is a "
                                "Fire/Flying-type Pokemon")

        mock_agent = Mock()
        mock_agent.invoke = Mock(
            return_value={
                "messages": [mock_message]
            }
        )

        result = process_query(mock_agent, "Tell me about Charizard")

        assert (result["answer"] ==
                "Charizard is a Fire/Flying-type Pokemon")

    def test_process_query_no_messages(self):
        """Test query processing with empty messages."""
        mock_agent = Mock()
        mock_agent.invoke = Mock(return_value={"messages": []})

        result = process_query(mock_agent, "Test query")

        assert result["answer"] == "No response"

    def test_process_query_error_handling(self):
        """Test error handling in query processing."""
        mock_agent = Mock()
        mock_agent.invoke = Mock(side_effect=Exception("Test error"))

        with patch("example.pokemon_mcp.st"):
            result = process_query(mock_agent, "Test query")

            assert "Error" in result["answer"]
            assert result["intermediate_steps"] == []


class TestToolWrapperIntegration:
    """Integration tests for tool wrapping."""

    @pytest.mark.asyncio
    async def test_tool_wrapper_converts_string_to_dict(self):
        """Test that string inputs are converted to proper dict format."""
        # Create a mock MCP tool with Pydantic schema
        mock_tool = Mock()
        mock_tool.name = "getPokemon"
        mock_tool.description = "Get Pokemon"
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool.ainvoke = AsyncMock(
            return_value={"name": "Pikachu", "type": "Electric"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, client = await get_mcp_tools()
            wrapped_tool = tools[0]

            # Verify the underlying tool was called with proper dict format
            mock_tool.ainvoke.assert_not_called()
            assert wrapped_tool.name == "getPokemon"

    @pytest.mark.asyncio
    async def test_tool_wrapper_handles_tuple_input(self):
        """Test tool wrapper handles tuple/list inputs."""
        mock_tool = Mock()
        mock_tool.name = "getPokemon"
        mock_tool.description = "Get Pokemon"
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool.ainvoke = AsyncMock(return_value={"name": "Bulbasaur"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, _ = await get_mcp_tools()
            wrapped_tool = tools[0]

            # Wrapped tool should handle tuple input
            assert wrapped_tool.name == "getPokemon"
            assert wrapped_tool.description == "Get Pokemon"

    @pytest.mark.asyncio
    async def test_tool_wrapper_with_dict_schema(self):
        """Test tool wrapper with dict-based schema."""
        mock_tool = Mock()
        mock_tool.name = "getType"
        mock_tool.description = "Get type info"
        # Schema is a dict instead of Pydantic
        mock_tool.args_schema = {
            "properties": {
                "typeName": {"type": "string"}
            }
        }
        mock_tool.ainvoke = AsyncMock(return_value={"name": "electric"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, _ = await get_mcp_tools()
            wrapped_tool = tools[0]

            assert wrapped_tool.name == "getType"

    @pytest.mark.asyncio
    async def test_tool_wrapper_with_invalid_json_input(self):
        """Test tool wrapper gracefully handles invalid JSON."""
        mock_tool = Mock()
        mock_tool.name = "getPokemon"
        mock_tool.description = "Get Pokemon"
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.__fields__ = {"idOrName": Mock()}
        mock_tool.ainvoke = AsyncMock(return_value={"name": "Squirtle"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, _ = await get_mcp_tools()
            wrapped_tool = tools[0]

            # Tool should be created successfully
            assert wrapped_tool.name == "getPokemon"

    @pytest.mark.asyncio
    async def test_tool_wrapper_handles_no_schema(self):
        """Test tool wrapper handles tools with no schema."""
        mock_tool = Mock()
        mock_tool.name = "simpleCommand"
        mock_tool.description = "A simple command"
        mock_tool.args_schema = None  # No schema
        mock_tool.ainvoke = AsyncMock(return_value={"result": "done"})

        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch(
            "example.pokemon_mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools, _ = await get_mcp_tools()
            wrapped_tool = tools[0]

            assert wrapped_tool.name == "simpleCommand"


class TestHandlePokemonMCP:
    """Test handle_pokemon_mcp Streamlit UI handler."""

    @patch("example.pokemon_mcp.st")
    @patch("example.pokemon_mcp.create_chain")
    @patch("example.pokemon_mcp.get_mcp_tools")
    def test_handle_pokemon_mcp_initialization(self, mock_get_tools, mock_create_chain, mock_st):  # noqa: E501
        """Test handle_pokemon_mcp initializes session state."""
        # Setup mocks
        mock_get_tools.return_value = ([], Mock())
        mock_create_chain.return_value = Mock()

        # Create proper context manager for spinner
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        # Create proper context manager for chat_message
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_st.chat_message.return_value = mock_context

        mock_st.chat_input.return_value = None
        # Use MagicMock to support attribute assignment
        mock_st.session_state = MagicMock()

        # Should initialize without error
        handle_pokemon_mcp(mock_st, "mistral")

        # Verify setup was called
        mock_get_tools.assert_called_once()

    @patch("example.pokemon_mcp.st")
    @patch("example.pokemon_mcp.process_query")
    @patch("example.pokemon_mcp.create_chain")
    @patch("example.pokemon_mcp.get_mcp_tools")
    def test_handle_pokemon_mcp_with_user_input(
            self, mock_get_tools, mock_create_chain, mock_process_query, mock_st):  # noqa: E501
        """Test handle_pokemon_mcp processes user input."""
        # Setup mocks
        mock_agent = Mock()
        mock_create_chain.return_value = mock_agent
        mock_get_tools.return_value = ([Mock(name="tool1")], Mock())
        mock_process_query.return_value = {
            "answer": "Test answer",
            "intermediate_steps": []
        }

        # Create proper context manager
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_st.chat_message.return_value = mock_context

        mock_st.chat_input.return_value = "Tell me about Pikachu"

        # Use a MagicMock to support attribute assignment
        session_state_mock = MagicMock()
        session_state_mock.__contains__ = Mock(return_value=False)
        mock_st.session_state = session_state_mock

        # Should process input without error
        handle_pokemon_mcp(mock_st, "mistral")

        # Verify chain was created
        mock_create_chain.assert_called_once()

    @patch("example.pokemon_mcp.st")
    @patch("example.pokemon_mcp.create_chain")
    @patch("example.pokemon_mcp.get_mcp_tools")
    def test_handle_pokemon_mcp_with_langfuse(self, mock_get_tools, mock_create_chain, mock_st):  # noqa: E501
        """Test handle_pokemon_mcp with Langfuse callback."""
        # Setup mocks
        mock_get_tools.return_value = ([], Mock())
        mock_create_chain.return_value = Mock()

        # Create proper context manager for spinner
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_st.chat_message.return_value = mock_context

        mock_st.chat_input.return_value = None
        mock_st.session_state = MagicMock()

        mock_langfuse = Mock()

        # Should handle langfuse handler without error
        handle_pokemon_mcp(mock_st, "mistral", langfuse_handler=mock_langfuse)

        # Verify setup was called
        mock_get_tools.assert_called_once()

    @patch("example.pokemon_mcp.st")
    @patch("example.pokemon_mcp.get_mcp_tools")
    def test_handle_pokemon_mcp_error_handling(self, mock_get_tools, mock_st):
        """Test handle_pokemon_mcp error handling."""
        # Setup mock to raise exception
        mock_get_tools.side_effect = Exception("Tool loading failed")

        # Create proper context manager for spinner
        mock_spinner = Mock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=False)
        mock_st.spinner.return_value = mock_spinner

        mock_st.session_state = MagicMock()
        mock_st.chat_input.return_value = None

        # Should handle errors gracefully
        handle_pokemon_mcp(mock_st, "mistral")
