"""
Integration tests for PokéAPI MCP example with Deep Agents.

Single end-to-end test to verify the full pipeline works:
loading MCP tools, creating a Deep Agents chain, and processing queries.
"""

import asyncio
import os
import time

import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")


def check_service_available(url: str, max_retries: int = 3) -> bool:
    """Check if a service is available with retries and exponential backoff.

    Args:
        url: The service URL to check
        max_retries: Maximum number of retry attempts

    Returns:
        True if service is available, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            continue
    return False


class TestPokemonMCPEndToEnd:
    """Integration test for the Pokemon MCP example with Deep Agents."""

    @pytest.mark.integration
    def test_pokemon_mcp_full_pipeline(self):
        """Test end-to-end Pokemon MCP pipeline: tools → agent → query."""
        from example.pokemon_mcp import (create_chain, process_query,
                                         get_mcp_tools)

        # Skip if Ollama not available
        if not OLLAMA_HOST:
            pytest.skip("OLLAMA_HOST not set")

        # Check if any models are available
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.fail("Ollama service not available")

            models = response.json().get("models", [])
            if not models:
                pytest.fail("No Ollama models found")

            # Prefer llama model (supports tool calling)
            model_names = [m["name"] for m in models]
            llama_models = [m for m in model_names if "llama" in m.lower()]

            if llama_models:
                model_name = llama_models[0]
            else:
                # Skip if no llama model available
                msg = (f"No llama model found. Available: "
                       f"{', '.join(model_names)}")
                pytest.skip(msg)

        except requests.exceptions.RequestException:
            pytest.fail("Ollama service not available")

        # Check if MCP Pokemon server is available (with retries for transient issues)  # noqa: E501
        mcp_host = os.getenv("POKEMON_MCP_SERVER_HOST", "http://mcp-pokemon:3001")  # noqa: E501
        if not check_service_available(f"{mcp_host}/api/tools", max_retries=3):
            pytest.skip("MCP Pokemon server not available after retries")

        # Step 1: Load MCP tools from Pokemon server
        tools, client = asyncio.run(get_mcp_tools())
        assert len(tools) > 0, "No tools loaded from MCP server"
        tool_names = [tool.name for tool in tools]

        # Verify expected tools are available
        expected_tools = ["getPokemon", "getPokemonSpecies", "getType",
                          "getAbility", "getMove"]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, \
                f"Expected tool '{expected_tool}' not found"

        # Step 2: Create Deep Agent with loaded tools
        agent = create_chain(model_name, tools)
        assert agent is not None
        assert hasattr(agent, 'invoke')

        # Step 3: Process a query end-to-end
        result = process_query(agent, "Tell me about Pikachu")

        # Verify response structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "intermediate_steps" in result

        # Verify answer quality
        answer = result["answer"]
        assert isinstance(answer, str)
        assert len(answer) > 0

        # Verify we got a valid response (not an error)
        assert not answer.lower().startswith("error"), \
            f"Got error response: {answer}"

    @pytest.mark.integration
    def test_pokemon_mcp_compare_prompt(self):
        """Test Pokemon MCP dynamic comparison prompt end-to-end"""
        from example.pokemon_mcp import (create_chain, process_query,
                                         get_mcp_tools)

        # Skip if Ollama not available
        if not OLLAMA_HOST:
            pytest.skip("OLLAMA_HOST not set")

        # Check if any models are available
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.fail("Ollama service not available")

            models = response.json().get("models", [])
            if not models:
                pytest.fail("No Ollama models found")

            # Prefer llama model (supports tool calling)
            model_names = [m["name"] for m in models]
            llama_models = [m for m in model_names if "llama" in m.lower()]

            if llama_models:
                model_name = llama_models[0]
            else:
                # Skip if no llama model available
                msg = (f"No llama model found. Available: "
                       f"{', '.join(model_names)}")
                pytest.skip(msg)

        except requests.exceptions.RequestException:
            pytest.fail("Ollama service not available")

        # Check if MCP Pokemon server is available (with retries for transient issues)  # noqa: E501
        mcp_host = os.getenv("POKEMON_MCP_SERVER_HOST", "http://mcp-pokemon:3001")  # noqa: E501
        if not check_service_available(f"{mcp_host}/api/tools", max_retries=3):
            pytest.skip("MCP Pokemon server not available after retries")

        # Step 1: Load MCP tools from Pokemon server
        tools, _ = asyncio.run(get_mcp_tools())

        # Step 2: Create Deep Agent with loaded tools
        agent = create_chain(model_name, tools)
        assert agent is not None
        assert hasattr(agent, 'invoke')

        # Step 3: Process a comparison query end-to-end
        result = process_query(agent, "Compare Pikachu and Bulbasaur")

        # Verify response structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "intermediate_steps" in result

        # Verify answer quality
        answer = result["answer"]
        assert isinstance(answer, str)
        assert len(answer) > 0

        # Verify we got a valid response (not an error)
        assert not answer.lower().startswith("error"), \
            f"Got error response: {answer}"
