"""
Integration tests for PokéAPI MCP example with Deep Agents.

Single end-to-end test to verify the full pipeline works:
loading MCP tools, creating a Deep Agents chain, and processing queries.
"""

import asyncio
import os

import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")


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

        # Verify answer contains relevant information
        answer_lower = answer.lower()
        assert ("pokemon" in answer_lower or "pikachu" in answer_lower or
                "electric" in answer_lower), \
            f"Answer doesn't contain expected Pokemon info: {answer}"
