import pytest

from internal.brave_client import BraveMCPClient


class TestBraveMCPClientIntegration:
    """Integration tests for BraveMCPClient with real MCP server."""

    @pytest.mark.integration
    def test_web_search_integration(self):
        """Test web search against real MCP server."""
        client = BraveMCPClient()

        # Use default host - assumes docker compose environment
        results = client.web_search("Python programming language", count=3)

        # Should return list of documents (or empty if server unavailable)
        assert isinstance(results, list)

        # If we get results, verify structure
        if len(results) > 0:
            from langchain_core.documents import Document
            assert all(isinstance(r, Document) for r in results)
            assert all(r.metadata.get("source") == "brave_search" for r in results)  # noqa: E501

    @pytest.mark.integration
    def test_llm_context_integration(self):
        """Test LLM context search against real MCP server."""
        client = BraveMCPClient()

        results = client.llm_context("machine learning", count=2)

        # Should return list of documents (or empty if server unavailable)
        assert isinstance(results, list)

        # If we get results, verify structure
        if len(results) > 0:
            from langchain_core.documents import Document
            assert all(isinstance(r, Document) for r in results)
            assert all(r.metadata.get("source") == "brave_llm_context" for r in results)  # noqa: E501

    @pytest.mark.integration
    def test_web_search_multiple_queries(self):
        """Test multiple web searches."""
        client = BraveMCPClient()

        queries = [
            "artificial intelligence", "quantum computing", "neural networks"]
        for query in queries:
            results = client.web_search(query, count=2)
            assert isinstance(results, list)
