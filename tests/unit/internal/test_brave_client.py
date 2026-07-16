"""Unit tests for brave_client module - error handling and SSE parsing."""

import json
from unittest.mock import Mock, patch

from internal.brave_client import BraveMCPClient, BraveSearchTool


class TestBraveMCPClientInit:
    """Tests for BraveMCPClient initialization."""

    def test_init_default_host(self):
        """Test initialization with default host."""
        client = BraveMCPClient()
        assert client.host == "http://brave-search-mcp:8080/mcp"
        assert client._request_id == 0

    def test_init_custom_host(self):
        """Test initialization with custom host."""
        custom_host = "http://custom:9000/mcp"
        client = BraveMCPClient(host=custom_host)
        assert client.host == custom_host


class TestBraveMCPClientRequestID:
    """Tests for request ID generation."""

    def test_get_request_id_increments(self):
        """Test that request IDs increment."""
        client = BraveMCPClient()
        assert client._get_request_id() == 1
        assert client._get_request_id() == 2
        assert client._get_request_id() == 3


class TestBraveMCPClientSSEParsing:
    """Tests for SSE response parsing."""

    def test_parse_sse_response_valid(self):
        """Test parsing valid SSE response."""
        client = BraveMCPClient()
        sse_text = 'data: {"jsonrpc": "2.0", "result": {"key": "value"}}\n'

        result = client._parse_sse_response(sse_text)

        assert result == {"key": "value"}

    def test_parse_sse_response_with_error(self):
        """Test parsing SSE response with error."""
        client = BraveMCPClient()
        sse_text = 'data: {"jsonrpc": "2.0", "error": {"code": -1, "message": "Error"}}\n'  # noqa: E501

        result = client._parse_sse_response(sse_text)

        assert result is None

    def test_parse_sse_response_no_data_line(self):
        """Test parsing SSE response without data line."""
        client = BraveMCPClient()
        sse_text = 'event: update\nid: 123\n'

        result = client._parse_sse_response(sse_text)

        assert result is None

    def test_parse_sse_response_invalid_json(self):
        """Test parsing SSE response with invalid JSON."""
        client = BraveMCPClient()
        sse_text = 'data: {invalid json}\n'

        result = client._parse_sse_response(sse_text)

        assert result is None

    def test_parse_sse_response_empty(self):
        """Test parsing empty SSE response."""
        client = BraveMCPClient()
        sse_text = ''

        result = client._parse_sse_response(sse_text)

        assert result is None


class TestBraveMCPClientErrorHandling:
    """Tests for error handling in tool calls."""

    @patch('internal.brave_client.httpx.Client.post')
    def test_call_tool_http_error(self, mock_post):
        """Test tool call with HTTP error."""
        client = BraveMCPClient()

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response

        result = client._call_tool("test_tool", {"param": "value"})

        assert result is None

    @patch('internal.brave_client.httpx.Client.post')
    def test_call_tool_network_error(self, mock_post):
        """Test tool call with network error."""
        client = BraveMCPClient()
        mock_post.side_effect = Exception("Connection error")

        result = client._call_tool("test_tool", {"param": "value"})

        assert result is None

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_null_result(self, mock_post):
        """Test web search when result is None."""
        client = BraveMCPClient()

        sse_response = 'data: {"jsonrpc": "2.0"}\n'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.web_search("test query")

        assert results == []

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_invalid_result_type(self, mock_post):
        """Test web search when result has unexpected type."""
        client = BraveMCPClient()

        search_result = "not a dict"
        sse_response = f'data: {json.dumps({"jsonrpc": "2.0", "result": search_result})}\n'  # noqa: E501
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.web_search("test query")

        assert results == []


class TestBraveMCPClientWebSearch:
    """Tests for web_search functionality."""

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_success_with_results(self, mock_post):
        """Test successful web search with results."""
        client = BraveMCPClient()

        # Mock successful SSE response with search results
        search_results = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "Python Programming",
                        "url": "https://python.org",
                        "description": "Official Python website"
                    })
                }
            ]
        }

        sse_response = f'data: {json.dumps({"jsonrpc": "2.0", "result": search_results})}\n'  # noqa: E501
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.web_search("Python")

        assert len(results) == 1
        assert results[0].page_content == "Official Python website"
        assert results[0].metadata["title"] == "Python Programming"
        assert results[0].metadata["url"] == "https://python.org"
        assert results[0].metadata["source"] == "brave_search"

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_multiple_results(self, mock_post):
        """Test web search with multiple results."""
        client = BraveMCPClient()

        search_results = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "Result 1",
                        "url": "https://example1.com",
                        "description": "First result"
                    })
                },
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "Result 2",
                        "url": "https://example2.com",
                        "description": "Second result"
                    })
                }
            ]
        }

        sse_response = f'data: {json.dumps({"jsonrpc": "2.0", "result": search_results})}\n'  # noqa: E501
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.web_search("test query", count=5)

        assert len(results) == 2
        assert results[0].page_content == "First result"
        assert results[1].page_content == "Second result"

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_empty_description(self, mock_post):
        """Test web search skips items with empty descriptions."""
        client = BraveMCPClient()

        search_results = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "Result with desc",
                        "url": "https://example.com",
                        "description": "Has description"
                    })
                },
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "Result without desc",
                        "url": "https://example2.com",
                        "description": ""
                    })
                }
            ]
        }

        sse_response = f'data: {json.dumps({"jsonrpc": "2.0", "result": search_results})}\n'  # noqa: E501
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.web_search("test")

        # Should only include result with description
        assert len(results) == 1
        assert results[0].metadata["title"] == "Result with desc"

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_non_text_items(self, mock_post):
        """Test web search skips non-text content items."""
        client = BraveMCPClient()

        search_results = {
            "content": [
                {
                    "type": "image",
                    "text": "image data"
                },
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "Valid result",
                        "url": "https://example.com",
                        "description": "Valid description"
                    })
                }
            ]
        }

        sse_response = f'data: {json.dumps({"jsonrpc": "2.0", "result": search_results})}\n'  # noqa: E501
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.web_search("test")

        assert len(results) == 1
        assert results[0].metadata["title"] == "Valid result"

    @patch('internal.brave_client.httpx.Client.post')
    def test_web_search_with_parameters(self, mock_post):
        """Test web search passes parameters correctly."""
        client = BraveMCPClient()

        sse_response = 'data: {"jsonrpc": "2.0", "result": {"content": []}}\n'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        client.web_search("test query", count=15, country="UK")

        # Verify the call was made with correct parameters
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['params']['arguments']['query'] == "test query"
        assert payload['params']['arguments']['count'] == 15
        assert payload['params']['arguments']['country'] == "UK"


class TestBraveMCPClientLLMContext:
    """Tests for llm_context functionality."""

    @patch('internal.brave_client.httpx.Client.post')
    def test_llm_context_success(self, mock_post):
        """Test successful LLM context search."""
        client = BraveMCPClient()

        search_results = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "title": "AI Article",
                        "url": "https://ai.example.com",
                        "description": "Article about AI"
                    })
                }
            ]
        }

        sse_response = f'data: {json.dumps({"jsonrpc": "2.0", "result": search_results})}\n'  # noqa: E501
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        results = client.llm_context("artificial intelligence")

        assert len(results) == 1
        assert results[0].metadata["source"] == "brave_llm_context"

    @patch('internal.brave_client.httpx.Client.post')
    def test_llm_context_with_count(self, mock_post):
        """Test LLM context with custom count."""
        client = BraveMCPClient()

        sse_response = 'data: {"jsonrpc": "2.0", "result": {"content": []}}\n'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sse_response
        mock_post.return_value = mock_response

        client.llm_context("test", count=8)

        # Verify parameters - llm_context uses maximum_number_of_urls
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['params']['arguments']['maximum_number_of_urls'] == 8


class TestBraveSearchTool:
    """Tests for BraveSearchTool."""

    def test_init_default(self):
        """Test BraveSearchTool initialization."""
        tool = BraveSearchTool()
        assert tool.client is not None

    def test_init_custom_host(self):
        """Test BraveSearchTool with custom host."""
        custom_host = "http://custom:9000/mcp"
        tool = BraveSearchTool(host=custom_host)
        assert tool.client.host == custom_host
