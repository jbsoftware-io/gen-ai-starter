# flake8: noqa: E501
"""JSON-RPC client for Brave Search MCP server integration via HTTP."""

import json
from typing import List, Optional

import httpx
from langchain_core.documents import Document
from langchain_core.tools import tool

from .logger import logger


class BraveMCPClient:
    """Client for calling Brave Search MCP server via MCP HTTP transport (stateless mode)."""

    def __init__(self, host: str = "http://brave-search-mcp:8080/mcp"):
        """Initialize Brave MCP client.

        Args:
            host: The host URL of the Brave MCP server (default: http://brave-search-mcp:8080/mcp)
        """
        self.host = host
        self.client = httpx.Client(timeout=30.0)
        self._request_id = 0

    def _get_request_id(self) -> int:
        """Get next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    def _call_tool(
        self,
        tool_name: str,
        params: dict,
    ) -> Optional[dict]:
        """Call a tool via MCP HTTP transport using JSON-RPC protocol.

        Args:
            tool_name: Name of the tool to call (e.g., "brave_web_search")
            params: Parameters for the tool

        Returns:
            The result from the tool, or None on error
        """
        try:
            request_id = self._get_request_id()

            # MCP HTTP uses JSON-RPC 2.0 with tools/call method
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params,
                }
            }

            logger.info(f"Calling MCP tool {tool_name} with params: {params}")
            logger.info(f"JSON-RPC payload: {json.dumps(payload)}")

            # MCP HTTP returns Server-Sent Events (SSE)
            response = self.client.post(
                self.host,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )

            logger.info(f"MCP response status: {response.status_code}")
            logger.info(f"MCP response text (first 1000 chars): {response.text[:1000]}")

            # Log error responses before raising
            if response.status_code >= 400:
                logger.error(f"MCP error response body: {response.text}")

            response.raise_for_status()

            # Parse Server-Sent Events format
            result = self._parse_sse_response(response.text)
            logger.info(f"Parsed MCP result: {result}")

            if not result:
                logger.warning("No result extracted from SSE response")
                return None

            return result

        except Exception as e:
            logger.error(f"MCP call error for {tool_name}: {str(e)}", exc_info=True)
            return None

    def _parse_sse_response(self, sse_text: str) -> Optional[dict]:
        """Parse Server-Sent Events response from MCP HTTP transport.

        Args:
            sse_text: Raw SSE response text

        Returns:
            Parsed result or None on error
        """
        try:
            lines = sse_text.strip().split('\n')
            for line in lines:
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove "data: " prefix
                    data = json.loads(data_str)

                    # Check for JSON-RPC error
                    if "error" in data:
                        logger.error(f"MCP error in response: {data['error']}")
                        return None

                    # Extract result
                    if "result" in data:
                        logger.info(f"Extracted result from SSE: {type(data['result'])}")
                        return data["result"]

            logger.warning(f"No 'data:' lines found in SSE response: {sse_text[:200]}")
            return None

        except Exception as e:
            logger.error(f"Error parsing SSE response: {str(e)}", exc_info=True)
            return None

    def web_search(
        self,
        query: str,
        count: int = 10,
        country: str = "US",
    ) -> List[Document]:
        """Perform a web search using Brave MCP server (stateless mode).

        Args:
            query: The search query
            count: Number of results to return (default: 10)
            country: Country code (default: "US")

        Returns:
            List of Document objects with search results
        """
        result = self._call_tool(
            "brave_web_search",
            {
                "query": query,
                "count": count,
                "country": country,
            }
        )

        logger.info(f"Raw Brave result type: {type(result)}")
        logger.info(f"Raw Brave result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

        documents = []
        if not result:
            logger.warning("No result from Brave search")
            return documents

        # Parse results - MCP returns {content: [...], isError: bool}
        if isinstance(result, dict):
            content_items = result.get("content", [])
            logger.info(f"Found content array with {len(content_items)} items")
        else:
            logger.error(f"Unexpected result type: {type(result)}")
            return documents

        if not isinstance(content_items, list):
            logger.warning(f"Content is not a list: {type(content_items)}")
            return documents

        for i, item in enumerate(content_items):
            try:
                logger.info(f"Processing content item {i}: {type(item)}")

                # Each item has type and text - text is a JSON string
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content = item.get("text", "")
                    logger.info(f"Text content (first 200 chars): {text_content[:200]}")

                    # Parse the JSON string inside text
                    try:
                        parsed = json.loads(text_content)
                        logger.info(f"Parsed item {i}: url={parsed.get('url')}, title={parsed.get('title')}")

                        description = parsed.get("description", "")
                        title = parsed.get("title", "")
                        url = parsed.get("url", "")

                        if description and description.strip():
                            doc = Document(
                                page_content=description,
                                metadata={
                                    "title": title,
                                    "url": url,
                                    "source": "brave_search",
                                }
                            )
                            documents.append(doc)
                            logger.info(f"✓ Created document {len(documents)}: {title}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON in item {i}: {str(e)}")
                        continue
                else:
                    logger.warning(f"Item {i} not in expected format: {type(item)}")

            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Created {len(documents)} documents from Brave search")
        return documents

    def llm_context(
        self,
        query: str,
        count: int = 5,
    ) -> List[Document]:
        """Get LLM context optimized search results from Brave (stateless mode).

        Args:
            query: The search query
            count: Number of results to consider

        Returns:
            List of Document objects with LLM-optimized content
        """
        result = self._call_tool(
            "brave_llm_context",
            {
                "query": query,
                "maximum_number_of_urls": count,
            }
        )

        logger.info(f"Raw Brave llm_context result type: {type(result)}")

        documents = []
        if not result:
            logger.warning("No result from Brave llm_context")
            return documents

        # Parse results - MCP returns {content: [...], isError: bool}
        if isinstance(result, dict):
            content_items = result.get("content", [])
            logger.info(f"Found content array with {len(content_items)} items")
        else:
            logger.error(f"Unexpected result type: {type(result)}")
            return documents

        if not isinstance(content_items, list):
            logger.warning(f"Content is not a list: {type(content_items)}")
            return documents

        for i, item in enumerate(content_items):
            try:
                logger.info(f"Processing content item {i}: {type(item)}")

                # Each item has type and text - text is a JSON string
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content = item.get("text", "")

                    # Parse the JSON string inside text
                    try:
                        parsed = json.loads(text_content)
                        logger.info(f"Parsed item {i}: url={parsed.get('url')}, title={parsed.get('title')}")

                        # For llm_context, prefer 'content' field, fall back to description
                        content = parsed.get("content", "")
                        if not content:
                            content = parsed.get("description", "")

                        title = parsed.get("title", "")
                        url = parsed.get("url", "")

                        if content and content.strip():
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "title": title,
                                    "url": url,
                                    "source": "brave_llm_context",
                                }
                            )
                            documents.append(doc)
                            logger.info(f"✓ Created document {len(documents)}: {title}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON in item {i}: {str(e)}")
                        continue
                else:
                    logger.warning(f"Item {i} not in expected format")

            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Created {len(documents)} documents from Brave llm_context")
        return documents


class BraveSearchTool:
    """Tool for using Brave Search in LangChain agents."""

    def __init__(self, host: str = "http://brave-search-mcp:8080/mcp"):
        """Initialize Brave Search tool.

        Args:
            host: The host URL of the Brave MCP server
        """
        self.client = BraveMCPClient(host)

    def search(self, query: str, count: int = 10) -> List[Document]:
        """Search using Brave and return documents.

        Args:
            query: The search query
            count: Number of results

        Returns:
            List of Document objects
        """
        return self.client.web_search(query, count=count)

    def get_llm_context(self, query: str, count: int = 5) -> List[Document]:
        """Get LLM-optimized search results.

        Args:
            query: The search query
            count: Number of results

        Returns:
            List of Document objects optimized for LLM context
        """
        return self.client.llm_context(query, count=count)


@tool
def brave_search(query: str) -> str:
    """Search the web using Brave Search MCP server.

    Args:
        query: The search query

    Returns:
        Formatted search results
    """
    client = BraveMCPClient()
    results = client.web_search(query, count=5)

    if not results:
        return "No search results found."

    formatted = []
    for doc in results:
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        content = doc.page_content[:200]
        formatted.append(f"**{title}**\n{url}\n{content}...\n")

    return "\n".join(formatted)
