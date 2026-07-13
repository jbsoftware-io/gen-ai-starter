"""MCP client wrapper for Brave Search API integration."""

import json
from typing import List, Optional

import httpx
from langchain_core.documents import Document
from langchain_core.tools import tool


class BraveMCPClient:
    """Client for calling Brave Search MCP server via HTTP."""

    def __init__(self, host: str = "http://brave-search-mcp:8080"):
        """Initialize Brave MCP client.
        
        Args:
            host: The host URL of the Brave MCP server (default: http://brave-search-mcp:8080)
        """
        self.host = host
        self.client = httpx.Client(timeout=30.0)

    def web_search(
        self,
        query: str,
        count: int = 10,
        country: str = "US",
    ) -> List[Document]:
        """Perform a web search using Brave MCP server.
        
        Args:
            query: The search query
            count: Number of results to return (default: 10)
            country: Country code (default: "US")
            
        Returns:
            List of Document objects with search results
        """
        try:
            # Call the MCP server's brave_web_search tool
            payload = {
                "query": query,
                "count": count,
                "country": country,
            }
            
            response = self.client.post(
                f"{self.host}/tool/brave_web_search",
                json=payload,
            )
            response.raise_for_status()
            
            results = response.json()
            
            documents = []
            if isinstance(results, dict) and "results" in results:
                for result in results.get("results", []):
                    doc = Document(
                        page_content=result.get("description", result.get("title", "")),
                        metadata={
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "source": "brave_search",
                        }
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            return []

    def llm_context(
        self,
        query: str,
        count: int = 5,
    ) -> List[Document]:
        """Get LLM context optimized search results from Brave.
        
        Args:
            query: The search query
            count: Number of results to consider
            
        Returns:
            List of Document objects with LLM-optimized content
        """
        try:
            payload = {
                "query": query,
                "maximum_number_of_urls": count,
            }
            
            response = self.client.post(
                f"{self.host}/tool/brave_llm_context",
                json=payload,
            )
            response.raise_for_status()
            
            results = response.json()
            
            documents = []
            if isinstance(results, dict) and "results" in results:
                for result in results.get("results", []):
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "source": "brave_llm_context",
                        }
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            return []


class BraveSearchTool:
    """Tool for using Brave Search in LangChain agents."""

    def __init__(self, host: str = "http://brave-search-mcp:8080"):
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
