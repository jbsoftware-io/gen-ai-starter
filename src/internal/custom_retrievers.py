# flake8: noqa: E501
"""Custom retrievers to replace langchain_community dependencies."""

from typing import List, Optional

import arxiv
import wikipedia
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool


class WikipediaAPIWrapper:
    """Wrapper around Wikipedia API for querying and retrieving page content."""

    def run(self, query: str) -> str:
        """Run a Wikipedia search and return the summary of the first result.

        Args:
            query: The search query

        Returns:
            The page summary or error message
        """
        try:
            results = wikipedia.search(query, results=1)
            if not results:
                return "No Wikipedia results found."

            page = wikipedia.page(results[0])
            return page.summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation page - could not determine which page. Options: {e.options[:3]}"
        except wikipedia.exceptions.PageError:
            return "Page not found."
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

    def get_page_content(self, page_title: str) -> Optional[str]:
        """Get the full content of a Wikipedia page.

        Args:
            page_title: The title of the Wikipedia page

        Returns:
            The page content or None if not found
        """
        try:
            page = wikipedia.page(page_title)
            return page.content
        except wikipedia.exceptions.PageError:
            return None
        except wikipedia.exceptions.DisambiguationError:
            return None
        except Exception:
            return None


class WikipediaRetriever(BaseRetriever):
    """Retriever that fetches documents from Wikipedia.

    This is a custom implementation to replace langchain_community.retrievers.WikipediaRetriever
    after the deprecation of langchain_community.
    """

    top_k_results: int = 4
    wiki_wrapper: WikipediaAPIWrapper = None

    def __init__(self, top_k_results: int = 4, **kwargs):
        """Initialize the Wikipedia retriever.

        Args:
            top_k_results: Number of top results to retrieve (default: 4)
        """
        super().__init__(**kwargs)
        self.top_k_results = top_k_results
        self.wiki_wrapper = WikipediaAPIWrapper()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents from Wikipedia.

        Args:
            query: The search query

        Returns:
            List of Document objects
        """
        try:
            search_results = wikipedia.search(query, results=self.top_k_results)
        except Exception:
            return []

        documents = []
        for result in search_results:
            try:
                page = wikipedia.page(result)
                doc = Document(
                    page_content=page.content,
                    metadata={"source": page.url, "title": page.title}
                )
                documents.append(doc)
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue
            except Exception:
                continue

        return documents


class ArxivRetriever(BaseRetriever):
    """Retriever that fetches papers from Arxiv.

    This is a custom implementation to replace langchain_community.retrievers.ArxivRetriever
    after the deprecation of langchain_community.
    """

    load_max_docs: int = 3
    get_full_documents: bool = False

    def __init__(
        self,
        load_max_docs: int = 3,
        get_full_documents: bool = False,
        **kwargs
    ):
        """Initialize the Arxiv retriever.

        Args:
            load_max_docs: Maximum number of documents to load (default: 3)
            get_full_documents: Whether to fetch full abstracts (default: False)
        """
        super().__init__(**kwargs)
        self.load_max_docs = load_max_docs
        self.get_full_documents = get_full_documents

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve papers from Arxiv.

        Args:
            query: The search query

        Returns:
            List of Document objects with paper information
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=self.load_max_docs,
                sort_by=arxiv.SortCriterion.Relevance
            )

            documents = []
            for result in client.results(search):
                content = result.summary if self.get_full_documents else result.summary[:500]

                doc = Document(
                    page_content=content,
                    metadata={
                        "title": result.title,
                        "authors": ", ".join([author.name for author in result.authors]),
                        "published": str(result.published),
                        "arxiv_url": result.arxiv_url,
                        "pdf_url": result.pdf_url,
                    }
                )
                documents.append(doc)

            return documents
        except Exception:
            return []


@tool
def wikipedia_query_run(query: str) -> str:
    """Search Wikipedia and return the summary of the first matching article.

    Args:
        query: The search query for Wikipedia

    Returns:
        The summary of the first matching Wikipedia article
    """
    wrapper = WikipediaAPIWrapper()
    return wrapper.run(query)


@tool
def arxiv_query_run(query: str) -> str:
    """Search Arxiv and return summaries of matching academic papers.

    Args:
        query: The search query for Arxiv

    Returns:
        Formatted summaries of matching Arxiv papers
    """
    retriever = ArxivRetriever(load_max_docs=3, get_full_documents=True)
    docs = retriever.invoke(query)

    if not docs:
        return "No Arxiv papers found for that query."

    formatted = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("authors", "Unknown")
        arxiv_url = doc.metadata.get("arxiv_url", "Unknown")
        content = doc.page_content[:300]
        formatted.append(
            f"**{title}**\nBy: {authors}\nURL: {arxiv_url}\n{content}...\n"
        )

    return "\n".join(formatted)
