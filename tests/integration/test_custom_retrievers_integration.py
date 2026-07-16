"""Integration tests for custom_retrievers module - testing real API calls."""

import pytest

from internal.custom_retrievers import (
    ArxivRetriever,
    WikipediaRetriever,
    WikipediaAPIWrapper,
)


class TestWikipediaRetrieverIntegration:
    """Integration tests for WikipediaRetriever with real Wikipedia API."""

    @pytest.mark.integration
    def test_wikipedia_search_python(self):
        """Test retrieving Wikipedia documents for 'Python'."""
        retriever = WikipediaRetriever(top_k_results=2)
        docs = retriever._get_relevant_documents("Python programming language")

        # Should find at least one result
        assert len(docs) > 0

        # Should have proper Document structure
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)

        # Should have metadata
        for doc in docs:
            assert 'source' in doc.metadata
            assert 'title' in doc.metadata

    @pytest.mark.integration
    def test_wikipedia_search_specific_article(self):
        """Test retrieving specific Wikipedia article."""
        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents("Machine Learning")

        assert len(docs) > 0
        # Content should be reasonably long
        assert len(docs[0].page_content) > 100

    @pytest.mark.integration
    def test_wikipedia_search_nonexistent(self):
        """Test search for nonexistent topic gracefully returns empty."""
        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents("xyzabc123nonexistent")

        # Should handle gracefully with empty results
        assert isinstance(docs, list)


class TestWikipediaAPIWrapperIntegration:
    """Integration tests for WikipediaAPIWrapper with real API."""

    @pytest.mark.integration
    def test_wrapper_run_success(self):
        """Test wrapper.run() with real Wikipedia API."""
        wrapper = WikipediaAPIWrapper()
        result = wrapper.run("Artificial Intelligence")

        # Should return a non-empty summary
        assert isinstance(result, str)
        assert len(result) > 50

    @pytest.mark.integration
    def test_wrapper_run_ambiguous(self):
        """Test wrapper handles disambiguation gracefully."""
        wrapper = WikipediaAPIWrapper()
        # "Python" is ambiguous (snake or language)
        result = wrapper.run("Python")

        # Should still return something (either a page or disambiguation message)  # noqa: E501
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.integration
    def test_wrapper_get_page_content(self):
        """Test getting full page content."""
        wrapper = WikipediaAPIWrapper()
        # Use exact page title to avoid ambiguity
        result = wrapper.get_page_content("Isaac Newton")

        # Should return substantial content
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 200


class TestArxivRetrieverIntegration:
    """Integration tests for ArxivRetriever with real Arxiv API."""

    @pytest.mark.integration
    def test_arxiv_search_machine_learning(self):
        """Test retrieving papers from Arxiv."""
        retriever = ArxivRetriever(load_max_docs=2)
        docs = retriever._get_relevant_documents("machine learning")

        # Should find papers
        assert len(docs) > 0

        # Should have Document structure
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)

    @pytest.mark.integration
    def test_arxiv_search_with_metadata(self):
        """Test that Arxiv results include proper metadata."""
        retriever = ArxivRetriever(load_max_docs=1)
        docs = retriever._get_relevant_documents("neural networks")

        assert len(docs) > 0
        doc = docs[0]

        # Check metadata fields
        assert 'title' in doc.metadata
        assert 'authors' in doc.metadata
        assert 'published' in doc.metadata
        assert 'arxiv_url' in doc.metadata

    @pytest.mark.integration
    def test_arxiv_search_full_documents(self):
        """Test retrieving full abstracts from Arxiv."""
        retriever = ArxivRetriever(load_max_docs=1, get_full_documents=True)
        docs = retriever._get_relevant_documents("optimization")

        assert len(docs) > 0
        # Full documents should be longer
        assert len(docs[0].page_content) > 100

    @pytest.mark.integration
    def test_arxiv_search_specific_query(self):
        """Test Arxiv search with specific query."""
        retriever = ArxivRetriever(load_max_docs=3)
        docs = retriever._get_relevant_documents("quantum computing")

        # Should return results
        assert len(docs) > 0

        # All should have non-empty page content
        assert all(len(doc.page_content) > 0 for doc in docs)
