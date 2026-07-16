"""Unit tests for custom_retrievers module - error handling and edge cases."""

from unittest.mock import Mock, patch

import wikipedia

from internal.custom_retrievers import (
    ArxivRetriever,
    WikipediaAPIWrapper,
    WikipediaRetriever,
)


class TestWikipediaAPIWrapperErrorHandling:
    """Tests for WikipediaAPIWrapper error handling."""

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_run_disambiguation_error(self, mock_page, mock_search):
        """Test Wikipedia search with disambiguation error."""
        mock_search.return_value = ["Test"]
        mock_page.side_effect = wikipedia.exceptions.DisambiguationError("Test", ["Option 1", "Option 2"])  # noqa: E501

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run("test query")

        assert "Disambiguation page" in result
        assert "Option 1" in result

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_run_page_error(self, mock_page, mock_search):
        """Test Wikipedia search with page not found error."""
        mock_search.return_value = ["Test"]
        mock_page.side_effect = wikipedia.exceptions.PageError("Test")

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run("test query")

        assert result == "Page not found."

    @patch('internal.custom_retrievers.wikipedia.search')
    def test_run_no_results(self, mock_search):
        """Test Wikipedia search with no results."""
        mock_search.return_value = []

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run("nonexistent query")

        assert result == "No Wikipedia results found."

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_run_generic_exception(self, mock_page, mock_search):
        """Test Wikipedia search with generic exception."""
        mock_search.return_value = ["Test"]
        mock_page.side_effect = Exception("Connection error")

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run("test query")

        assert "Error searching Wikipedia" in result

    @patch('internal.custom_retrievers.wikipedia.page')
    def test_get_page_content_not_found(self, mock_page):
        """Test getting page content when page not found."""
        mock_page.side_effect = wikipedia.exceptions.PageError("Test")

        wrapper = WikipediaAPIWrapper()
        result = wrapper.get_page_content("Nonexistent")

        assert result is None

    @patch('internal.custom_retrievers.wikipedia.page')
    def test_get_page_content_disambiguation(self, mock_page):
        """Test getting page content for disambiguation page."""
        mock_page.side_effect = wikipedia.exceptions.DisambiguationError("Test", [])  # noqa: E501

        wrapper = WikipediaAPIWrapper()
        result = wrapper.get_page_content("Ambiguous")

        assert result is None

    @patch('internal.custom_retrievers.wikipedia.page')
    def test_get_page_content_exception(self, mock_page):
        """Test getting page content with exception."""
        mock_page.side_effect = Exception("Network error")

        wrapper = WikipediaAPIWrapper()
        result = wrapper.get_page_content("Test")

        assert result is None


class TestWikipediaRetrieverErrorHandling:
    """Tests for WikipediaRetriever error handling."""

    def test_init_default(self):
        """Test WikipediaRetriever initialization."""
        retriever = WikipediaRetriever()
        assert retriever.top_k_results == 4
        assert retriever.wiki_wrapper is not None

    def test_init_custom_top_k(self):
        """Test WikipediaRetriever with custom top_k."""
        retriever = WikipediaRetriever(top_k_results=10)
        assert retriever.top_k_results == 10

    @patch('internal.custom_retrievers.wikipedia.search')
    def test_get_relevant_documents_search_exception(self, mock_search):
        """Test retrieving documents when search raises exception."""
        mock_search.side_effect = Exception("Network error")

        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents("test query")

        assert docs == []

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_get_relevant_documents_skips_errors(self, mock_page, mock_search):
        """Test that retriever skips items with errors and continues."""
        mock_search.return_value = ["Good", "Bad", "Good2"]

        good_page = Mock()
        good_page.content = "Good content"
        good_page.url = "https://example.com/good"
        good_page.title = "Good"

        good_page2 = Mock()
        good_page2.content = "Good content 2"
        good_page2.url = "https://example.com/good2"
        good_page2.title = "Good2"

        mock_page.side_effect = [
            good_page,
            wikipedia.exceptions.DisambiguationError("Bad", []),
            good_page2
        ]

        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents("test query")

        # Should have 2 docs (skipped the disambiguation error)
        assert len(docs) == 2


class TestArxivRetrieverErrorHandling:
    """Tests for ArxivRetriever error handling."""

    def test_init_defaults(self):
        """Test ArxivRetriever initialization defaults."""
        retriever = ArxivRetriever()
        assert retriever.load_max_docs == 3
        assert retriever.get_full_documents is False

    def test_init_custom(self):
        """Test ArxivRetriever with custom settings."""
        retriever = ArxivRetriever(load_max_docs=5, get_full_documents=True)
        assert retriever.load_max_docs == 5
        assert retriever.get_full_documents is True

    @patch('internal.custom_retrievers.arxiv.Client')
    def test_get_relevant_documents_client_error(self, mock_client_class):
        """Test retriever when client raises exception."""
        mock_client_class.side_effect = Exception("API Error")

        retriever = ArxivRetriever()
        docs = retriever._get_relevant_documents("test query")

        assert docs == []

    @patch('internal.custom_retrievers.arxiv.Client')
    @patch('internal.custom_retrievers.arxiv.Search')
    def test_get_relevant_documents_empty_results(self, mock_search_class, mock_client_class):  # noqa: E501
        """Test retriever when API returns no results."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.results.return_value = []

        retriever = ArxivRetriever()
        docs = retriever._get_relevant_documents("nonexistent query")

        assert docs == []
