"""Unit tests for custom_retrievers module with comprehensive coverage."""

from unittest.mock import MagicMock, Mock, patch

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


class TestWikipediaRetrieverSuccessful:
    """Unit tests for successful WikipediaRetriever operations."""

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_get_relevant_documents_success(self, mock_page, mock_search):
        """Test successful document retrieval with mock data."""
        mock_search.return_value = ["Python", "Programming"]

        page1 = Mock()
        page1.content = "Python is a high-level programming language."
        page1.url = "https://wikipedia.org/Python"
        page1.title = "Python"

        page2 = Mock()
        page2.content = "Programming is the process of creating software."
        page2.url = "https://wikipedia.org/Programming"
        page2.title = "Programming"

        mock_page.side_effect = [page1, page2]

        retriever = WikipediaRetriever(top_k_results=2)
        docs = retriever._get_relevant_documents("Python programming")

        assert len(docs) == 2
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)
        for doc in docs:
            assert 'source' in doc.metadata
            assert 'title' in doc.metadata


class TestWikipediaAPIWrapperSuccessful:
    """Unit tests for successful WikipediaAPIWrapper operations."""

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_run_success(self, mock_page, mock_search):
        """Test successful wrapper.run() with mock data."""
        mock_search.return_value = ["AI"]

        page = Mock()
        page.summary = "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems."  # noqa: E501
        mock_page.return_value = page

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run("Artificial Intelligence")

        assert isinstance(result, str)
        assert len(result) > 50

    @patch('internal.custom_retrievers.wikipedia.page')
    def test_get_page_content_success(self, mock_page):
        """Test successful get_page_content() with mock data."""
        page = Mock()
        page.content = "Isaac Newton was an English mathematician, physicist, astronomer, author and political figure. He is widely recognized as one of the most influential scientists of all time."  # noqa: E501
        mock_page.return_value = page

        wrapper = WikipediaAPIWrapper()
        result = wrapper.get_page_content("Isaac Newton")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 100


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


class TestArxivRetrieverSuccessful:
    """Unit tests for successful ArxivRetriever operations."""

    @patch('internal.custom_retrievers.arxiv.Client')
    @patch('internal.custom_retrievers.arxiv.Search')
    def test_get_relevant_documents_success(self, mock_search_class, mock_client_class):  # noqa: E501
        """Test successful document retrieval with mock Arxiv data."""
        # Create mock paper results
        paper1 = Mock()
        paper1.title = "Machine Learning Fundamentals"
        paper1.summary = "This paper covers the basics of machine learning theory and practice."  # noqa: E501
        paper1.arxiv_url = "http://arxiv.org/abs/2101.00001v1"
        paper1.pdf_url = "http://arxiv.org/pdf/2101.00001v1"
        paper1.published = "2021-01-01T00:00:00Z"
        author1 = Mock()
        author1.name = "John Smith"
        paper1.authors = [author1]

        paper2 = Mock()
        paper2.title = "Deep Learning Advances"
        paper2.summary = "Recent advances in deep learning techniques."
        paper2.arxiv_url = "http://arxiv.org/abs/2101.00002v1"
        paper2.pdf_url = "http://arxiv.org/pdf/2101.00002v1"
        paper2.published = "2021-01-02T00:00:00Z"
        author2 = Mock()
        author2.name = "Jane Doe"
        paper2.authors = [author2]

        # Mock the client and search
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.results.return_value = [paper1, paper2]

        retriever = ArxivRetriever(load_max_docs=2)
        docs = retriever._get_relevant_documents("machine learning")

        # Should have created documents from papers
        assert len(docs) == 2
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)


class TestWikipediaRetrieverUnitComprehensive:
    """Comprehensive unit tests for WikipediaRetriever with mocked Wikipedia API."""  # noqa: E501

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_wikipedia_search_python(self, mock_page, mock_search):
        """Test retrieving Wikipedia documents - mocked."""
        # Mock the search results
        mock_search.return_value = ['Python', 'Python (programming language)']

        # Mock the page objects
        mock_page_obj1 = MagicMock()
        mock_page_obj1.content = 'Python is a programming language...' * 50
        mock_page_obj1.url = 'https://en.wikipedia.org/wiki/Python'
        mock_page_obj1.title = 'Python'

        mock_page_obj2 = MagicMock()
        mock_page_obj2.content = 'Python is a high-level language...' * 50
        mock_page_obj2.url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'  # noqa: E501
        mock_page_obj2.title = 'Python (programming language)'

        mock_page.side_effect = [mock_page_obj1, mock_page_obj2]

        # Test the retriever
        retriever = WikipediaRetriever(top_k_results=2)
        docs = retriever._get_relevant_documents('Python')

        # Verify results
        assert len(docs) == 2
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)

        for doc in docs:
            assert 'source' in doc.metadata
            assert 'title' in doc.metadata
            assert len(doc.page_content) > 100

        # Verify mock was called correctly
        mock_search.assert_called_once_with('Python', results=2)

    @patch('internal.custom_retrievers.wikipedia.search')
    def test_wikipedia_search_empty_result(self, mock_search):
        """Test handling empty search results."""
        mock_search.return_value = []

        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents('xyzabc123nonexistent')

        assert len(docs) == 0
        assert isinstance(docs, list)

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_wikipedia_search_with_page_error(self, mock_page, mock_search):
        """Test handling PageError gracefully."""
        mock_search.return_value = ['Result1', 'Result2']

        # First page fails with PageError, second succeeds
        mock_page_obj = MagicMock()
        mock_page_obj.content = 'Valid content...' * 50
        mock_page_obj.url = 'https://en.wikipedia.org/wiki/Result2'
        mock_page_obj.title = 'Result2'

        mock_page.side_effect = [
            wikipedia.exceptions.PageError('Not found'),
            mock_page_obj
        ]

        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents('test')

        # Should return only the valid result
        assert len(docs) == 1
        assert docs[0].metadata['title'] == 'Result2'

    @patch('internal.custom_retrievers.wikipedia.search')
    def test_wikipedia_search_api_exception(self, mock_search):
        """Test handling API exceptions."""
        mock_search.side_effect = Exception('API Error')

        retriever = WikipediaRetriever()
        docs = retriever._get_relevant_documents('test')

        # Should return empty list on API error
        assert len(docs) == 0


class TestWikipediaAPIWrapperUnitComprehensive:
    """Comprehensive unit tests for WikipediaAPIWrapper with mocked API."""

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_wrapper_run_success(self, mock_page, mock_search):
        """Test wrapper.run() with mocked API."""
        mock_search.return_value = ['Artificial Intelligence']

        mock_page_obj = MagicMock()
        mock_page_obj.summary = 'AI is...' * 100
        mock_page.return_value = mock_page_obj

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run('Artificial Intelligence')

        assert isinstance(result, str)
        assert 'AI is' in result

    @patch('internal.custom_retrievers.wikipedia.search')
    def test_wrapper_run_empty_search(self, mock_search):
        """Test wrapper.run() with empty search results."""
        mock_search.return_value = []

        wrapper = WikipediaAPIWrapper()
        result = wrapper.run('xyzabc')

        assert result == 'No Wikipedia results found.'

    @patch('internal.custom_retrievers.wikipedia.search')
    @patch('internal.custom_retrievers.wikipedia.page')
    def test_wrapper_get_page_content(self, mock_page, mock_search):
        """Test getting full page content - mocked."""
        mock_page_obj = MagicMock()
        mock_page_obj.content = 'Isaac Newton was...' * 100
        mock_page.return_value = mock_page_obj

        wrapper = WikipediaAPIWrapper()
        result = wrapper.get_page_content('Isaac Newton')

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 200
        assert 'Isaac Newton' in result

    @patch('internal.custom_retrievers.wikipedia.page')
    def test_wrapper_get_page_content_not_found(self, mock_page):
        """Test handling page not found."""
        mock_page.side_effect = wikipedia.exceptions.PageError('Not found')

        wrapper = WikipediaAPIWrapper()
        result = wrapper.get_page_content('xyz123nonexistent')

        assert result is None
