import os
import pytest
from unittest.mock import Mock, patch
import tempfile
import sys

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Test fixtures for common test data
@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        "models": [
            {"name": "llama3.2:latest"},
            {"name": "mistral:latest"},
            {"name": "codellama:latest"}
        ]
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables"""
    env_vars = {
        "OLLAMA_HOST": "http://localhost:11434",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "BRAVE_SEARCH_API_KEY": "test-api-key"
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj"


@pytest.fixture
def temp_pdf_file(sample_pdf_content):
    """Create temporary PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(sample_pdf_content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components"""
    class MockSessionState:
        def __init__(self):
            self._data = {}

        def __contains__(self, key):
            return key in self._data

        def __getattr__(self, key):
            return self._data.get(key)

        def __setattr__(self, key, value):
            if key.startswith('_'):
                super().__setattr__(key, value)
            else:
                self._data[key] = value

    mock_st = Mock()
    mock_st.selectbox.return_value = "test_option"
    mock_st.file_uploader.return_value = None
    mock_st.text_input.return_value = "test input"
    mock_st.button.return_value = False
    mock_st.session_state = MockSessionState()
    mock_st.chat_input.return_value = None
    mock_st.spinner.return_value.__enter__ = Mock()
    mock_st.spinner.return_value.__exit__ = Mock()
    mock_st.chat_message.return_value.__enter__ = Mock()
    mock_st.chat_message.return_value.__exit__ = Mock()
    return mock_st


@pytest.fixture
def mock_langchain_llm():
    """Mock LangChain LLM"""
    mock_llm = Mock()
    mock_llm.invoke.return_value = "Test AI response"
    mock_llm.stream.return_value = iter(["Test", " AI", " response"])
    return mock_llm


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    mock_client = Mock()
    mock_collection = Mock()

    # Mock collection methods
    mock_collection.query.return_value = {
        'documents': [['Test document content']],
        'metadatas': [[{'source': 'test.pdf'}]],
        'distances': [[0.1]]
    }
    mock_collection.add.return_value = None
    mock_collection.delete.return_value = None

    # Mock client methods
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.delete_collection.return_value = None

    return mock_client


@pytest.fixture
def mock_postgres_connection():
    """Mock PostgreSQL connection"""
    mock_conn = Mock()
    mock_cursor = Mock()

    mock_cursor.fetchall.return_value = [
        ('Test City', 'Test State', 100000),
        ('Another City', 'Another State', 200000)
    ]
    mock_cursor.execute.return_value = None

    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None

    return mock_conn


@pytest.fixture
def sample_city_data():
    """Sample city data for testing"""
    return [
        {"city": "New York", "state": "NY", "population": 8000000},
        {"city": "Los Angeles", "state": "CA", "population": 4000000},
        {"city": "Chicago", "state": "IL", "population": 2700000}
    ]


@pytest.fixture
def sample_country_data():
    """Sample country data for testing"""
    return [
        {"name": "United States", "code": "US", "capital": "Washington, D.C."},
        {"name": "Canada", "code": "CA", "capital": "Ottawa"},
        {"name": "Mexico", "code": "MX", "capital": "Mexico City"}
    ]


@pytest.fixture
def mock_wikipedia_response():
    """Mock Wikipedia API response"""
    return {
        "title": "Test Article",
        "content": "This is test content from Wikipedia article.",
        "url": "https://en.wikipedia.org/wiki/Test_Article"
    }


@pytest.fixture
def mock_arxiv_response():
    """Mock ArXiv API response"""
    return [
        {
            "title": "Test Research Paper",
            "summary": "This is a test research paper summary.",
            "authors": ["Test Author 1", "Test Author 2"],
            "published": "2024-01-01",
            "pdf_url": "https://arxiv.org/pdf/2401.0001.pdf"
        }
    ]
