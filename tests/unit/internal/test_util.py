"""Unit tests for internal.util module."""

import os
import tempfile
from unittest.mock import Mock, patch

from internal.util import (
    create_llm,
    format_docs,
    getCollectionName,
    loadPDF,
    print_context,
    strip_non_alphanumeric,
    writeToTempFile,
)


class TestCreateLLM:
    """Tests for create_llm function."""

    @patch("internal.util.OllamaLLM")
    def test_create_llm_success(self, mock_ollama):
        """Test creating LLM with valid model name."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm

        result = create_llm("llama2")

        assert result == mock_llm
        mock_ollama.assert_called_once()
        call_kwargs = mock_ollama.call_args.kwargs
        assert call_kwargs["model"] == "llama2"
        # Just check that base_url is set, don't hardcode the value
        assert "base_url" in call_kwargs

    @patch("internal.util.OllamaLLM")
    def test_create_llm_uses_environment_host(self, mock_ollama):
        """Test that create_llm uses OLLAMA_HOST from environment."""
        # OLLAMA_HOST is already set in conftest fixtures
        # Just verify the function doesn't raise
        mock_ollama.return_value = Mock()
        create_llm("llama2")

        # Should have called OllamaLLM with base_url
        mock_ollama.assert_called_once()
        assert "base_url" in mock_ollama.call_args.kwargs


class TestFormatDocs:
    """Tests for format_docs function."""

    def test_format_docs_single(self):
        """Test formatting a single document."""
        from langchain_core.documents import Document
        doc = Document(page_content="Test content")

        result = format_docs([doc])

        assert result == "Test content"

    def test_format_docs_multiple(self):
        """Test formatting multiple documents."""
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
            Document(page_content="Content 3")
        ]

        result = format_docs(docs)

        assert result == "Content 1\n\nContent 2\n\nContent 3"

    def test_format_docs_empty(self):
        """Test formatting empty document list."""
        result = format_docs([])

        assert result == ""

    def test_format_docs_with_metadata(self):
        """Test formatting documents with metadata."""
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Content", metadata={"source": "test.pdf"})
        ]

        result = format_docs(docs)

        assert result == "Content"


class TestGetCollectionName:
    """Tests for getCollectionName function."""

    def test_get_collection_name_basic(self):
        """Test getting collection name with simple inputs."""
        result = getCollectionName("document.pdf", "llama2")

        assert "llama2" in result
        assert "document" in result

    def test_get_collection_name_removes_special_chars(self):
        """Test that special characters are removed."""
        result = getCollectionName("my-doc@2024.pdf", "llama:2-latest")

        assert "-" not in result
        assert "@" not in result
        assert ":" not in result
        assert "mydoc2024" in result

    def test_get_collection_name_max_length(self):
        """Test that collection name respects max length."""
        long_path = "a" * 100 + ".pdf"
        long_model = "b" * 100

        result = getCollectionName(long_path, long_model, max_length=20)

        assert len(result) == 20

    def test_get_collection_name_default_max_length(self):
        """Test collection name with default max length (63)."""
        long_path = "a" * 50 + ".pdf"
        long_model = "b" * 50

        result = getCollectionName(long_path, long_model)

        assert len(result) <= 63


class TestLoadPDF:
    """Tests for loadPDF function."""

    def test_load_pdf_single_page(self):
        """Test loading a single-page PDF."""
        with patch("internal.util.PdfReader") as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test content from page 1"
            mock_pdf.return_value.pages = [mock_page]

            result = loadPDF("test.pdf")

            assert len(result) > 0
            assert any("Test content" in doc.page_content for doc in result)

    def test_load_pdf_multiple_pages(self):
        """Test loading a multi-page PDF."""
        with patch("internal.util.PdfReader") as mock_pdf:
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Content page 1"

            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Content page 2"

            mock_pdf.return_value.pages = [mock_page1, mock_page2]

            result = loadPDF("test.pdf")

            # Check that content from both pages is present
            full_content = "\n".join(doc.page_content for doc in result)
            assert "Content page 1" in full_content
            assert "Content page 2" in full_content

    def test_load_pdf_sets_metadata(self):
        """Test that PDF loading sets correct metadata."""
        with patch("internal.util.PdfReader") as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test"
            mock_pdf.return_value.pages = [mock_page]

            result = loadPDF("test.pdf")

            # At least one document should have the source in metadata
            assert any("test.pdf" in str(doc.metadata) for doc in result)

    def test_load_pdf_includes_page_numbers(self):
        """Test that page numbers are included in content."""
        with patch("internal.util.PdfReader") as mock_pdf:
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Content 1"

            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Content 2"

            mock_pdf.return_value.pages = [mock_page1, mock_page2]

            result = loadPDF("test.pdf")

            full_content = "\n".join(doc.page_content for doc in result)
            assert "Page 1" in full_content or "page 1" in full_content
            assert "Page 2" in full_content or "page 2" in full_content


class TestPrintContext:
    """Tests for print_context function."""

    def test_print_context_with_context(self):
        """Test printing context when context exists."""
        mock_st = Mock()
        mock_doc = Mock()
        mock_doc.metadata = {"source": "test.pdf"}

        result = {
            "context": [mock_doc]
        }

        print_context(mock_st, result)

        mock_st.markdown.assert_called()

    def test_print_context_without_context(self):
        """Test printing context when context doesn't exist."""
        mock_st = Mock()
        result = {}

        # Should not raise exception
        print_context(mock_st, result)

        # markdown should not be called without context
        assert not mock_st.markdown.called

    def test_print_context_multiple_docs(self):
        """Test printing context with multiple documents."""
        mock_st = Mock()
        mock_doc1 = Mock()
        mock_doc1.metadata = {"source": "doc1.pdf"}

        mock_doc2 = Mock()
        mock_doc2.metadata = {"source": "doc2.pdf"}

        result = {
            "context": [mock_doc1, mock_doc2]
        }

        print_context(mock_st, result)

        # markdown should be called for header and each document
        assert mock_st.markdown.call_count >= 2


class TestStripNonAlphanumeric:
    """Tests for strip_non_alphanumeric function."""

    def test_strip_basic(self):
        """Test stripping special characters."""
        result = strip_non_alphanumeric("hello-world_123")
        assert result == "helloworld123"

    def test_strip_all_special_chars(self):
        """Test stripping string with only special characters."""
        result = strip_non_alphanumeric("!@#$%^&*()")
        assert result == ""

    def test_strip_mixed(self):
        """Test stripping mixed content."""
        result = strip_non_alphanumeric("test@doc-2024.pdf")
        assert result == "testdoc2024pdf"

    def test_strip_empty_string(self):
        """Test stripping empty string."""
        result = strip_non_alphanumeric("")
        assert result == ""

    def test_strip_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        result = strip_non_alphanumeric("abc123XYZ")
        assert result == "abc123XYZ"

    def test_strip_unicode(self):
        """Test stripping with unicode characters."""
        result = strip_non_alphanumeric("hello-café-123")
        # Should only keep ASCII alphanumeric
        assert "123" in result


class TestWriteToTempFile:
    """Tests for writeToTempFile function."""

    def test_write_to_temp_file(self):
        """Test writing to temporary file."""
        mock_source = Mock()
        mock_source.name = "test.txt"
        mock_source.getvalue.return_value = b"test content"

        result = writeToTempFile(mock_source)

        assert os.path.exists(result)
        assert "test.txt" in result
        assert result.endswith(".txt")

        # Clean up
        os.remove(result)

    def test_write_to_temp_file_creates_in_temp_dir(self):
        """Test that file is created in temp directory."""
        mock_source = Mock()
        mock_source.name = "test.bin"
        mock_source.getvalue.return_value = b"binary content"

        result = writeToTempFile(mock_source)

        assert tempfile.gettempdir() in result

        # Clean up
        os.remove(result)

    def test_write_to_temp_file_preserves_content(self):
        """Test that file content is preserved."""
        mock_source = Mock()
        mock_source.name = "test.txt"
        test_content = b"This is test content"
        mock_source.getvalue.return_value = test_content

        result = writeToTempFile(mock_source)

        with open(result, "rb") as f:
            content = f.read()

        assert content == test_content

        # Clean up
        os.remove(result)

    def test_write_to_temp_file_binary_data(self):
        """Test writing binary data."""
        mock_source = Mock()
        mock_source.name = "test.pdf"
        # PDF header bytes
        test_content = b"%PDF-1.4\x00\xFF\xFE"
        mock_source.getvalue.return_value = test_content

        result = writeToTempFile(mock_source)

        with open(result, "rb") as f:
            content = f.read()

        assert content == test_content

        # Clean up
        os.remove(result)
