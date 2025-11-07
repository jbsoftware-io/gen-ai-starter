from unittest.mock import Mock, patch

# Import the modules we're testing
from example.chroma import handle_chroma, process_chroma_query, vectorizePDF
from internal.util import getCollectionName


class TestChroma:
    """Test cases for Chroma vector database functionality"""
    def test_vectorizePDF_unit(self):

        class UploadedFile(Mock):
            def __init__(self, bytes_data, name):
                super().__init__()
                self._bytes_data = bytes_data
                self.name = name

            def read(self):
                return self._bytes_data

            def getvalue(self):
                return self._bytes_data
        mock_pdf_bytes = b"%PDF-1.4 valid pdf content"
        source_doc = UploadedFile(mock_pdf_bytes, "test.pdf")
        # Patch all dependencies inside vectorizePDF
        with patch("example.chroma.writeToTempFile", return_value="/tmp/test.pdf"), \
             patch("example.chroma.loadPDF", return_value=[{"page_content": "Test PDF content"}]), \
             patch("example.chroma.getCollectionName", return_value="test_collection"), \
             patch("example.chroma.OllamaEmbeddings") as mock_embeddings, \
             patch("example.chroma.Chroma") as mock_chroma, \
             patch("example.chroma.chromadb.HttpClient") as mock_http_client:  # noqa: E501
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_http_client.return_value.create_collection.return_value = mock_collection  # noqa: E501
            mock_vectorstore = Mock()
            mock_chroma.return_value = mock_vectorstore
            result = vectorizePDF(source_doc, "test_model")
            assert result == mock_vectorstore
            mock_chroma.assert_called_once()
            mock_embeddings.assert_called_once()
            mock_http_client.assert_called_once()

    def test_process_chroma_query_invokes_chain(self):
        from example.chroma import process_chroma_query
        mock_chain = Mock()
        mock_chain.invoke.return_value = {"answer": "Test answer"}
        search_query = "What is the capital of France?"
        result = process_chroma_query(mock_chain, search_query)
        mock_chain.invoke.assert_called_once_with(
            {"question": search_query}, config={"callbacks": None})
        assert result == {"answer": "Test answer"}

    def test_handle_chroma_no_file_upload(self, mock_streamlit):
        """Test chroma handler when no file is uploaded"""

        # Mock no file uploaded
        mock_streamlit.file_uploader.return_value = None

        handle_chroma(mock_streamlit, "test_model")

        # Should call file_uploader but not process anything
        mock_streamlit.file_uploader.assert_called_once()
        # Should call text_input and button UI components
        mock_streamlit.text_input.assert_called_once()
        mock_streamlit.button.assert_called_once()

    def test_handle_chroma_with_file_upload_success(self, mock_streamlit):
        """Test chroma handler with successful file processing - much simpler!"""  # noqa: E501

        # Mock file upload
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_streamlit.file_uploader.return_value = mock_file
        mock_streamlit.text_input.return_value = "Test question"
        mock_streamlit.button.return_value = True

        # Mock the vectorization, chain creation, and query processing functions  # noqa: E501
        with patch('example.chroma.vectorizePDF') as mock_vectorize:
            with patch('example.chroma.create_chroma_chain') as mock_create_chain:  # noqa: E501
                with patch('example.chroma.process_chroma_query') as mock_process:  # noqa: E501

                    # Mock the vectorstore, chain, and result
                    mock_vectorstore = Mock()
                    mock_vectorize.return_value = mock_vectorstore

                    mock_chain = Mock()
                    mock_create_chain.return_value = mock_chain

                    mock_result = {
                        'answer': 'This document discusses machine learning concepts...',  # noqa: E501
                        'context': ['doc1', 'doc2']
                    }
                    mock_process.return_value = mock_result

                    handle_chroma(mock_streamlit, "test_model")

                    # Verify functions were called correctly
                    mock_vectorize.assert_called_once_with(
                        mock_file, "test_model")
                    mock_create_chain.assert_called_once_with(
                        mock_vectorstore, "test_model")
                    mock_process.assert_called_once_with(
                        mock_chain, "Test question", langfuse_handler=None)

                    # Verify success result was displayed
                    mock_streamlit.success.assert_called_once_with(
                        'This document discusses machine learning concepts...'
                    )

    def test_handle_chroma_no_answer(self, mock_streamlit):
        """Test chroma handler when no answer is found"""

        # Mock file upload
        mock_file = Mock()
        mock_streamlit.file_uploader.return_value = mock_file
        mock_streamlit.text_input.return_value = "Test question"
        mock_streamlit.button.return_value = True

        # Mock the functions with no answer result
        with patch('example.chroma.vectorizePDF') as mock_vectorize:
            with patch('example.chroma.create_chroma_chain') as mock_create_chain:  # noqa: E501
                with patch('example.chroma.process_chroma_query') as mock_process:  # noqa: E501

                    mock_vectorize.return_value = Mock()
                    mock_create_chain.return_value = Mock()

                    # Mock result with no answer
                    mock_result = {'answer': None}
                    mock_process.return_value = mock_result

                    handle_chroma(mock_streamlit, "test_model")

                    # Verify warning was displayed
                    mock_streamlit.warning.assert_called_once_with(
                        "No answer was found.")

    def test_handle_chroma_exception_handling(self, mock_streamlit):
        """Test chroma handler exception handling"""

        # Mock file upload
        mock_file = Mock()
        mock_streamlit.file_uploader.return_value = mock_file
        mock_streamlit.text_input.return_value = "Test question"
        mock_streamlit.button.return_value = True

        # Mock vectorizePDF to raise exception
        with patch('example.chroma.vectorizePDF') as mock_vectorize:
            mock_vectorize.side_effect = Exception("Test error")

            handle_chroma(mock_streamlit, "test_model")

            # Verify exception was handled
            mock_streamlit.exception.assert_called_once_with(
                "An error occurred: Test error")

    def test_process_chroma_query(self):
        """Test query processing function separately"""

        # Create a mock chain
        mock_chain = Mock()
        mock_result = {
            'answer': 'Test answer',
            'context': ['doc1', 'doc2']
        }
        mock_chain.invoke.return_value = mock_result

        # Test the processing function
        result = process_chroma_query(mock_chain, "test question")

        # Verify chain was invoked correctly
        mock_chain.invoke.assert_called_once_with({
            "question": "test question"
        }, config={"callbacks": None})

        # Verify result was returned
        assert result == mock_result

    def test_environment_variables_loaded(self):
        """Test that environment variables are properly loaded from .env"""
        # Since we now use .env file, verify that environment variables are available  # noqa: E501
        import os
        assert os.getenv("OLLAMA_HOST") is not None
        assert os.getenv("CHROMA_HOST") is not None
        assert os.getenv("CHROMA_PORT") is not None

    def test_collection_name_generation(self):
        """Test collection name generation"""

        collection_name = getCollectionName("test_file.pdf", "test_model")

        # Expected: clean_model_name + clean_basename
        # "test_model" -> "testmodel" (stripped non-alphanumeric)
        # "test_file.pdf" -> "testfilepdf" (basename stripped)
        expected = "testmodeltestfilepdf"
        assert collection_name == expected
