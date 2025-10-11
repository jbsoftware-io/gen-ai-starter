from unittest.mock import Mock, patch

# Import the modules we're testing
from example.chroma import handle_chroma, process_chroma_query
from internal.util import getCollectionName


class TestChroma:
    """Test cases for Chroma vector database functionality"""

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
                        mock_chain, "Test question")

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
        })

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
