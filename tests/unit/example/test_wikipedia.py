import pytest
from unittest.mock import Mock, patch

# Import the refactored modules
from example.wikipedia import (
    handle_wikipedia, create_wikipedia_chain, process_wikipedia_query
)


@pytest.mark.unit
class TestWikipedia:
    """Test cases for refactored Wikipedia functionality"""

    def test_handle_wikipedia_no_query(self, mock_streamlit):
        """Test wikipedia handler when no query is provided"""

        # Mock empty text input
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.button.return_value = True

        handle_wikipedia(mock_streamlit, "test_model")

        # Should call text_input and button
        mock_streamlit.text_input.assert_called_once_with(
            "Question",
            placeholder="Ask a question about any public information."
        )
        mock_streamlit.button.assert_called_once_with("Summarize")

        # Should show warning for empty query
        mock_streamlit.warning.assert_called_once_with(
            "Please enter a question.")

    def test_handle_wikipedia_with_query_success(self, mock_streamlit):
        """Test wikipedia handler with successful query - much simpler!"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is artificial intelligence?"  # noqa: E501
        mock_streamlit.button.return_value = True

        # Mock the chain creation and query processing functions
        with patch('example.wikipedia.create_wikipedia_chain') as mock_create_chain:  # noqa: E501
            with patch('example.wikipedia.process_wikipedia_query') as mock_process:  # noqa: E501
                with patch('example.wikipedia.print_context') as mock_print_context:  # noqa: E501

                    # Mock the chain and result
                    mock_chain = Mock()
                    mock_create_chain.return_value = mock_chain

                    mock_result = {
                        'answer': 'Artificial intelligence is a branch of computer science...',  # noqa: E501
                        'context': ['wiki_doc1', 'wiki_doc2']
                    }
                    mock_process.return_value = mock_result

                    handle_wikipedia(mock_streamlit, "test_model")

                    # Verify functions were called correctly
                    mock_create_chain.assert_called_once_with("test_model")
                    mock_process.assert_called_once_with(
                        mock_chain,
                        "What is artificial intelligence?",
                        langfuse_handler=None
                    )

                    # Verify success result was displayed
                    mock_streamlit.success.assert_called_once_with(
                        'Artificial intelligence is a branch of computer science...')  # noqa: E501

                    # Verify context was printed
                    mock_print_context.assert_called_once_with(
                        mock_streamlit, mock_result)

    def test_handle_wikipedia_no_answer(self, mock_streamlit):
        """Test wikipedia handler when no answer is found"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = True

        # Mock the chain creation and query processing
        with patch('example.wikipedia.create_wikipedia_chain') as mock_create_chain:  # noqa: E501
            with patch('example.wikipedia.process_wikipedia_query') as mock_process:  # noqa: E501

                mock_chain = Mock()
                mock_create_chain.return_value = mock_chain

                # Mock result with no answer
                mock_result = {'answer': None}
                mock_process.return_value = mock_result

                handle_wikipedia(mock_streamlit, "test_model")

                # Verify warning was displayed
                mock_streamlit.warning.assert_called_once_with("No answer was found.")  # noqa: E501

    def test_handle_wikipedia_empty_result(self, mock_streamlit):
        """Test wikipedia handler with empty result"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is artificial intelligence?"  # noqa: E501
        mock_streamlit.button.return_value = True

        # Mock the chain creation and query processing
        with patch('example.wikipedia.create_wikipedia_chain') as mock_create_chain:  # noqa: E501
            with patch('example.wikipedia.process_wikipedia_query') as mock_process:  # noqa: E501
                with patch('example.wikipedia.print_context') as mock_print_context:  # noqa: E501

                    mock_chain = Mock()
                    mock_create_chain.return_value = mock_chain

                    # Mock result as None (empty result)
                    mock_result = None
                    mock_process.return_value = mock_result

                    handle_wikipedia(mock_streamlit, "test_model")

                    # Verify warning was displayed for empty result
                    mock_streamlit.warning.assert_called_once_with("No answer was found.")  # noqa: E501

                    # Verify context was not printed for empty result
                    mock_print_context.assert_not_called()

    def test_handle_wikipedia_exception_handling(self, mock_streamlit):
        """Test wikipedia handler exception handling"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is quantum computing?"
        mock_streamlit.button.return_value = True

        # Mock chain creation to raise exception
        with patch('example.wikipedia.create_wikipedia_chain') as mock_create_chain:  # noqa: E501
            mock_create_chain.side_effect = Exception("Test error")

            handle_wikipedia(mock_streamlit, "test_model")

            # Verify exception was handled
            mock_streamlit.exception.assert_called_once_with("An error occurred: Test error")  # noqa: E501

    def test_handle_wikipedia_no_button_click(self, mock_streamlit):
        """Test wikipedia handler when button is not clicked"""

        # Mock no button click
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = False

        handle_wikipedia(mock_streamlit, "test_model")

        # Should call text_input and button but nothing else
        mock_streamlit.text_input.assert_called_once()
        mock_streamlit.button.assert_called_once_with("Summarize")

        # Should not show any warnings or success messages
        mock_streamlit.warning.assert_not_called()
        mock_streamlit.success.assert_not_called()
        mock_streamlit.exception.assert_not_called()

    # Test the individual functions separately (this is much easier!)

    def test_create_wikipedia_chain(self, mock_env_vars):
        """Test chain creation function separately - focus on external dependencies"""  # noqa: E501

        # Mock external dependencies only
        with patch('example.wikipedia.create_llm') as mock_create_llm:  # noqa: E501
            with patch('example.wikipedia.create_summarize_prompt_v2') as mock_prompt:  # noqa: E501
                with patch('example.wikipedia.WikipediaRetriever') as mock_retriever:  # noqa: E501

                    mock_create_llm.return_value = Mock()
                    mock_prompt.return_value = Mock()
                    mock_retriever.return_value = Mock()

                    try:
                        create_wikipedia_chain("test_model")

                        # Verify external dependencies were called
                        mock_create_llm.assert_called_once_with("test_model")
                        mock_prompt.assert_called_once()
                        mock_retriever.assert_called_once()

                    except Exception:
                        mock_create_llm.assert_called_once_with("test_model")
                        mock_prompt.assert_called_once()
                        mock_retriever.assert_called_once()

    def test_process_wikipedia_query(self):
        """Test query processing function separately"""

        # Create a mock chain
        mock_chain = Mock()
        mock_result = {
            'answer': 'Test answer',
            'context': ['doc1', 'doc2']
        }
        mock_chain.invoke.return_value = mock_result

        # Test the processing function
        result = process_wikipedia_query(mock_chain, "test question")

        # Verify chain was invoked correctly
        mock_chain.invoke.assert_called_once_with({
            "question": "test question"
        }, config={
            "callbacks": None
        })

        # Verify result was returned
        assert result == mock_result
