from unittest.mock import Mock, patch

# Import the modules we're testing
from example.arxiv import handle_arxiv


class TestArxiv:
    """Test cases for Arxiv functionality"""

    @patch('example.arxiv.create_llm')
    def test_handle_arxiv_no_query(self, mock_create_llm, mock_streamlit):
        """Test arxiv handler when no query is provided"""

        # Mock empty text input
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.button.return_value = True

        handle_arxiv(mock_streamlit, "test_model")

        # Should call text_input and button
        mock_streamlit.text_input.assert_called_once()
        mock_streamlit.button.assert_called_once_with("Summarize")

        # Should show warning for empty query
        mock_streamlit.warning.assert_called_once_with(
            "Please enter a question.")

        # Should not create LLM for empty query
        mock_create_llm.assert_not_called()

    @patch('example.arxiv.print_context')
    def test_handle_arxiv_with_query_success(self, mock_print_context, mock_streamlit):  # noqa: E501
        """Test arxiv handler with successful query"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = True

        # Mock the final result
        mock_result = {
            'answer': 'Machine learning is a subset of AI...',
            'context': ['doc1', 'doc2']
        }

        # Mock the entire chain construction and execution
        with patch('example.arxiv.create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm

            with patch('example.arxiv.ArxivRetriever') as mock_retriever:
                with patch('example.arxiv.StrOutputParser'):
                    with patch('example.arxiv.RunnablePassthrough') as mock_runnable:  # noqa: E501
                        # Mock a complete chain that when invoked returns our result  # noqa: E501
                        mock_chain = Mock()
                        mock_chain.invoke.return_value = mock_result

                        # Mock the chain building process
                        mock_intermediate = Mock()
                        mock_intermediate.assign.return_value = mock_chain
                        mock_runnable.assign.return_value = mock_intermediate

                        handle_arxiv(mock_streamlit, "test_model")

                        # Verify external dependencies were called
                        mock_create_llm.assert_called_once_with(
                            "test_model")
                        mock_retriever.assert_called_once_with(
                            load_max_docs=3,
                            get_full_documents=True,
                        )

                        # Verify chain was invoked
                        mock_chain.invoke.assert_called_once_with({
                            "question": "What is machine learning?"
                        }, config={
                            "callbacks": None
                        })

                        # Verify success result was displayed
                        mock_streamlit.success.assert_called_once_with(
                            'Machine learning is a subset of AI...')
                        mock_print_context.assert_called_once_with(
                            mock_streamlit, mock_result)

    @patch('example.arxiv.print_context')
    def test_handle_arxiv_no_answer(self, mock_print_context, mock_streamlit):
        """Test arxiv handler when no answer is found"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = True

        # Mock the final result
        mock_result = {'answer': None}

        # Mock the entire chain construction and execution
        with patch('example.arxiv.create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm

            with patch('example.arxiv.ArxivRetriever'):
                with patch('example.arxiv.StrOutputParser'):
                    with patch('example.arxiv.RunnablePassthrough') as mock_runnable:  # noqa: E501
                        # Mock a complete chain that when invoked returns our result  # noqa: E501
                        mock_chain = Mock()
                        mock_chain.invoke.return_value = mock_result

                        # Mock the chain building process
                        mock_intermediate = Mock()
                        mock_intermediate.assign.return_value = mock_chain
                        mock_runnable.assign.return_value = mock_intermediate

                        handle_arxiv(mock_streamlit, "test_model")

                        # Verify warning was displayed
                        mock_streamlit.warning.assert_called_once_with(
                            "No answer was found.")

    @patch('example.arxiv.create_llm')
    def test_handle_arxiv_exception_handling(self, mock_create_llm, mock_streamlit):  # noqa: E501
        """Test arxiv handler exception handling"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = True

        # Mock LLM creation to raise exception
        mock_create_llm.side_effect = Exception("Test error")

        handle_arxiv(mock_streamlit, "test_model")

        # Verify exception was handled
        mock_streamlit.exception.assert_called_once_with(
            "An error occurred: Test error")

    def test_handle_arxiv_no_button_click(self, mock_streamlit):
        """Test arxiv handler when button is not clicked"""

        # Mock no button click
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = False

        handle_arxiv(mock_streamlit, "test_model")

        # Should call text_input and button but nothing else
        mock_streamlit.text_input.assert_called_once()
        mock_streamlit.button.assert_called_once_with("Summarize")

        # Should not show any warnings or success messages
        mock_streamlit.warning.assert_not_called()
        mock_streamlit.success.assert_not_called()
        mock_streamlit.exception.assert_not_called()
