from unittest.mock import Mock, patch

# Import the modules we're testing
from example.web import handle_web


class TestWeb:
    """Test cases for web search functionality"""

    def test_handle_web_no_api_key(self, mock_streamlit):
        """Test web handler when BRAVE_SEARCH_API_KEY is not set"""

        # Mock missing API key
        with patch('example.web.BRAVE_SEARCH_API_KEY', None):
            handle_web(mock_streamlit, "test_model")

            # Should show warning and return early
            mock_streamlit.warning.assert_called_once_with(
                "BRAVE_SEARCH_API_KEY not set, refer to README Optional Pre-requisites for instructions."  # noqa: E501
            )

            # Should not call other UI components
            mock_streamlit.text_input.assert_not_called()
            mock_streamlit.button.assert_not_called()

    def test_handle_web_no_query(self, mock_streamlit):
        """Test web handler when no query is provided"""

        # Mock empty text input
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.button.return_value = True

        with patch('example.web.BRAVE_SEARCH_API_KEY', 'test-api-key'):
            with patch('example.web.create_llm') as mock_create_llm:

                handle_web(mock_streamlit, "test_model")

                # Should call text_input and button
                mock_streamlit.text_input.assert_called_once_with(
                    "Question",
                    placeholder="Ask a question about any public information."
                )
                mock_streamlit.button.assert_called_once_with("Summarize")

                # Should show warning for empty query
                mock_streamlit.warning.assert_called_once_with(
                    "Please enter a question.")

                # Should not create LLM for empty query
                mock_create_llm.assert_not_called()

    def test_handle_web_with_query_success(self, mock_streamlit):
        """Test web handler with successful query"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is the latest news about AI?"  # noqa: E501
        mock_streamlit.button.return_value = True

        with patch('example.web.BRAVE_SEARCH_API_KEY', 'test-api-key'):
            with patch('example.web.create_llm') as mock_create_llm:
                with patch('example.web.create_summarize_prompt_v2') as mock_prompt:  # noqa: E501
                    with patch('example.web.BraveSearchLoader') as mock_loader:  # noqa: E501
                        with patch('example.web.RunnablePassthrough') as mock_runnable:  # noqa: E501
                            with patch('example.web.print_context') as mock_print_context:  # noqa: E501

                                # Mock external LangChain components
                                mock_llm = Mock()
                                mock_create_llm.return_value = mock_llm

                                mock_prompt_instance = Mock()
                                mock_prompt.return_value = mock_prompt_instance

                                mock_loader_instance = Mock()
                                mock_loader_instance.load.return_value = ['doc1', 'doc2']  # noqa: E501
                                mock_loader.return_value = mock_loader_instance

                                # Mock chain execution
                                mock_chain = Mock()
                                mock_result = {
                                    'answer': 'Recent AI developments include new language models...',  # noqa: E501
                                    'context': ['article1', 'article2']
                                }
                                mock_chain.invoke.return_value = mock_result
                                mock_runnable.assign.return_value.assign.return_value = mock_chain  # noqa: E501

                                handle_web(mock_streamlit, "test_model")

                                # Verify external dependencies were called
                                mock_create_llm.assert_called_once_with("test_model")  # noqa: E501
                                mock_prompt.assert_called_once()
                                mock_loader.assert_called_once_with(
                                    query="What is the latest news about AI?",
                                    api_key="test-api-key",
                                    search_kwargs={"count": 3}
                                )

                                # Verify chain was invoked
                                mock_chain.invoke.assert_called_once_with({
                                    "question": "What is the latest news about AI?"  # noqa: E501
                                })

                                # Verify success result was displayed
                                mock_streamlit.success.assert_called_once_with(
                                    'Recent AI developments include new language models...'  # noqa: E501
                                )

                                # Verify context was printed
                                mock_print_context.assert_called_once_with(
                                    mock_streamlit, mock_result
                                )

    def test_handle_web_no_answer(self, mock_streamlit):
        """Test web handler when no answer is found"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is quantum computing?"
        mock_streamlit.button.return_value = True

        with patch('example.web.BRAVE_SEARCH_API_KEY', 'test-api-key'):
            with patch('example.web.create_llm') as mock_create_llm:
                with patch('example.web.create_summarize_prompt_v2') as mock_prompt:  # noqa: E501
                    with patch('example.web.BraveSearchLoader') as mock_loader:  # noqa: E501
                        with patch('example.web.RunnablePassthrough') as mock_runnable:  # noqa: E501

                            # Mock external LangChain components
                            mock_create_llm.return_value = Mock()
                            mock_prompt.return_value = Mock()

                            mock_loader_instance = Mock()
                            mock_loader_instance.load.return_value = ['doc1']
                            mock_loader.return_value = mock_loader_instance

                            # Mock chain execution with no answer
                            mock_chain = Mock()
                            mock_result = {'answer': None}
                            mock_chain.invoke.return_value = mock_result
                            mock_runnable.assign.return_value.assign.return_value = mock_chain  # noqa: E501

                            handle_web(mock_streamlit, "test_model")

                            # Verify warning was displayed
                            mock_streamlit.warning.assert_called_once_with(
                                "No answer was found.")

    def test_handle_web_exception_handling(self, mock_streamlit):
        """Test web handler exception handling"""

        # Mock user input
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = True

        with patch('example.web.BRAVE_SEARCH_API_KEY', 'test-api-key'):
            with patch('example.web.create_llm') as mock_create_llm:

                # Mock LLM creation to raise exception
                mock_create_llm.side_effect = Exception("Test error")

                handle_web(mock_streamlit, "test_model")

                # Verify exception was handled
                mock_streamlit.exception.assert_called_once_with(
                    "An error occurred: Test error")

    def test_handle_web_no_button_click(self, mock_streamlit):
        """Test web handler when button is not clicked"""

        # Mock no button click
        mock_streamlit.text_input.return_value = "What is machine learning?"
        mock_streamlit.button.return_value = False

        with patch('example.web.BRAVE_SEARCH_API_KEY', 'test-api-key'):

            handle_web(mock_streamlit, "test_model")

            # Should call text_input and button but nothing else
            mock_streamlit.text_input.assert_called_once()
            mock_streamlit.button.assert_called_once_with("Summarize")

            # Should not show any warnings or success messages
            mock_streamlit.warning.assert_not_called()
            mock_streamlit.success.assert_not_called()
            mock_streamlit.exception.assert_not_called()

        # Should call text_input and button but nothing else
        mock_streamlit.text_input.assert_called_once()
        mock_streamlit.button.assert_called_once_with("Summarize")

        # Should not show any warnings or success messages
        mock_streamlit.warning.assert_not_called()
        mock_streamlit.success.assert_not_called()
        mock_streamlit.exception.assert_not_called()

    def test_handle_web_brave_search_loader_config(self, mock_streamlit):
        """Test that BraveSearchLoader is configured correctly"""

        mock_streamlit.text_input.return_value = "Test query"
        mock_streamlit.button.return_value = True

        with patch('example.web.BRAVE_SEARCH_API_KEY', 'test-api-key'):
            with patch('example.web.BraveSearchLoader') as mock_loader:
                with patch('example.web.create_llm') as mock_create_llm:
                    with patch('example.web.create_summarize_prompt_v2') as mock_prompt:  # noqa: E501
                        with patch('example.web.RunnablePassthrough') as mock_runnable:  # noqa: E501

                            # Setup the loader mock
                            mock_loader_instance = Mock()
                            mock_loader_instance.load.return_value = ['doc1']
                            mock_loader.return_value = mock_loader_instance

                            # Mock LLM and prompt
                            mock_create_llm.return_value = Mock()
                            mock_prompt.return_value = Mock()

                            # Mock chain execution to raise exception after loader is called  # noqa: E501
                            mock_chain = Mock()
                            mock_chain.invoke.side_effect = Exception("Stop execution")  # noqa: E501
                            mock_runnable.assign.return_value.assign.return_value = mock_chain  # noqa: E501

                            try:
                                handle_web(mock_streamlit, "test_model")
                            except Exception:
                                pass  # Expected exception to stop execution

                            mock_loader.assert_called_once_with(
                                query="Test query",
                                api_key="test-api-key",
                                search_kwargs={"count": 3}
                            )
