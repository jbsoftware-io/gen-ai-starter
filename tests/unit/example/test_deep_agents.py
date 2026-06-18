from unittest.mock import Mock, patch

# Import the modules we're testing
from example.deep_agents import (
    get_tools,
    get_wikipedia_search_tool,
    create_deep_agents_chain,
    process_deep_agents_query,
    handle_deep_agents,
)


class TestDeepAgents:
    """Test cases for deep agents functionality"""

    @patch('example.deep_agents.BRAVE_SEARCH_API_KEY', None)
    @patch('example.deep_agents.load_tools')
    def test_get_tools_without_brave(self, mock_load_tools):
        """Test tools loading without Brave Search API key"""
        mock_arxiv_tool = Mock()
        mock_load_tools.return_value = [mock_arxiv_tool]

        with patch('example.deep_agents.WikipediaQueryRun') as mock_query_run:
            with patch('example.deep_agents.WikipediaAPIWrapper') as mock_wrapper:  # noqa: E501
                mock_wiki_instance = Mock()
                mock_query_run.return_value = mock_wiki_instance
                mock_wrapper.return_value = Mock()

                tools = get_tools()

                mock_load_tools.assert_called_once_with(["arxiv"])
                mock_wrapper.assert_called_once_with(
                    top_k_results=1,
                    doc_content_chars_max=500
                )
                mock_query_run.assert_called_once()
                # Should have arxiv + wikipedia (no brave)
                assert len(tools) == 2
                assert mock_arxiv_tool in tools
                assert mock_wiki_instance in tools

    @patch('example.deep_agents.load_tools')
    @patch('example.deep_agents.BraveSearch')
    def test_get_tools_with_brave(self, mock_brave_search, mock_load_tools):
        """Test tools loading with Brave Search API key"""
        mock_arxiv_tool = Mock()
        mock_load_tools.return_value = [mock_arxiv_tool]

        with patch('example.deep_agents.WikipediaQueryRun') as mock_query_run:
            with patch(
                'example.deep_agents.WikipediaAPIWrapper'
            ) as mock_wrapper:  # noqa: E501
                with patch.dict(
                    'os.environ',
                    {'BRAVE_SEARCH_API_KEY': 'test-key'}
                ):  # noqa: E501
                    mock_wiki_instance = Mock()
                    mock_query_run.return_value = mock_wiki_instance
                    mock_wrapper.return_value = Mock()

                    mock_brave_instance = Mock()
                    mock_brave_search.return_value = mock_brave_instance

                    # Temporarily set BRAVE_SEARCH_API_KEY in the module
                    import example.deep_agents
                    original_brave_key = (
                        example.deep_agents.BRAVE_SEARCH_API_KEY
                    )
                    example.deep_agents.BRAVE_SEARCH_API_KEY = 'test-key'

                    tools = get_tools()

                    # Restore original value
                    example.deep_agents.BRAVE_SEARCH_API_KEY = (
                        original_brave_key
                    )

                    mock_load_tools.assert_called_once_with(["arxiv"])
                    # Should have arxiv + wikipedia + brave
                    assert len(tools) == 3
                    assert mock_arxiv_tool in tools
                    assert mock_wiki_instance in tools
                    assert mock_brave_instance in tools

    @patch('example.deep_agents.WikipediaQueryRun')
    @patch('example.deep_agents.WikipediaAPIWrapper')
    def test_get_wikipedia_search_tool(self, mock_wrapper, mock_query_run):
        """Test Wikipedia search tool creation"""
        mock_api_wrapper = Mock()
        mock_wrapper.return_value = mock_api_wrapper

        mock_wiki_tool = Mock()
        mock_query_run.return_value = mock_wiki_tool

        tool = get_wikipedia_search_tool(
            top_k_results=2,
            doc_content_chars_max=1000
        )

        mock_wrapper.assert_called_once_with(
            top_k_results=2,
            doc_content_chars_max=1000
        )
        mock_query_run.assert_called_once_with(api_wrapper=mock_api_wrapper)
        assert tool == mock_wiki_tool

    @patch('example.deep_agents.create_deep_agent')
    @patch('example.deep_agents.get_tools')
    @patch('example.deep_agents.create_deep_agents_system_prompt')
    def test_create_deep_agents_chain(
        self,
        mock_prompt,
        mock_get_tools,
        mock_create_deep_agent
    ):
        """Test deep agents chain creation"""
        mock_tools = [Mock(), Mock()]
        mock_get_tools.return_value = mock_tools

        mock_system_prompt = "Test system prompt"
        mock_prompt.return_value = mock_system_prompt

        mock_agent = Mock()
        mock_create_deep_agent.return_value = mock_agent

        agent = create_deep_agents_chain("test-model")

        mock_prompt.assert_called_once()
        mock_get_tools.assert_called_once()
        mock_create_deep_agent.assert_called_once_with(
            model="ollama:test-model",
            tools=mock_tools,
            system_prompt=mock_system_prompt
        )
        assert agent == mock_agent

    def test_process_deep_agents_query_without_langfuse(self):
        """Test processing a query without Langfuse callback"""
        mock_agent = Mock()
        mock_response = {
            "messages": [Mock(content="Test response")]
        }
        mock_agent.invoke.return_value = mock_response

        result = process_deep_agents_query(mock_agent, "Test query")

        mock_agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Test query"}]},
            config={"callbacks": None}
        )
        assert result == mock_response

    def test_process_deep_agents_query_with_langfuse(self):
        """Test processing a query with Langfuse callback"""
        mock_agent = Mock()
        mock_response = {
            "messages": [Mock(content="Test response")]
        }
        mock_agent.invoke.return_value = mock_response

        mock_langfuse_handler = Mock()

        result = process_deep_agents_query(
            mock_agent,
            "Test query",
            langfuse_handler=mock_langfuse_handler
        )

        mock_agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Test query"}]},
            config={"callbacks": [mock_langfuse_handler]}
        )
        assert result == mock_response

    def test_handle_deep_agents_initialization(self, mock_streamlit):
        """Test deep agents handler initialization"""
        model_name = "test_model"

        # Setup streamlit session state (new conversation)
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

        session_state = MockSessionState()
        mock_streamlit.session_state = session_state

        # Mock no user input
        mock_streamlit.chat_input.return_value = None

        # Mock external dependencies
        with patch('example.deep_agents.create_deep_agents_chain') as mock_create_chain:  # noqa: E501
            with patch('example.deep_agents.load_tools') as mock_load_tools:
                with patch('example.deep_agents.WikipediaAPIWrapper') as mock_wrapper:  # noqa: E501
                    with patch('example.deep_agents.WikipediaQueryRun') as mock_query_run:  # noqa: E501
                        # Setup mocks
                        mock_load_tools.return_value = [Mock()]
                        mock_wrapper.return_value = Mock()
                        mock_query_run.return_value = Mock()

                        mock_agent = Mock()
                        mock_create_chain.return_value = mock_agent

                        handle_deep_agents(mock_streamlit, model_name)

                        # Verify session state was initialized
                        assert "messages" in session_state
                        assert "session_id" in session_state
                        assert len(session_state.messages) == 1
                        assert session_state.messages[0]["role"] == "assistant"

    def test_handle_deep_agents_user_input(self, mock_streamlit):
        """Test deep agents handler with user input"""

        # Setup existing conversation
        mock_streamlit.session_state.messages = [
            {
                "role": "assistant",
                "content": "How can I help you with research today?"
            }
        ]
        mock_streamlit.session_state.session_id = "test_session"

        # Mock user input
        mock_streamlit.chat_input.return_value = "What is machine learning?"

        # Mock external dependencies
        with patch(
            'example.deep_agents.create_deep_agents_chain'
        ) as mock_create_chain:  # noqa: E501
            with patch(
                'example.deep_agents.process_deep_agents_query'
            ) as mock_process:  # noqa: E501
                with patch(
                    'example.deep_agents.load_tools'
                ) as mock_load_tools:
                    with patch(
                        'example.deep_agents.WikipediaAPIWrapper'
                    ) as mock_wrapper:  # noqa: E501
                        with patch(
                            'example.deep_agents.WikipediaQueryRun'
                        ) as mock_query_run:  # noqa: E501

                            # Setup mocks
                            mock_load_tools.return_value = [Mock()]
                            mock_wrapper.return_value = Mock()
                            mock_query_run.return_value = Mock()

                            mock_agent = Mock()
                            mock_create_chain.return_value = mock_agent

                            mock_response = {
                                "messages": [
                                    Mock(
                                        content=(
                                            "Machine learning is a subset "
                                            "of AI..."
                                        )
                                    )
                                ]
                            }
                            mock_process.return_value = mock_response

                            handle_deep_agents(
                                mock_streamlit,
                                "test_model",
                                langfuse_handler=None
                            )

                            # Verify user message was processed
                            messages = mock_streamlit.session_state.messages
                            assert len(messages) == 3
                            assert messages[1]["content"] == (
                                "What is machine learning?"
                            )
                            assert messages[1]["role"] == "user"
                            assert (
                                "Machine learning" in messages[2]["content"]
                            )
                            assert messages[2]["role"] == "assistant"

                            # Verify process was called
                            mock_process.assert_called_once()

    def test_handle_deep_agents_with_langfuse(self, mock_streamlit):
        """Test deep agents handler with Langfuse callback"""

        # Setup existing conversation
        mock_streamlit.session_state.messages = [
            {
                "role": "assistant",
                "content": "How can I help you with research today?"
            }
        ]
        mock_streamlit.session_state.session_id = "test_session"

        # Mock user input
        mock_streamlit.chat_input.return_value = "Test query"

        mock_langfuse_handler = Mock()

        with patch(
            'example.deep_agents.create_deep_agents_chain'
        ) as mock_create_chain:  # noqa: E501
            with patch(
                'example.deep_agents.process_deep_agents_query'
            ) as mock_process:  # noqa: E501
                with patch(
                    'example.deep_agents.load_tools'
                ) as mock_load_tools:
                    with patch(
                        'example.deep_agents.WikipediaAPIWrapper'
                    ) as mock_wrapper:  # noqa: E501
                        with patch(
                            'example.deep_agents.WikipediaQueryRun'
                        ) as mock_query_run:  # noqa: E501

                            # Setup mocks
                            mock_load_tools.return_value = [Mock()]
                            mock_wrapper.return_value = Mock()
                            mock_query_run.return_value = Mock()

                            mock_agent = Mock()
                            mock_create_chain.return_value = mock_agent

                            mock_response = {
                                "messages": [
                                    Mock(content="Response message")
                                ]
                            }
                            mock_process.return_value = mock_response

                            handle_deep_agents(
                                mock_streamlit,
                                "test_model",
                                langfuse_handler=mock_langfuse_handler
                            )

                            # Verify process was called with langfuse handler
                            call_args = mock_process.call_args
                            assert (
                                call_args[1]["langfuse_handler"] ==
                                mock_langfuse_handler
                            )  # noqa: E501

    def test_handle_deep_agents_error_handling(self, mock_streamlit):
        """Test deep agents handler error handling"""

        # Setup existing conversation
        mock_streamlit.session_state.messages = [
            {
                "role": "assistant",
                "content": "How can I help you with research today?"
            }
        ]
        mock_streamlit.session_state.session_id = "test_session"

        # Mock user input
        mock_streamlit.chat_input.return_value = "Test query"

        with patch('example.deep_agents.create_deep_agents_chain') as mock_create_chain:  # noqa: E501
            with patch('example.deep_agents.load_tools') as mock_load_tools:
                with patch('example.deep_agents.WikipediaAPIWrapper') as mock_wrapper:  # noqa: E501
                    with patch('example.deep_agents.WikipediaQueryRun') as mock_query_run:  # noqa: E501

                        # Setup mocks
                        mock_load_tools.return_value = [Mock()]
                        mock_wrapper.return_value = Mock()
                        mock_query_run.return_value = Mock()

                        # Make chain creation raise an error
                        mock_create_chain.side_effect = Exception("Test error")

                        handle_deep_agents(mock_streamlit, "test_model")

                        # Verify error was caught and st.error was called
                        mock_streamlit.error.assert_called_once()
                        error_message = mock_streamlit.error.call_args[0][0]
                        assert "Test error" in error_message
