from unittest.mock import Mock, patch

# Import the modules we're testing
from example.agentic_chat import (
    handle_agentic_chat, get_tools, get_wikipedia_search_tool
)


class TestAgenticChat:
    """Test cases for agentic chat functionality"""

    @patch('example.agentic_chat.load_tools')
    def test_get_tools(self, mock_load_tools):
        """Test tools loading"""
        mock_arxiv_tool = Mock()
        mock_load_tools.return_value = [mock_arxiv_tool]

        with patch('example.agentic_chat.WikipediaQueryRun') as mock_query_run:
            with patch('example.agentic_chat.WikipediaAPIWrapper') as mock_wrapper:  # noqa: E501
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
                assert len(tools) == 2
                assert mock_arxiv_tool in tools
                assert mock_wiki_instance in tools

    @patch('example.agentic_chat.WikipediaQueryRun')
    @patch('example.agentic_chat.WikipediaAPIWrapper')
    def test_get_wikipedia_search_tool(self, mock_wrapper, mock_query_run):
        """Test Wikipedia search tool creation"""
        mock_api_wrapper = Mock()
        mock_wrapper.return_value = mock_api_wrapper

        mock_wiki_tool = Mock()
        mock_query_run.return_value = mock_wiki_tool

        tool = get_wikipedia_search_tool(top_k_results=2, doc_content_chars_max=1000)  # noqa: E501

        mock_wrapper.assert_called_once_with(
            top_k_results=2,
            doc_content_chars_max=1000
        )
        mock_query_run.assert_called_once_with(api_wrapper=mock_api_wrapper)
        assert tool == mock_wiki_tool

    def test_handle_agentic_chat_initialization(self, mock_streamlit):
        """Test agentic chat handler initialization"""
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

        # Mock external LangChain dependencies
        with patch('example.agentic_chat.Ollama') as mock_ollama:
            with patch('example.agentic_chat.load_tools') as mock_load_tools:
                with patch('example.agentic_chat.WikipediaAPIWrapper') as mock_wrapper:  # noqa: E501
                    with patch('example.agentic_chat.WikipediaQueryRun') as mock_query_run:  # noqa: E501
                        with patch('example.agentic_chat.create_react_agent') as mock_create_agent:  # noqa: E501
                            with patch('example.agentic_chat.AgentExecutor') as mock_agent_executor:  # noqa: E501

                                # Setup mocks
                                mock_llm = Mock()
                                mock_ollama.return_value = mock_llm

                                mock_load_tools.return_value = [Mock()]
                                mock_wrapper.return_value = Mock()
                                mock_query_run.return_value = Mock()

                                mock_agent = Mock()
                                mock_create_agent.return_value = mock_agent

                                mock_executor = Mock()
                                mock_agent_executor.return_value = mock_executor  # noqa: E501

                                handle_agentic_chat(mock_streamlit, model_name)  # noqa: E501

                                # Verify external components were created
                                mock_ollama.assert_called_once_with(
                                    model=model_name,
                                    base_url="http://host.docker.internal:11434"  # noqa: E501
                                )
                                mock_load_tools.assert_called_once_with(["arxiv"])  # noqa: E501
                                mock_create_agent.assert_called_once()
                                mock_agent_executor.assert_called_once()

                                # Verify session state was initialized
                                assert "messages" in session_state
                                assert "session_id" in session_state
                                assert len(session_state.messages) == 1
                                assert session_state.messages[0]["role"] == "assistant"  # noqa: E501

    def test_handle_agentic_chat_user_input(self, mock_streamlit):
        """Test agentic chat handler with user input"""

        # Setup existing conversation
        mock_streamlit.session_state.messages = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
        mock_streamlit.session_state.session_id = "test_session"

        # Mock user input
        mock_streamlit.chat_input.return_value = "What is quantum computing?"

        # Mock external LangChain dependencies
        with patch('example.agentic_chat.Ollama') as mock_ollama:
            with patch('example.agentic_chat.load_tools') as mock_load_tools:
                with patch('example.agentic_chat.WikipediaAPIWrapper') as mock_wrapper:  # noqa: E501
                    with patch('example.agentic_chat.WikipediaQueryRun') as mock_query_run:  # noqa: E501
                        with patch('example.agentic_chat.create_react_agent') as mock_create_agent:  # noqa: E501
                            with patch('example.agentic_chat.AgentExecutor') as mock_agent_executor:  # noqa: E501

                                # Setup mocks
                                mock_ollama.return_value = Mock()
                                mock_load_tools.return_value = [Mock()]
                                mock_wrapper.return_value = Mock()
                                mock_query_run.return_value = Mock()
                                mock_create_agent.return_value = Mock()

                                mock_executor = Mock()
                                mock_response = {
                                    'output': 'Quantum computing is...',
                                    'intermediate_steps': [('tool1', 'result1')]  # noqa: E501
                                }
                                mock_executor.invoke.return_value = mock_response  # noqa: E501
                                mock_agent_executor.return_value = mock_executor  # noqa: E501

                                handle_agentic_chat(mock_streamlit, "test_model")  # noqa: E501

                                # Verify user message was processed
                                assert len(mock_streamlit.session_state.messages) == 3  # initial + user + AI  # noqa: E501
                                assert mock_streamlit.session_state.messages[1]["content"] == "What is quantum computing?"  # noqa: E501
                                assert mock_streamlit.session_state.messages[1]["role"] == "user"  # noqa: E501
                                assert mock_streamlit.session_state.messages[2]["content"] == "Quantum computing is..."  # noqa: E501
                                assert mock_streamlit.session_state.messages[2]["role"] == "assistant"  # noqa: E501

                                # Verify agent executor was invoked
                                mock_executor.invoke.assert_called_once_with({
                                    "input": "What is quantum computing?",
                                    "chat_history": mock_streamlit.session_state.messages,  # noqa: E501
                                }, config={
                                    "callbacks": None
                                })
