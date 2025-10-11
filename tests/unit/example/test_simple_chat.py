from unittest.mock import Mock, patch

# Import the modules we're testing
from example.simple_chat import handle_simple_chat, get_session_history, store


class TestSimpleChat:
    """Test cases for simple_chat functionality"""

    def test_get_session_history_new_session(self):
        """Test creation of new session history"""
        session_id = "test_session_123"
        history = get_session_history(session_id)

        assert history is not None
        # Should create a new InMemoryChatMessageHistory
        assert session_id in store

    def test_get_session_history_existing_session(self):
        """Test retrieval of existing session history"""
        session_id = "existing_session"
        # Create initial session
        history1 = get_session_history(session_id)
        # Retrieve same session
        history2 = get_session_history(session_id)

        assert history1 is history2
        assert len(store) >= 1

    def test_handle_simple_chat_initialization(self, mock_streamlit):
        """Test simple chat handler initialization"""
        model_name = "test_model"

        # Setup streamlit session state properly
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

        # Mock external dependencies
        with patch('example.simple_chat.ChatOllama') as mock_chat_ollama:
            with patch('example.simple_chat.RunnableWithMessageHistory') as mock_runnable:  # noqa: E501

                # Mock external LangChain components
                mock_llm = Mock()
                mock_chat_ollama.return_value = mock_llm

                mock_chain = Mock()
                mock_runnable.return_value = mock_chain

                handle_simple_chat(mock_streamlit, model_name)

                # Verify external LangChain components were created
                mock_chat_ollama.assert_called_once_with(
                    temperature=0.0,
                    model=model_name,
                    base_url="http://host.docker.internal:11434",
                    streaming=True
                )
                mock_runnable.assert_called_once()

                # Verify session state was initialized
                assert "messages" in session_state
                assert "session_id" in session_state

    def test_handle_simple_chat_new_conversation(self, mock_streamlit):
        """Test handling of new conversation initialization"""

        # Mock external dependencies
        with patch('example.simple_chat.ChatOllama') as mock_chat_ollama:
            with patch('example.simple_chat.RunnableWithMessageHistory') as mock_runnable:  # noqa: E501

                # Mock external LangChain components
                mock_llm = Mock()
                mock_chat_ollama.return_value = mock_llm
                mock_chain = Mock()
                mock_runnable.return_value = mock_chain

                handle_simple_chat(mock_streamlit, "test_model")

                # Should initialize messages and session_id
                assert hasattr(mock_streamlit.session_state, 'messages')
                assert hasattr(mock_streamlit.session_state, 'session_id')
                assert len(mock_streamlit.session_state.messages) == 1
                assert mock_streamlit.session_state.messages[0]["role"] == "assistant"  # noqa: E501

    def test_handle_simple_chat_user_input(self, mock_streamlit):
        """Test handling of user input and AI response"""

        # Setup existing conversation
        mock_streamlit.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]  # noqa: E501
        mock_streamlit.session_state.session_id = "test_session"

        # Mock user input
        mock_streamlit.chat_input.return_value = "Hello, AI!"

        # Mock external dependencies
        with patch('example.simple_chat.ChatOllama') as mock_chat_ollama:
            with patch('example.simple_chat.RunnableWithMessageHistory') as mock_runnable:  # noqa: E501

                # Mock external LangChain components
                mock_llm = Mock()
                mock_chat_ollama.return_value = mock_llm

                # Mock AI response
                mock_response = Mock()
                mock_response.content = "Hello! How can I assist you today?"
                mock_chain = Mock()
                mock_chain.invoke.return_value = mock_response
                mock_runnable.return_value = mock_chain

                handle_simple_chat(mock_streamlit, "test_model")

                # Verify user message was added
                assert len(mock_streamlit.session_state.messages) == 3  # initial + user + AI  # noqa: E501
                assert mock_streamlit.session_state.messages[1]["content"] == "Hello, AI!"  # noqa: E501
                assert mock_streamlit.session_state.messages[1]["role"] == "user"  # noqa: E501
                assert mock_streamlit.session_state.messages[2]["content"] == "Hello! How can I assist you today?"  # noqa: E501
                assert mock_streamlit.session_state.messages[2]["role"] == "assistant"  # noqa: E501
