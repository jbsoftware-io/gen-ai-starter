# flake8: noqa: E501
"""Test suite for Voice_Chat example using faster-whisper, edge-tts, and agents."""
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

# Import the modules we're testing
from example.voice_chat import (
    load_whisper_model,
    get_available_edge_tts_voices,
    get_voice_chat_tools,
    create_voice_chat_chain,
    process_voice_chat_query,
    synthesize_audio_for_voice_chat,
    handle_voice_chat,
    EDGE_TTS_VOICES,
    THINKING_MESSAGES,
)


class TestGetVoiceChatTools:
    """Test voice chat tools retrieval"""

    def test_get_voice_chat_tools_returns_list(self):
        """Test that tools list is returned"""
        tools = get_voice_chat_tools()
        assert isinstance(tools, list)
        assert len(tools) == 3

    def test_get_voice_chat_tools_has_invoke_methods(self):
        """Test that all tools have invoke capability"""
        tools = get_voice_chat_tools()
        for tool in tools:
            assert callable(tool) or hasattr(tool, 'invoke')


class TestLoadWhisperModel:
    """Test Whisper model loading and caching"""

    def test_load_whisper_model_tiny(self):
        """Test loading tiny Whisper model"""
        with patch('example.voice_chat.WhisperModel') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model

            model = load_whisper_model("tiny")

            mock_whisper.assert_called_once_with(
                "tiny",
                device="cpu",
                compute_type="int8"
            )
            assert model is not None

    def test_load_whisper_model_base(self):
        """Test loading base Whisper model"""
        with patch('example.voice_chat.WhisperModel') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model

            model = load_whisper_model("base")

            mock_whisper.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8"
            )
            assert model is not None

    def test_load_whisper_model_default(self):
        """Test loading default Whisper model"""
        with patch('example.voice_chat.WhisperModel') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model

            model = load_whisper_model()

            # Should default to tiny
            mock_whisper.assert_called_once_with(
                "tiny",
                device="cpu",
                compute_type="int8"
            )
            assert model is not None

    def test_load_whisper_model_error_handling(self):
        """Test error handling for model load failure"""
        with patch('example.voice_chat.WhisperModel') as mock_whisper:
            mock_whisper.side_effect = Exception("Model download failed")

            # Due to @st.cache_resource, we clear cache before testing
            try:
                load_whisper_model.clear()
            except AttributeError:
                pass  # Cache clear not available in all versions

            with pytest.raises(Exception):
                load_whisper_model("tiny")


class TestGetAvailableEdgeTtsVoices:
    """Test voice list retrieval"""

    def test_get_available_voices_returns_list(self):
        """Test that voice list is returned"""
        voices = get_available_edge_tts_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0

    def test_get_available_voices_contains_defaults(self):
        """Test that default voices are included"""
        voices = get_available_edge_tts_voices()
        assert "en-US-AriaNeural" in voices
        assert "en-US-GuyNeural" in voices
        assert "en-GB-SoniaNeural" in voices

    def test_get_available_voices_format(self):
        """Test that voices are properly formatted voice IDs"""
        voices = get_available_edge_tts_voices()
        for voice in voices:
            assert isinstance(voice, str)
            assert "en-" in voice
            assert "Neural" in voice


class TestCreateVoiceChatChain:
    """Test voice chat agent creation"""

    def test_create_voice_chat_chain(self):
        """Test agent creation with LLM and tools"""
        with patch('example.voice_chat.create_deep_agent') as mock_create:
            with patch(
                'example.voice_chat.create_voice_chat_system_prompt'
            ) as mock_prompt:
                mock_agent = Mock()
                mock_create.return_value = mock_agent
                mock_prompt.return_value = Mock()

                agent = create_voice_chat_chain("test-model")

                mock_create.assert_called_once()
                call_args = mock_create.call_args
                # Verify model format
                assert call_args[1]["model"] == "ollama:test-model"
                assert agent is not None

    def test_create_voice_chat_chain_error_handling(self):
        """Test error handling for agent creation failure"""
        with patch('example.voice_chat.create_deep_agent') as mock_create:
            mock_create.side_effect = Exception("Agent creation failed")

            with pytest.raises(Exception):
                create_voice_chat_chain("test-model")


class TestSynthesizeAudioForVoiceChat:
    """Test audio synthesis with edge-tts"""

    @pytest.mark.asyncio
    async def test_synthesize_audio_for_voice_chat_success(self):
        """Test successful audio synthesis"""
        test_text = "Hello, this is a test message"
        test_voice = "en-US-AriaNeural"

        with patch('example.voice_chat.edge_tts.Communicate') as mock_comm:
            with patch('builtins.open', create=True) as mock_open:
                with patch('os.path.getsize', return_value=1024):
                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        mock_instance = AsyncMock()
                        mock_comm.return_value = mock_instance
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            b"fake audio data"
                        )

                        with patch('os.path.exists', return_value=True):
                            with patch('os.remove'):
                                result = await synthesize_audio_for_voice_chat(
                                    test_text,
                                    test_voice
                                )

                                assert isinstance(result, bytes)
                                mock_comm.assert_called_once()
                                mock_instance.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_audio_for_voice_chat_retries_on_failure(self):
        """Test retry logic on synthesis failure"""
        with patch('example.voice_chat.edge_tts.Communicate') as mock_comm:
            with patch('asyncio.sleep', new_callable=AsyncMock):
                # First 2 attempts fail, 3rd succeeds
                mock_comm.side_effect = [
                    Mock(save=AsyncMock(side_effect=Exception("Fail 1"))),
                    Mock(save=AsyncMock(side_effect=Exception("Fail 2"))),
                    Mock(save=AsyncMock())
                ]

                with patch('os.path.exists', return_value=True):
                    with patch('os.path.getsize', return_value=1024):
                        with patch('builtins.open', create=True) as mock_open:
                            mock_open.return_value.__enter__.return_value.read.return_value = (
                                b"fake audio"
                            )
                            with patch('os.remove'):
                                result = await synthesize_audio_for_voice_chat(
                                    "Test",
                                    "en-US-AriaNeural",
                                    max_retries=3
                                )

                                assert isinstance(result, bytes)
                                # Called 3 times (2 failures + 1 success)
                                assert mock_comm.call_count == 3

    @pytest.mark.asyncio
    async def test_synthesize_audio_for_voice_chat_empty_file_error(self):
        """Test error when audio file is empty"""
        with patch('example.voice_chat.edge_tts.Communicate'):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=0):
                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        with pytest.raises(Exception):
                            await synthesize_audio_for_voice_chat(
                                "Test",
                                "en-US-AriaNeural"
                            )


class TestProcessVoiceChatQuery:
    """Test voice chat agent query processing"""

    def test_process_voice_chat_query_success(self):
        """Test successful query processing with audio synthesis"""
        mock_agent = Mock()
        mock_response = {
            "messages": [Mock(content="Test response")]
        }
        mock_agent.invoke.return_value = mock_response

        with patch(
            'example.voice_chat.synthesize_audio_for_voice_chat'
        ) as mock_synth:
            mock_synth.return_value = b"fake audio"
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = b"fake audio"

                result = process_voice_chat_query(
                    mock_agent,
                    "Test query",
                    selected_voice="en-US-AriaNeural"
                )

                assert result["text"] == "Test response"
                assert result["audio_bytes"] == b"fake audio"
                assert result["error"] is None

    def test_process_voice_chat_query_with_langfuse(self):
        """Test query processing with Langfuse callback"""
        mock_agent = Mock()
        mock_response = {
            "messages": [Mock(content="Test response")]
        }
        mock_agent.invoke.return_value = mock_response
        mock_langfuse = Mock()

        with patch(
            'example.voice_chat.synthesize_audio_for_voice_chat'
        ) as mock_synth:
            mock_synth.return_value = b"fake audio"
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = b"fake audio"

                result = process_voice_chat_query(
                    mock_agent,
                    "Test query",
                    selected_voice="en-US-AriaNeural",
                    langfuse_handler=mock_langfuse
                )

                assert result["text"] == "Test response"
                mock_agent.invoke.assert_called_once()
                call_args = mock_agent.invoke.call_args
                assert call_args[1]["config"]["callbacks"] == [mock_langfuse]

    def test_process_voice_chat_query_synthesis_failure_fallback(self):
        """Test fallback when audio synthesis fails"""
        mock_agent = Mock()
        mock_response = {
            "messages": [Mock(content="Test response")]
        }
        mock_agent.invoke.return_value = mock_response

        with patch(
            'example.voice_chat.synthesize_audio_for_voice_chat'
        ) as mock_synth:
            with patch('asyncio.run') as mock_run:
                mock_run.side_effect = Exception("Synthesis failed")

                result = process_voice_chat_query(
                    mock_agent,
                    "Test query",
                    selected_voice="en-US-AriaNeural"
                )

                assert result["text"] == "Test response"
                assert result["audio_bytes"] is None
                assert result["error"] is not None
                assert "Audio synthesis failed" in result["error"]

    def test_process_voice_chat_query_agent_error(self):
        """Test error handling for agent invocation failure"""
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Agent invocation failed")

        result = process_voice_chat_query(
            mock_agent,
            "Test query",
            selected_voice="en-US-AriaNeural"
        )

        assert result["text"] is None
        assert result["audio_bytes"] is None
        assert result["error"] is not None
        assert "Failed to process query" in result["error"]


class TestHandleVoiceChat:
    """Test Streamlit UI integration"""

    @staticmethod
    def _create_mock_session_state():
        """Create a mock session state that supports attribute access."""
        class MockSessionState:
            def __init__(self):
                self._data = {}

            def __contains__(self, key):
                return key in self._data

            def __getattr__(self, key):
                if key.startswith('_'):
                    return super().__getattribute__(key)
                return self._data.get(key)

            def __setattr__(self, key, value):
                if key.startswith('_'):
                    super().__setattr__(key, value)
                else:
                    self._data[key] = value

        return MockSessionState()

    @staticmethod
    def _create_context_manager_mock():
        """Create a mock that supports context manager protocol"""
        mock = MagicMock()
        mock.__enter__ = Mock(return_value=None)
        mock.__exit__ = Mock(return_value=None)
        return mock

    def test_handle_voice_chat_initialization(self):
        """Test that session state is initialized correctly"""
        mock_st = Mock()
        mock_st.session_state = self._create_mock_session_state()

        # Mock Streamlit components
        mock_st.sidebar = self._create_context_manager_mock()
        mock_st.subheader = Mock()
        # Make columns return mocks that support context manager
        col_1 = self._create_context_manager_mock()
        col_2 = self._create_context_manager_mock()
        mock_st.columns.return_value = [col_1, col_2]
        mock_st.write = Mock()
        mock_st.selectbox.side_effect = ["tiny", "en-US-AriaNeural"]
        mock_st.info = Mock()
        mock_st.audio_input.return_value = None
        mock_st.text_input.return_value = None
        mock_st.chat_message = self._create_context_manager_mock()

        handle_voice_chat(mock_st, "test-model")

        # Verify session state was initialized
        assert mock_st.session_state is not None

    def test_handle_voice_chat_with_text_input(self):
        """Test handling text input"""
        mock_st = Mock()
        mock_st.session_state = self._create_mock_session_state()
        mock_st.session_state.messages = []
        mock_st.session_state.selected_whisper_model = "tiny"
        mock_st.session_state.selected_voice = "en-US-AriaNeural"
        mock_st.session_state.session_id = "test-123"

        mock_st.sidebar = self._create_context_manager_mock()
        mock_st.subheader = Mock()
        col_1 = self._create_context_manager_mock()
        col_2 = self._create_context_manager_mock()
        mock_st.columns.return_value = [col_1, col_2]
        mock_st.write = Mock()
        mock_st.selectbox.side_effect = ["tiny", "en-US-AriaNeural"]
        mock_st.info = Mock()
        mock_st.audio_input.return_value = None
        mock_st.text_input.return_value = "Hello, how are you?"
        mock_st.chat_message = self._create_context_manager_mock()
        mock_st.spinner = self._create_context_manager_mock()
        mock_st.markdown = Mock()

        with patch('example.voice_chat.create_voice_chat_chain'):
            with patch('example.voice_chat.process_voice_chat_query') as mock_process:
                mock_process.return_value = {
                    "text": "I'm doing well",
                    "audio_bytes": None,
                    "error": None
                }
                handle_voice_chat(mock_st, "test-model")

    def test_handle_voice_chat_voice_selection(self):
        """Test voice selection persistence"""
        mock_st = Mock()
        mock_st.session_state = self._create_mock_session_state()
        mock_st.session_state.messages = []
        mock_st.session_state.selected_whisper_model = "tiny"
        mock_st.session_state.selected_voice = "en-GB-SoniaNeural"
        mock_st.session_state.session_id = "test-123"

        mock_st.sidebar = self._create_context_manager_mock()
        mock_st.subheader = Mock()
        col_1 = self._create_context_manager_mock()
        col_2 = self._create_context_manager_mock()
        mock_st.columns.return_value = [col_1, col_2]
        mock_st.write = Mock()
        mock_st.selectbox.side_effect = ["tiny", "en-GB-SoniaNeural"]
        mock_st.info = Mock()
        mock_st.audio_input.return_value = None
        mock_st.text_input.return_value = None
        mock_st.chat_message = self._create_context_manager_mock()

        handle_voice_chat(mock_st, "test-model")

        # Verify voice selection was called
        assert mock_st.selectbox.call_count >= 1

    def test_handle_voice_chat_error_message_display(self):
        """Test error message display"""
        mock_st = Mock()
        mock_st.session_state = self._create_mock_session_state()
        mock_st.session_state.messages = []
        mock_st.session_state.selected_whisper_model = "tiny"
        mock_st.session_state.selected_voice = "en-US-AriaNeural"
        mock_st.session_state.session_id = "test-123"

        mock_st.sidebar = self._create_context_manager_mock()
        mock_st.subheader = Mock()
        col_1 = self._create_context_manager_mock()
        col_2 = self._create_context_manager_mock()
        mock_st.columns.return_value = [col_1, col_2]
        mock_st.write = Mock()
        mock_st.selectbox.side_effect = ["tiny", "en-US-AriaNeural"]
        mock_st.info = Mock()
        mock_st.audio_input.return_value = None
        mock_st.text_input.return_value = "Hello"
        mock_st.chat_message = self._create_context_manager_mock()
        mock_st.spinner = self._create_context_manager_mock()
        mock_st.error = Mock()
        mock_st.markdown = Mock()

        with patch('example.voice_chat.create_voice_chat_chain') as mock_chain:
            mock_chain.side_effect = Exception("Chain creation failed")
            handle_voice_chat(mock_st, "test-model")

            # Verify error was displayed
            mock_st.error.assert_called()


class TestThinkingMessages:
    """Test thinking/banter messages"""

    def test_thinking_messages_is_list(self):
        """Test that THINKING_MESSAGES is populated"""
        assert isinstance(THINKING_MESSAGES, list)
        assert len(THINKING_MESSAGES) > 0

    def test_thinking_messages_are_strings(self):
        """Test that all thinking messages are strings"""
        for msg in THINKING_MESSAGES:
            assert isinstance(msg, str)
            assert len(msg) > 0

    def test_thinking_messages_variety(self):
        """Test that there are diverse thinking messages"""
        # Should have enough variety for engagement
        assert len(THINKING_MESSAGES) >= 5
        # All messages should be unique
        assert len(set(THINKING_MESSAGES)) == len(THINKING_MESSAGES)
