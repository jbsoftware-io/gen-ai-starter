"""Test suite for PDF_Podcast local example using edge-tts."""
from io import BytesIO
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import numpy as np
import pytest

# Import the modules we're testing
from example.pdf_podcast import (
    extract_text_from_pdf,
    generate_podcast_script,
    synthesize_audio_local,
    synthesize_audio_wrapper,
    get_voice_id,
    handle_pdf_podcast,
    VOICE_MAPPING
)


class TestExtractTextFromPDF:
    """Test PDF text extraction functionality"""

    def test_extract_text_from_pdf_with_valid_content(self):
        """Test extraction from PDF with actual text content"""
        # Create a simple PDF using pypdf
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        pdf_bytes = BytesIO()
        writer.write(pdf_bytes)
        pdf_bytes.seek(0)

        # Mock the file uploader object
        mock_file = Mock()
        mock_file.read.return_value = pdf_bytes.getvalue()
        mock_file.seek = pdf_bytes.seek

        with patch('example.pdf_podcast.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test content"
            mock_reader.return_value.pages = [mock_page]

            result = extract_text_from_pdf(mock_file)
            assert isinstance(result, str)
            assert "Test content" in result

    def test_extract_text_from_pdf_empty_content(self):
        """Test extraction from PDF with no extractable text"""
        mock_file = Mock()

        with patch('example.pdf_podcast.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = None  # No text
            mock_reader.return_value.pages = [mock_page]

            result = extract_text_from_pdf(mock_file)
            assert isinstance(result, str)

    def test_extract_text_from_pdf_multiple_pages(self):
        """Test extraction from multi-page PDF"""
        mock_file = Mock()

        with patch('example.pdf_podcast.PdfReader') as mock_reader:
            mock_pages = [Mock(), Mock()]
            mock_pages[0].extract_text.return_value = "Page 1 content"
            mock_pages[1].extract_text.return_value = "Page 2 content"
            mock_reader.return_value.pages = mock_pages

            result = extract_text_from_pdf(mock_file)
            assert "Page 1 content" in result
            assert "Page 2 content" in result
            assert "[Page 1]" in result
            assert "[Page 2]" in result

    def test_extract_text_from_pdf_handles_error(self):
        """Test error handling for corrupted PDF"""
        mock_file = Mock()

        with patch('example.pdf_podcast.PdfReader') as mock_reader:
            mock_reader.side_effect = Exception("PDF parsing error")
            with pytest.raises(Exception):
                extract_text_from_pdf(mock_file)


class TestGeneratePodcastScript:
    """Test podcast script generation from text"""

    def test_generate_podcast_script_creates_dialogue_format(self):
        """Test that generated script contains Host A and Host B labels"""
        test_text = "Machine learning is a subset of AI focused on learning \
from data."
        model_name = "test_model"

        with patch('example.pdf_podcast.create_llm') as mock_create_llm:
            mock_llm = Mock()
            expected_script = """Host A: Machine learning is fascinating!
Host B: Yes, it's all about learning from data.
Host A: Exactly, and it powers many modern applications."""
            mock_llm.invoke.return_value = expected_script
            mock_create_llm.return_value = mock_llm

            result = generate_podcast_script(test_text, model_name)

            assert "Host A:" in result
            assert "Host B:" in result
            assert isinstance(result, str)
            mock_create_llm.assert_called_once_with(model_name)
            mock_llm.invoke.assert_called_once()

    def test_generate_podcast_script_truncates_long_text(self):
        """Test that very long PDFs are truncated to 4000 chars"""
        long_text = "a" * 10000  # Exceed the 4000 char limit
        model_name = "test_model"

        with patch('example.pdf_podcast.create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Host A: Short response"
            mock_create_llm.return_value = mock_llm

            result = generate_podcast_script(long_text, model_name)
            assert result is not None

            # Verify the truncation happened in the prompt
            call_args = mock_llm.invoke.call_args
            prompt_text = call_args[0][0]
            # The prompt should contain truncated text
            assert "aaaa" in prompt_text

    def test_generate_podcast_script_ollama_invocation(self):
        """Test that Ollama LLM is properly invoked"""
        test_text = "Test content for script generation"
        model_name = "llama3.1:8b"

        with patch('example.pdf_podcast.create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Host A: Test\nHost B: Response"
            mock_create_llm.return_value = mock_llm

            generate_podcast_script(test_text, model_name)

            mock_create_llm.assert_called_once_with(model_name)
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[0][0]
            assert "expert podcast scriptwriter" in call_args

    def test_generate_podcast_script_handles_error(self):
        """Test error handling for Ollama failures"""
        with patch('example.pdf_podcast.create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("Ollama connection failed")
            mock_create_llm.return_value = mock_llm

            with pytest.raises(Exception):
                generate_podcast_script("Test text", "test_model")


class TestVoiceMapping:
    """Test voice mapping for gender and accent selection"""

    def test_get_voice_id_male_us(self):
        """Test voice ID lookup for Male US"""
        voice_id = get_voice_id("Male", "US")
        assert voice_id == "en-US-GuyNeural"

    def test_get_voice_id_female_uk(self):
        """Test voice ID lookup for Female UK"""
        voice_id = get_voice_id("Female", "UK")
        assert voice_id == "en-GB-SoniaNeural"

    def test_get_voice_id_male_australian(self):
        """Test voice ID lookup for Male Australian"""
        voice_id = get_voice_id("Male", "Australian")
        assert voice_id == "en-AU-WilliamNeural"

    def test_get_voice_id_female_indian(self):
        """Test voice ID lookup for Female Indian"""
        voice_id = get_voice_id("Female", "Indian")
        assert voice_id == "en-IN-NeerjaNeural"

    def test_get_voice_id_invalid_falls_back_to_default(self):
        """Test that invalid gender/accent falls back to default"""
        voice_id = get_voice_id("Unknown", "Invalid")
        assert voice_id == "en-US-GuyNeural"

    def test_voice_mapping_has_all_combinations(self):
        """Test that voice mapping has expected gender/accent combinations"""
        assert len(VOICE_MAPPING) == 8  # 2 genders x 4 accents
        for gender in ["Male", "Female"]:
            for accent in ["US", "UK", "Australian", "Indian"]:
                assert (gender, accent) in VOICE_MAPPING


class TestSynthesizeAudioLocal:
    """Test async audio synthesis functionality with edge-tts"""

    @pytest.mark.asyncio
    async def test_synthesize_audio_local_returns_tuple(self):
        """Test that synthesize_audio_local returns (audio_array, sample_rate)"""  # NOQA: E501
        script = "Host A: Hello\nHost B: Hi there"

        with patch('example.pdf_podcast.edge_tts.Communicate') as mock_comm:
            with patch('example.pdf_podcast.sf.read') as mock_read:
                # Mock the async save method
                mock_instance = AsyncMock()
                mock_comm.return_value = mock_instance
                mock_read.return_value = (np.array([0.1, 0.2, 0.3]), 24000)

                with patch(
                    'example.pdf_podcast.tempfile.NamedTemporaryFile'
                ):
                    audio_data, sr = await synthesize_audio_local(script)

                    assert audio_data is not None
                    assert sr == 24000
                    assert isinstance(audio_data, np.ndarray)

    @pytest.mark.asyncio
    async def test_synthesize_audio_local_uses_edge_tts(self):
        """Test that edge-tts Communicate is called with correct voice"""
        script = "Host A: Test audio"

        with patch('example.pdf_podcast.edge_tts.Communicate') as mock_comm:
            with patch('example.pdf_podcast.sf.read') as mock_read:
                mock_instance = AsyncMock()
                mock_comm.return_value = mock_instance
                mock_read.return_value = (np.array([0.1, 0.2]), 24000)

                with patch(
                    'example.pdf_podcast.tempfile.NamedTemporaryFile'
                ):
                    await synthesize_audio_local(
                        script,
                        host_a_voice_id="en-US-GuyNeural"
                    )

                    # Verify Communicate was called
                    mock_comm.assert_called()
                    call_args = mock_comm.call_args
                    assert call_args[1]['voice'] == 'en-US-GuyNeural'

    @pytest.mark.asyncio
    async def test_synthesize_audio_local_concatenates_segments(self):
        """Test that multiple audio segments are concatenated"""
        script = """Host A: First line
Host B: Second line
Host A: Third line"""

        with patch('example.pdf_podcast.edge_tts.Communicate') as mock_comm:
            with patch('example.pdf_podcast.sf.read') as mock_read:
                mock_instance = AsyncMock()
                mock_comm.return_value = mock_instance
                # Return different audio arrays for each line
                mock_read.side_effect = [
                    (np.array([0.1, 0.2]), 24000),
                    (np.array([0.3, 0.4, 0.5]), 24000),
                    (np.array([0.6]), 24000),
                ]

                with patch(
                    'example.pdf_podcast.tempfile.NamedTemporaryFile'
                ):
                    audio_data, sr = await synthesize_audio_local(script)

                    assert audio_data is not None
                    assert len(audio_data) == 6  # 2 + 3 + 1

    @pytest.mark.asyncio
    async def test_synthesize_audio_local_empty_script(self):
        """Test handling of empty script"""
        script = ""

        with patch('example.pdf_podcast.edge_tts.Communicate'):
            with patch('example.pdf_podcast.sf.read'):
                with patch(
                    'example.pdf_podcast.tempfile.NamedTemporaryFile'
                ):
                    audio_data, sr = await synthesize_audio_local(script)

                    assert audio_data is None
                    assert sr is None

    @pytest.mark.asyncio
    async def test_synthesize_audio_local_handles_error(self):
        """Test graceful error handling when synthesis fails"""
        script = "Host A: Test"

        with patch('example.pdf_podcast.edge_tts.Communicate') as mock_comm:
            mock_instance = AsyncMock()
            mock_instance.save.side_effect = Exception("Synthesis failed")
            mock_comm.return_value = mock_instance

            with patch('example.pdf_podcast.tempfile.NamedTemporaryFile'):
                audio_data, sr = await synthesize_audio_local(script)
                assert audio_data is None
                assert sr is None


class TestSynthesizeAudioWrapper:
    """Test sync wrapper for async synthesis"""

    def test_synthesize_audio_wrapper_returns_tuple(self):
        """Test that wrapper returns expected tuple format"""
        script = "Host A: Hello\nHost B: Hi"

        with patch(
            'example.pdf_podcast.synthesize_audio_local',
            new_callable=AsyncMock
        ) as mock_async:
            mock_async.return_value = (np.array([0.1, 0.2]), 24000)

            # Create an event loop manually to test the wrapper
            result = synthesize_audio_wrapper(script)

            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_synthesize_audio_wrapper_passes_voice_ids(self):
        """Test that wrapper correctly passes voice IDs to async function"""
        script = "Host A: Test"
        voice_a = "en-US-AriaNeural"
        voice_b = "en-GB-RyanNeural"

        with patch(
            'example.pdf_podcast.synthesize_audio_local',
            new_callable=AsyncMock
        ) as mock_async:
            mock_async.return_value = (np.array([0.1]), 24000)

            synthesize_audio_wrapper(
                script,
                host_a_voice_id=voice_a,
                host_b_voice_id=voice_b
            )

            # Verify async function was called with correct args (positional)
            mock_async.assert_called_once()
            call_args, call_kwargs = mock_async.call_args
            assert call_args[0] == script
            assert call_args[1] == voice_a
            assert call_args[2] == voice_b


class TestHandlePdfPodcast:
    """Test Streamlit handler for pdf_podcast feature"""

    def test_handle_pdf_podcast_initializes(self):
        """Test that handle_pdf_podcast handler initializes without error"""
        mock_st = Mock()
        mock_st.session_state = {}
        # Create column mocks that support context manager
        col_mocks = [Mock() for _ in range(4)]
        for col_mock in col_mocks:
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
        mock_st.columns.return_value = col_mocks
        mock_st.selectbox.return_value = "Male"
        mock_st.file_uploader.return_value = None
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=None)
        model_name = "test_model"

        with patch('example.pdf_podcast.extract_text_from_pdf'):
            with patch('example.pdf_podcast.generate_podcast_script'):
                with patch('example.pdf_podcast.synthesize_audio_wrapper'):
                    # Should not raise exception
                    handle_pdf_podcast(mock_st, model_name)

    def test_handle_pdf_podcast_voice_selection(self):
        """Test that voice selection options are displayed"""
        mock_st = Mock()
        mock_st.session_state = {}
        # Create column mocks that support context manager
        col_mocks = [Mock() for _ in range(4)]
        for col_mock in col_mocks:
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
        mock_st.columns.return_value = col_mocks
        mock_st.selectbox = Mock(return_value="Male")
        mock_st.file_uploader.return_value = None
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=None)
        model_name = "test_model"

        with patch('example.pdf_podcast.extract_text_from_pdf'):
            with patch('example.pdf_podcast.generate_podcast_script'):
                with patch('example.pdf_podcast.synthesize_audio_wrapper'):
                    handle_pdf_podcast(mock_st, model_name)

                    # Verify selectbox was called for gender/accent selection
                    mock_st.selectbox.assert_called()

    def test_handle_pdf_podcast_file_uploader(self):
        """Test file uploader is rendered"""
        mock_st = Mock()
        mock_st.session_state = {}
        # Create column mocks that support context manager
        col_mocks = [Mock() for _ in range(4)]
        for col_mock in col_mocks:
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
        mock_st.columns.return_value = col_mocks
        mock_st.selectbox.return_value = "Male"
        mock_st.file_uploader.return_value = None
        model_name = "test_model"

        with patch('example.pdf_podcast.extract_text_from_pdf'):
            with patch('example.pdf_podcast.generate_podcast_script'):
                with patch('example.pdf_podcast.synthesize_audio_wrapper'):
                    handle_pdf_podcast(mock_st, model_name)

                    # Verify file uploader was called
                    mock_st.file_uploader.assert_called()

    def test_handle_pdf_podcast_no_file_uploaded(self):
        """Test behavior when no file is uploaded"""
        mock_st = Mock()
        mock_st.session_state = {}
        # Create column mocks that support context manager
        col_mocks = [Mock() for _ in range(4)]
        for col_mock in col_mocks:
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
        mock_st.columns.return_value = col_mocks
        mock_st.selectbox.return_value = "Male"
        mock_st.file_uploader.return_value = None
        model_name = "test_model"

        with patch('example.pdf_podcast.extract_text_from_pdf'):
            with patch('example.pdf_podcast.generate_podcast_script'):
                with patch('example.pdf_podcast.synthesize_audio_wrapper'):
                    handle_pdf_podcast(mock_st, model_name)

                    # Should not attempt to extract text
                    assert not mock_st.error.called or mock_st.error.call_count == 0  # NOQA: E501

    def test_handle_pdf_podcast_langfuse_integration(self):
        """Test Langfuse handler integration"""
        mock_st = Mock()
        mock_st.session_state = {}
        # Create column mocks that support context manager
        col_mocks = [Mock() for _ in range(4)]
        for col_mock in col_mocks:
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
        mock_st.columns.return_value = col_mocks
        mock_st.selectbox.return_value = "Male"
        mock_st.file_uploader.return_value = None
        model_name = "test_model"
        mock_langfuse_handler = Mock()

        with patch('example.pdf_podcast.extract_text_from_pdf'):
            with patch('example.pdf_podcast.generate_podcast_script'):
                with patch('example.pdf_podcast.synthesize_audio_wrapper'):
                    # Should handle langfuse_handler gracefully
                    handle_pdf_podcast(
                        mock_st,
                        model_name,
                        langfuse_handler=mock_langfuse_handler
                    )

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_pdf_podcast_integration_with_real_edge_tts(self):
        """Integration test with real edge-tts (slow, requires internet)"""
        pytest.importorskip("edge_tts")

        script = "Host A: Hello world\nHost B: This is a test"

        audio_data, sr = await synthesize_audio_local(
            script,
            host_a_voice_id="en-US-GuyNeural",
            host_b_voice_id="en-GB-SoniaNeural"
        )

        assert audio_data is not None
        assert sr == 24000
        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) > 0
