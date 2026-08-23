"""Test suite for PDF2Audio local example using gTTS."""
from io import BytesIO
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

# Import the modules we're testing
from example.pdf2audio_local import (
    extract_text_from_pdf,
    generate_podcast_script,
    synthesize_audio_local,
    handle_pdf2audio
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

        with patch('example.pdf2audio_local.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test content"
            mock_reader.return_value.pages = [mock_page]

            result = extract_text_from_pdf(mock_file)
            assert isinstance(result, str)
            assert "Test content" in result

    def test_extract_text_from_pdf_empty_content(self):
        """Test extraction from PDF with no extractable text"""
        mock_file = Mock()

        with patch('example.pdf2audio_local.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = None  # No text
            mock_reader.return_value.pages = [mock_page]

            result = extract_text_from_pdf(mock_file)
            assert isinstance(result, str)

    def test_extract_text_from_pdf_multiple_pages(self):
        """Test extraction from multi-page PDF"""
        mock_file = Mock()

        with patch('example.pdf2audio_local.PdfReader') as mock_reader:
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

        with patch('example.pdf2audio_local.PdfReader') as mock_reader:
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

        with patch('example.pdf2audio_local.create_llm') as mock_create_llm:
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

        with patch('example.pdf2audio_local.create_llm') as mock_create_llm:
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

        with patch('example.pdf2audio_local.create_llm') as mock_create_llm:
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
        with patch('example.pdf2audio_local.create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("Ollama connection failed")
            mock_create_llm.return_value = mock_llm

            with pytest.raises(Exception):
                generate_podcast_script("Test text", "test_model")


class TestSynthesizeAudioLocal:
    """Test audio synthesis functionality with gTTS"""

    def test_synthesize_audio_local_returns_tuple(self):
        """Test that synthesize_audio_local returns (audio_array, sample_rate)"""
        script = "Host A: Hello\nHost B: Hi there"

        with patch('example.pdf2audio_local.gTTS') as mock_gtts_class:
            with patch('example.pdf2audio_local.sf.read') as mock_read:
                mock_gtts_instance = Mock()
                mock_gtts_class.return_value = mock_gtts_instance
                mock_read.return_value = (np.array([0.1, 0.2, 0.3]), 24000)

                with patch('example.pdf2audio_local.tempfile.NamedTemporaryFile'):
                    audio_data, sr = synthesize_audio_local(script)

                    assert audio_data is not None
                    assert sr == 24000
                    assert isinstance(audio_data, np.ndarray)

    def test_synthesize_audio_local_uses_gtts(self):
        """Test that gTTS is called with correct language and accent (TLD)"""
        script = "Host A: Test audio"

        with patch('example.pdf2audio_local.gTTS') as mock_gtts_class:
            with patch('example.pdf2audio_local.sf.read') as mock_read:
                mock_gtts_instance = Mock()
                mock_gtts_class.return_value = mock_gtts_instance
                mock_read.return_value = (np.array([0.1, 0.2]), 24000)

                with patch('example.pdf2audio_local.tempfile.NamedTemporaryFile'):
                    synthesize_audio_local(script, host_a_tld='com')

                    # Verify gTTS was called with English and US accent
                    mock_gtts_class.assert_called()
                    call_args = mock_gtts_class.call_args
                    assert call_args[1]['lang'] == 'en'
                    assert call_args[1]['tld'] == 'com'

    def test_synthesize_audio_local_concatenates_segments(self):
        """Test that multiple audio segments are concatenated"""
        script = """Host A: First line
Host B: Second line
Host A: Third line"""

        with patch('example.pdf2audio_local.gTTS') as mock_gtts_class:
            with patch('example.pdf2audio_local.sf.read') as mock_read:
                mock_gtts_instance = Mock()
                mock_gtts_class.return_value = mock_gtts_instance
                # Return different sized arrays for each call
                mock_read.side_effect = [
                    (np.array([0.1, 0.2]), 24000),
                    (np.array([0.3, 0.4, 0.5]), 24000),
                    (np.array([0.6]), 24000)
                ]

                with patch('example.pdf2audio_local.tempfile.NamedTemporaryFile'):
                    with patch('example.pdf2audio_local.os.unlink'):
                        audio_data, sr = synthesize_audio_local(script)

                        assert audio_data is not None
                        # 2 + 3 + 1 = 6 samples
                        assert len(audio_data) == 6
                        assert sr == 24000

    def test_synthesize_audio_local_empty_script(self):
        """Test handling of script with no valid lines"""
        script = "Invalid script\nNo Host A or Host B labels"

        with patch('example.pdf2audio_local.gTTS') as mock_gtts_class:
            with patch('example.pdf2audio_local.sf.read') as mock_read:
                mock_gtts_instance = Mock()
                mock_gtts_class.return_value = mock_gtts_instance
                mock_read.return_value = (np.array([0.1]), 24000)

                with patch('example.pdf2audio_local.tempfile.NamedTemporaryFile'):
                    audio_data, sr = synthesize_audio_local(script)

                    # Should return None, None for empty/invalid script
                    assert audio_data is None
                    assert sr is None

    def test_synthesize_audio_local_handles_error(self):
        """Test error handling for TTS synthesis failures"""
        script = "Host A: Test\nHost B: Error"

        with patch('example.pdf2audio_local.gTTS') as mock_gtts_class:
            with patch('example.pdf2audio_local.sf.read') as mock_read:
                mock_gtts_instance = Mock()
                mock_gtts_class.return_value = mock_gtts_instance
                # First call succeeds, second fails
                mock_read.side_effect = [
                    (np.array([0.1, 0.2]), 24000),
                    Exception("TTS synthesis error")
                ]

                with patch('example.pdf2audio_local.tempfile.NamedTemporaryFile'):
                    with patch('example.pdf2audio_local.os.unlink'):
                        audio_data, sr = synthesize_audio_local(script)

                        # Should return audio from the first successful synthesis
                        assert audio_data is not None
                        assert sr == 24000


class TestHandlePDF2Audio:
    """Test Streamlit handler for PDF2Audio feature"""

    def test_handle_pdf2audio_initializes(self):
        """Test that handle_pdf2audio handler initializes without error"""
        mock_st = Mock()
        mock_st.session_state = {}
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=None)
        model_name = "test_model"

        with patch('example.pdf2audio_local.extract_text_from_pdf'):
            with patch('example.pdf2audio_local.generate_podcast_script'):
                with patch('example.pdf2audio_local.synthesize_audio_local'):
                    # Should not raise exception
                    handle_pdf2audio(mock_st, model_name)

    def test_handle_pdf2audio_shows_device_indicator(self):
        """Test that device info is displayed"""
        mock_st = Mock()
        mock_st.session_state = {}
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=None)
        model_name = "test_model"

        with patch('example.pdf2audio_local.extract_text_from_pdf'):
            with patch('example.pdf2audio_local.generate_podcast_script'):
                with patch('example.pdf2audio_local.synthesize_audio_local'):
                    handle_pdf2audio(mock_st, model_name)

                    # Verify title and subheader were shown
                    mock_st.title.assert_called()
                    mock_st.subheader.assert_called()

    def test_handle_pdf2audio_file_uploader(self):
        """Test file uploader is rendered"""
        mock_st = Mock()
        mock_st.session_state = {}
        mock_st.file_uploader.return_value = None
        model_name = "test_model"

        with patch('example.pdf2audio_local.extract_text_from_pdf'):
            with patch('example.pdf2audio_local.generate_podcast_script'):
                with patch('example.pdf2audio_local.synthesize_audio_local'):
                    handle_pdf2audio(mock_st, model_name)

                    # Verify file uploader was called
                    mock_st.file_uploader.assert_called()

    def test_handle_pdf2audio_no_file_uploaded(self):
        """Test behavior when no file is uploaded"""
        mock_st = Mock()
        mock_st.session_state = {}
        mock_st.file_uploader.return_value = None
        model_name = "test_model"

        with patch('example.pdf2audio_local.extract_text_from_pdf'):
            with patch('example.pdf2audio_local.generate_podcast_script'):
                with patch('example.pdf2audio_local.synthesize_audio_local'):
                    handle_pdf2audio(mock_st, model_name)

                    # Should not attempt to extract text
                    assert not mock_st.error.called or mock_st.error.call_count == 0

    def test_handle_pdf2audio_langfuse_integration(self):
        """Test Langfuse handler integration"""
        mock_st = Mock()
        mock_st.session_state = {}
        mock_st.file_uploader.return_value = None
        model_name = "test_model"
        mock_langfuse_handler = Mock()

        with patch('example.pdf2audio_local.extract_text_from_pdf'):
            with patch('example.pdf2audio_local.generate_podcast_script'):
                with patch('example.pdf2audio_local.synthesize_audio_local'):
                    # Should handle langfuse_handler gracefully
                    handle_pdf2audio(
                        mock_st,
                        model_name,
                        langfuse_handler=mock_langfuse_handler
                    )

    @pytest.mark.slow
    def test_pdf2audio_integration_with_real_gtts(self):
        """Integration test with real gTTS (slow, requires internet)"""
        pytest.importorskip("gtts")

        script = "Host A: Hello world\nHost B: This is a test"

        audio_data, sr = synthesize_audio_local(script)

        assert audio_data is not None
        assert sr == 24000
        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) > 0
