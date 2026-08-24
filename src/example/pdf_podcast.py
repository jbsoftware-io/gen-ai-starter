import asyncio
import os
import tempfile
import time

import edge_tts
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
import soundfile as sf

from internal.logger import logger
from internal.util import create_llm

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


# Voice mapping: (gender, accent) -> edge-tts voice ID
VOICE_MAPPING = {
    ("Male", "US"): "en-US-GuyNeural",
    ("Female", "US"): "en-US-AriaNeural",
    ("Male", "UK"): "en-GB-RyanNeural",
    ("Female", "UK"): "en-GB-SoniaNeural",
    ("Male", "Australian"): "en-AU-WilliamNeural",
    ("Female", "Australian"): "en-AU-NatashaNeural",
    ("Male", "Indian"): "en-IN-PrabhatNeural",
    ("Female", "Indian"): "en-IN-NeerjaNeural",
}


def get_voice_id(gender: str, accent: str) -> str:
    """
    Get edge-tts voice ID from gender and accent.

    Args:
        gender: "Male" or "Female"
        accent: "US", "UK", "Australian", or "Indian"

    Returns:
        str: edge-tts voice ID (e.g., "en-US-GuyNeural")
    """
    key = (gender, accent)
    if key not in VOICE_MAPPING:
        logger.warning(
            f"Voice {key} not found, defaulting to US male"
        )
        return VOICE_MAPPING[("Male", "US")]
    return VOICE_MAPPING[key]


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from an uploaded PDF file.

    Args:
        uploaded_file: Streamlit UploadedFile object (PDF)

    Returns:
        str: Combined text from all PDF pages
    """
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
                text += f"\n[Page {page_num + 1}]\n"
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise


def generate_podcast_script(
    text: str,
    model_name: str,
    langfuse_handler=None
) -> str:
    """
    Generate a dual-speaker podcast script from source text using Ollama.

    Args:
        text: Source text to convert into dialogue
        model_name: Name of Ollama model to use
        langfuse_handler: Optional Langfuse callback handler for tracing

    Returns:
        str: Formatted podcast script with "Host A:" and "Host B:" speaker labels  # NOQA: E501
    """
    try:
        llm = create_llm(model_name)

        # Chunk text to fit within reasonable context limits
        max_chars = 16000
        chunked_text = text[:max_chars]
        if len(text) > max_chars:
            logger.warning(
                f"PDF text truncated from {len(text)} to {max_chars} chars"
            )

        prompt = (
            "You are an expert podcast scriptwriter. Convert the following "
            "source text into a highly engaging, conversational 2-person "
            "podcast script between Host A and Host B only (no other speakers, "  # NOQA: E501
            "simply discuss the topic). Format the output strictly with each "
            "speaker on a new line:\n\n"
            "Host A: [text]\n"
            "Host B: [text]\n\n"
            "Make it natural, informative, and entertaining. Alternate speakers "  # NOQA: E501
            "regularly. Try not to use asterisks or other special characters "
            "(this will be read by a text-to-speech engine). Do not include "
            "any narration or stage directions. Only output the dialogue "
            "between Host A and Host B.\n\n"
            f"Source Text:\n{chunked_text}"
        )

        logger.info("Generating podcast script via Ollama")
        config = {
            "callbacks": [langfuse_handler] if langfuse_handler else None
        }
        response = llm.invoke(prompt, config=config)
        logger.info("Podcast script generated successfully")
        return response

    except Exception as e:
        logger.error(f"Error generating podcast script: {e}")
        raise


async def synthesize_audio_local(
    script_text: str,
    host_a_voice_id: str = "en-US-GuyNeural",
    host_b_voice_id: str = "en-GB-SoniaNeural"
) -> tuple:
    """
    Synthesize audio from podcast script using Microsoft Edge TTS (edge-tts).

    Uses edge-tts API for natural-sounding synthesis with male/female voices.
    Works on CPU, no model download. Hundreds of high-quality voices available.

    Args:
        script_text: Podcast script with "Host A:" and "Host B:" labels
        host_a_voice_id: edge-tts voice ID for Host A (default: US Male)
        host_b_voice_id: edge-tts voice ID for Host B (default: UK Female)

    Returns:
        tuple: (audio_array: np.ndarray, sample_rate: int) or (None, None) on error  # NOQA: E501
    """
    try:
        logger.info("Preparing edge-tts synthesis")

        lines = script_text.split("\n")
        combined_audio = []
        sample_rate = 24000  # Standard sample rate for audio

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                if line.startswith("Host A:"):
                    text = line.replace("Host A:", "").strip()
                    if text:
                        logger.info(
                            f"Synthesizing Host A ({host_a_voice_id}): "
                            f"{text[:50]}..."
                        )
                        start_time = time.time()

                        # Use edge-tts for synthesis
                        communicate = edge_tts.Communicate(
                            text=text,
                            voice=host_a_voice_id
                        )

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix='.mp3',
                            delete=False
                        ) as tmp:
                            tmp_path = tmp.name
                            await communicate.save(tmp_path)

                        try:
                            # Read MP3 file as audio
                            audio_data, sr = sf.read(tmp_path)
                            # Resample if needed
                            if sr != sample_rate:
                                # Upsample/downsample to target rate
                                ratio = sample_rate / sr
                                new_len = int(len(audio_data) * ratio)
                                audio_data = np.interp(
                                    np.linspace(
                                        0,
                                        len(audio_data)-1,
                                        new_len
                                    ),
                                    np.arange(len(audio_data)),
                                    audio_data
                                )
                            combined_audio.append(audio_data)
                        finally:
                            os.unlink(tmp_path)

                        elapsed = time.time() - start_time
                        logger.info(
                            f"Host A synthesis completed in {elapsed:.1f}s "
                            f"({len(text)} chars)"
                        )

                elif line.startswith("Host B:"):
                    text = line.replace("Host B:", "").strip()
                    if text:
                        logger.info(
                            f"Synthesizing Host B ({host_b_voice_id}): "
                            f"{text[:50]}..."
                        )
                        start_time = time.time()

                        # Use edge-tts for synthesis
                        communicate = edge_tts.Communicate(
                            text=text,
                            voice=host_b_voice_id
                        )

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix='.mp3',
                            delete=False
                        ) as tmp:
                            tmp_path = tmp.name
                            await communicate.save(tmp_path)

                        try:
                            # Read MP3 file as audio
                            audio_data, sr = sf.read(tmp_path)
                            # Resample if needed
                            if sr != sample_rate:
                                # Upsample/downsample to target rate
                                ratio = sample_rate / sr
                                new_len = int(len(audio_data) * ratio)
                                audio_data = np.interp(
                                    np.linspace(
                                        0,
                                        len(audio_data)-1,
                                        new_len
                                    ),
                                    np.arange(len(audio_data)),
                                    audio_data
                                )
                            combined_audio.append(audio_data)
                        finally:
                            os.unlink(tmp_path)

                        elapsed = time.time() - start_time
                        logger.info(
                            f"Host B synthesis completed in {elapsed:.1f}s "
                            f"({len(text)} chars)"
                        )

            except Exception as line_error:
                logger.warning(f"Skipping line due to error: {line_error}")
                continue

        if combined_audio:
            logger.info(f"Concatenating {len(combined_audio)} audio segments")
            final_audio = np.concatenate(combined_audio)
            return final_audio, sample_rate

        logger.warning("No audio segments were generated")
        return None, None

    except Exception as e:
        logger.error(f"Error during audio synthesis: {e}")
        raise


def synthesize_audio_wrapper(
    script_text: str,
    host_a_voice_id: str = "en-US-GuyNeural",
    host_b_voice_id: str = "en-GB-SoniaNeural"
) -> tuple:
    """
    Sync wrapper for async synthesize_audio_local function.

    Bridges Streamlit's synchronous context with edge-tts's async API.

    Args:
        script_text: Podcast script with "Host A:" and "Host B:" labels
        host_a_voice_id: edge-tts voice ID for Host A
        host_b_voice_id: edge-tts voice ID for Host B

    Returns:
        tuple: (audio_array: np.ndarray, sample_rate: int) or (None, None) on error  # NOQA: E501
    """
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            synthesize_audio_local(
                script_text,
                host_a_voice_id,
                host_b_voice_id
            )
        )
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in audio synthesis wrapper: {e}")
        raise


def handle_pdf_podcast(st, model_name, langfuse_handler=None):
    """
    Streamlit UI handler for PDF to Podcast.

    Full pipeline: PDF upload → text extraction → script generation → audio synthesis.  # NOQA: E501
    Generated audio can be played inline and downloaded as WAV.

    Args:
        st: Streamlit module instance
        model_name: Name of Ollama model for script generation
        langfuse_handler: Optional Langfuse callback handler for tracing
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        host_a_gender = st.selectbox(
            "Host A Gender",
            ["Male", "Female"],
            key="host_a_gender_select"
        )

    with col2:
        host_a_accent = st.selectbox(
            "Host A Accent",
            ["US", "UK", "Australian", "Indian"],
            key="host_a_accent_select"
        )

    # Host B voice selection (gender + accent)
    with col3:
        host_b_gender = st.selectbox(
            "Host B Gender",
            ["Male", "Female"],
            index=1,
            key="host_b_gender_select"
        )

    with col4:
        host_b_accent = st.selectbox(
            "Host B Accent",
            ["US", "UK", "Australian", "Indian"],
            index=1,
            key="host_b_accent_select"
        )

    # Get voice IDs from selections
    host_a_voice_id = get_voice_id(host_a_gender, host_a_accent)
    host_b_voice_id = get_voice_id(host_b_gender, host_b_accent)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your source PDF document",
        type=["pdf"],
        help="Select a PDF file to convert into a podcast"
    )

    if uploaded_file is not None:
        st.success("PDF uploaded successfully")

        # Extract PDF text
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
            st.info(f"Extracted {len(raw_text)} characters from PDF")
        except Exception as e:
            st.error(f"Failed to extract PDF text: {e}")
            return

        # Generate podcast script
        if st.button("Generate Script"):
            with st.spinner("Generating podcast script..."):
                try:
                    script = generate_podcast_script(
                        raw_text,
                        model_name,
                        langfuse_handler=langfuse_handler
                    )
                    # Store script in session state for audio generation
                    st.session_state.generated_script = script
                    st.success("Script generated! Edit if needed, then generate audio.")  # NOQA: E501
                except Exception as e:
                    st.error(f"Script generation failed: {e}")
                    return

        # Display script editor (persists after generation)
        if "generated_script" in st.session_state:
            st.subheader("Generated Script")
            edited_script = st.text_area(
                "Review and edit the script below:",
                value=st.session_state.generated_script,
                height=250,
                key="script_editor"
            )
            # Update session state with any edits user makes
            st.session_state.generated_script = edited_script

            # Audio synthesis button
            if st.button("Generate Audio"):
                with st.spinner("Synthesizing audio..."):
                    try:
                        script_to_use = st.session_state.generated_script
                        audio_data, sr = synthesize_audio_wrapper(
                            script_to_use,
                            host_a_voice_id=host_a_voice_id,
                            host_b_voice_id=host_b_voice_id
                        )

                        if audio_data is not None and sr is not None:
                            # Save audio to session state
                            with tempfile.NamedTemporaryFile(
                                suffix=".wav",
                                delete=False
                            ) as tmp_file:
                                sf.write(
                                    tmp_file.name,
                                    audio_data,
                                    sr
                                )
                                tmp_path = tmp_file.name

                            # Read audio bytes
                            with open(tmp_path, "rb") as audio_file:
                                st.session_state.audio_bytes = audio_file.read()  # NOQA: E501

                            # Cleanup temp file
                            try:
                                os.remove(tmp_path)
                            except Exception as cleanup_error:
                                logger.warning(
                                    f"Could not clean up temp file: {cleanup_error}"  # NOQA: E501
                                )

                            st.success("Audio synthesis complete.")
                        else:
                            st.error(
                                "No audio was generated. "
                                "Check the script for Host A/Host B labels."
                            )

                    except Exception as e:
                        st.error(f"Audio synthesis failed: {e}")

        # Display audio player and download (persists after generation)
        if "audio_bytes" in st.session_state:

            # Create a temporary file for playback
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as tmp_file:
                tmp_file.write(st.session_state.audio_bytes)
                tmp_path = tmp_file.name

            # Play audio inline
            st.audio(tmp_path, format="audio/wav")

            # Download button
            st.download_button(
                label="Download Podcast Audio (WAV)",
                data=st.session_state.audio_bytes,
                file_name="podcast_output.wav",
                mime="audio/wav"
            )

            # Cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp file: {cleanup_error}")  # NOQA: E501
