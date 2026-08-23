import os
import tempfile
import time

import numpy as np
from dotenv import load_dotenv
from gtts import gTTS
from pypdf import PdfReader
import soundfile as sf

from internal.logger import logger
from internal.util import create_llm

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


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


def generate_podcast_script(text: str, model_name: str) -> str:
    """
    Generate a dual-speaker podcast script from source text using Ollama.

    Args:
        text: Source text to convert into dialogue
        model_name: Name of Ollama model to use

    Returns:
        str: Formatted podcast script with "Host A:" and "Host B:" speaker labels
    """
    try:
        llm = create_llm(model_name)

        # Chunk text to fit within reasonable context limits
        max_chars = 4000
        chunked_text = text[:max_chars]
        if len(text) > max_chars:
            logger.warning(
                f"PDF text truncated from {len(text)} to {max_chars} chars"
            )

        prompt = f"""You are an expert podcast scriptwriter. Convert the following \
source text into a highly engaging, conversational 2-person podcast script between \
Host A and Host B. Format the output strictly with each speaker on a new line:

Host A: [text]
Host B: [text]

Make it natural, informative, and entertaining. Alternate speakers regularly.

Source Text:
{chunked_text}"""

        logger.info("Generating podcast script via Ollama")
        response = llm.invoke(prompt)
        logger.info("Podcast script generated successfully")
        return response

    except Exception as e:
        logger.error(f"Error generating podcast script: {e}")
        raise


def synthesize_audio_local(
    script_text: str,
    host_a_tld: str = 'com',
    host_b_tld: str = 'com'
) -> tuple:
    """
    Synthesize audio from podcast script using Google Text-to-Speech (gTTS).

    Uses gTTS API for natural-sounding synthesis. Works on CPU, no model download.
    Fast and lightweight for educational use. Supports different English accents via TLD.

    Args:
        script_text: Podcast script with "Host A:" and "Host B:" labels
        host_a_tld: TLD for Host A accent (default: 'com' = US, options: 'com', 'co.uk', 'com.au', 'co.in')
        host_b_tld: TLD for Host B accent (default: 'co.uk' = US, options: 'com', 'co.uk', 'com.au', 'co.in')

    Returns:
        tuple: (audio_array: np.ndarray, sample_rate: int) or (None, None) on error
    """
    try:
        logger.info("Preparing gTTS synthesis (no model loading needed)")

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
                        logger.info(f"Synthesizing Host A ({host_a_tld}): {text[:50]}...")
                        start_time = time.time()

                        # Use gTTS for synthesis with specified accent (via TLD)
                        tts = gTTS(text=text, lang='en', tld=host_a_tld, slow=False)

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix='.mp3',
                            delete=False
                        ) as tmp:
                            tmp_path = tmp.name
                            tts.save(tmp_path)

                        try:
                            # Read MP3 file as audio
                            audio_data, sr = sf.read(tmp_path)
                            # Resample if needed
                            if sr != sample_rate:
                                # Upsample/downsample to target rate
                                ratio = sample_rate / sr
                                new_len = int(len(audio_data) * ratio)
                                audio_data = np.interp(
                                    np.linspace(0, len(audio_data)-1, new_len),
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
                        logger.info(f"Synthesizing Host B ({host_b_tld}): {text[:50]}...")
                        start_time = time.time()

                        # Use gTTS for synthesis with specified accent (via TLD)
                        tts = gTTS(text=text, lang='en', tld=host_b_tld, slow=False)

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(
                            suffix='.mp3',
                            delete=False
                        ) as tmp:
                            tmp_path = tmp.name
                            tts.save(tmp_path)

                        try:
                            # Read MP3 file as audio
                            audio_data, sr = sf.read(tmp_path)
                            # Resample if needed
                            if sr != sample_rate:
                                # Upsample/downsample to target rate
                                ratio = sample_rate / sr
                                new_len = int(len(audio_data) * ratio)
                                audio_data = np.interp(
                                    np.linspace(0, len(audio_data)-1, new_len),
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


def handle_pdf2audio(st, model_name, langfuse_handler=None):
    """
    Streamlit UI handler for PDF2Audio NotebookLM clone.

    Full pipeline: PDF upload → text extraction → script generation → audio synthesis.
    Generated audio can be played inline and downloaded as WAV.

    Args:
        st: Streamlit module instance
        model_name: Name of Ollama model for script generation
        langfuse_handler: Optional Langfuse callback handler for tracing
    """
    st.title("🎙️ Local PDF2Audio NotebookLM Clone")
    st.subheader("Transform PDFs into interactive podcasts fully offline")

    # Voice/Accent selection (English with different accents)
    col1, col2 = st.columns(2)
    with col1:
        host_a_tld = st.selectbox(
            "Host A Accent",
            [
                ("🇺🇸 United States", "com"),
                ("🇬🇧 United Kingdom", "co.uk"),
                ("🇦🇺 Australian", "com.au"),
                ("🇮🇳 Indian", "co.in"),
            ],
            format_func=lambda x: x[0],
            key="host_a_accent_select"
        )[1]
    with col2:
        host_b_tld = st.selectbox(
            "Host B Accent",
            [
                ("🇺🇸 United States", "com"),
                ("🇬🇧 United Kingdom", "co.uk"),
                ("🇦🇺 Australian", "com.au"),
                ("🇮🇳 Indian", "co.in"),
            ],
            format_func=lambda x: x[0],
            key="host_b_accent_select",
            index=1
        )[1]  # Index 1 = UK (co.uk)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your source PDF document",
        type=["pdf"],
        help="Select a PDF file to convert into a podcast"
    )

    if uploaded_file is not None:
        st.success("✅ PDF uploaded successfully!")

        # Extract PDF text
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
            st.info(f"Extracted {len(raw_text)} characters from PDF")
        except Exception as e:
            st.error(f"Failed to extract PDF text: {e}")
            return

        # Generate podcast script
        if st.button("🎬 Generate Podcast Script"):
            with st.spinner(
                "🔄 Generating dual-speaker script using Ollama..."
            ):
                try:
                    script = generate_podcast_script(raw_text, model_name)

                    # Store script in session state for audio generation
                    st.session_state.generated_script = script

                    st.subheader("📝 Generated Show Script")
                    st.text_area(
                        "Review the script below (edit if needed):",
                        value=script,
                        height=250,
                        key="script_display"
                    )
                    st.success("✅ Script generated! Scroll down to generate audio.")

                except Exception as e:
                    st.error(f"Script generation failed: {e}")
                    return

        # Audio synthesis
        if "generated_script" in st.session_state:
            if st.button("🎵 Generate Audio"):
                with st.spinner(
                    "🎙️ Synthesizing audio locally (this may take 1-5 minutes)..."
                ):
                    try:
                        script_to_use = st.session_state.generated_script
                        audio_data, sr = synthesize_audio_local(
                            script_to_use,
                            host_a_tld=host_a_tld,
                            host_b_tld=host_b_tld
                        )

                        if audio_data is not None and sr is not None:
                            st.subheader("🎧 Listen to Your Podcast")

                            # Save to temporary WAV file
                            with tempfile.NamedTemporaryFile(
                                suffix=".wav",
                                delete=False
                            ) as tmp_file:
                                import soundfile as sf
                                sf.write(
                                    tmp_file.name,
                                    audio_data,
                                    sr
                                )
                                tmp_path = tmp_file.name

                            # Play audio inline
                            st.audio(tmp_path, format="audio/wav")

                            # Download button
                            with open(tmp_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()

                            st.download_button(
                                label="📥 Download Podcast Audio (WAV)",
                                data=audio_bytes,
                                file_name="podcast_output.wav",
                                mime="audio/wav"
                            )

                            # Cleanup temp file
                            try:
                                os.remove(tmp_path)
                            except Exception as cleanup_error:
                                logger.warning(
                                    f"Could not clean up temp file: \
{cleanup_error}"
                                )

                            st.success(
                                "✅ Audio synthesis complete! \
Download or listen above."
                            )
                        else:
                            st.error(
                                "No audio was generated. \
Check the script for Host A/Host B labels."
                            )

                    except Exception as e:
                        st.error(f"Audio synthesis failed: {e}")
