# flake8: noqa: E501
import asyncio
import os
import random
import tempfile
import time
import uuid

import edge_tts
import streamlit as st
from deepagents import create_deep_agent
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from internal.brave_client import brave_search
from internal.custom_retrievers import arxiv_query_run, wikipedia_query_run
from internal.logger import logger
from internal.prompts import create_voice_chat_system_prompt
from internal.util import create_llm

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"


# Random banter messages to show while thinking/processing
THINKING_MESSAGES = [
    "Let me look that up for you...",
    "Searching my knowledge base...",
    "One moment, gathering information...",
    "Thinking about that...",
    "Let me research that quickly...",
    "Looking into it...",
    "Checking the sources...",
    "Let me find the latest on that...",
    "Searching high and low...",
    "Consulting my research tools...",
]


# Popular edge-tts voices for voice selection
EDGE_TTS_VOICES = [
    # Working US voices (tested and verified)
    "en-US-GuyNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-US-MichelleNeural",
    # Working UK voices
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    # Working Australia voices
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
]


@st.cache_resource
def load_whisper_model(model_name: str = "tiny"):
    """Load and cache the Whisper model for speech-to-text transcription.

    Args:
        model_name: Size of Whisper model (tiny, base, small)
            - tiny: ~39M params, ~1-3s transcription on CPU
            - base: ~74M params, ~5-15s transcription on CPU
            - small: ~244M params, ~15-30s transcription on CPU

    Returns:
        WhisperModel: Cached Whisper model instance
    """
    try:
        logger.info(f"Loading Whisper model: {model_name}")
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"
        )
        logger.info(f"Whisper model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        raise


@st.cache_data
def get_available_edge_tts_voices():
    """Get list of available edge-tts voices for voice selection.

    Returns:
        list: List of voice ID strings (e.g., "en-US-GuyNeural")
    """
    return EDGE_TTS_VOICES


def play_audio_with_autoplay(audio_bytes: bytes, format_str: str = "audio/mp3"):
    """Play audio with JavaScript autoplay (bypasses Streamlit autoplay limitations).

    Args:
        audio_bytes: Audio data as bytes
        format_str: Audio format (e.g., "audio/mp3")
    """
    import base64
    
    # Encode audio to base64
    audio_b64 = base64.b64encode(audio_bytes).decode()
    
    # Create HTML audio player with autoplay
    audio_html = f"""
    <audio autoplay>
        <source src="data:{format_str};base64,{audio_b64}" type="{format_str}">
        Your browser does not support the audio element.
    </audio>
    """
    
    st.html(audio_html)


def get_voice_chat_tools():
    """Get tools for the voice chat agent.

    Returns:
        list: Tools for Wikipedia, Arxiv, and web search
    """
    tools = [
        arxiv_query_run,
        wikipedia_query_run,
        brave_search,
    ]
    return tools


def create_voice_chat_chain(model_name: str):
    """Create and return the voice chat agent for conversational responses.

    Uses agentic pattern with research tools:
    - Wikipedia for general knowledge
    - Arxiv for academic research
    - Web search for current events
    - Voice-optimized prompt keeps responses concise

    Args:
        model_name: Name of the Ollama model to use

    Returns:
        Compiled agent graph ready to invoke
    """
    try:
        system_prompt = create_voice_chat_system_prompt()
        tools = get_voice_chat_tools()

        # Create agent with Ollama model prefix
        # Deep Agents requires format: provider:model-name
        agent = create_deep_agent(
            model=f"ollama:{model_name}",
            tools=tools,
            system_prompt=system_prompt
        )

        logger.info("Voice chat agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Error creating voice chat chain: {e}")
        raise


def process_voice_chat_query(
    agent,
    user_text: str,
    selected_voice: str = "en-US-AriaNeural",
    langfuse_handler=None
) -> dict:
    """Process a user query through the voice chat agent and generate audio.

    Uses agentic reasoning with tools. Returns the final answer synthesized
    to speech.

    Args:
        agent: The compiled Deep Agent graph
        user_text: The user's input text (transcribed or typed)
        selected_voice: edge-tts voice ID for audio synthesis
        langfuse_handler: Optional Langfuse callback handler for tracing

    Returns:
        dict: Response with keys:
            - "text": Agent response text
            - "audio_bytes": WAV audio bytes for speech synthesis
            - "error": Error message if processing failed
    """
    try:
        config = {
            "callbacks": [langfuse_handler] if langfuse_handler else None,
        }

        # Invoke the agent with user input
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config
        )

        # Extract the final message from agent response
        if "messages" in response and response["messages"]:
            response_text = response["messages"][-1].content
        else:
            response_text = str(response)

        logger.info(
            f"Voice chat agent response generated ({len(response_text)} chars): "
            f'"{response_text[:100]}"'
        )

        # Synthesize audio response
        try:
            logger.info(f"[SYNTH] About to synthesize agent response ({len(response_text)} chars)")
            audio_bytes = asyncio.run(
                synthesize_audio_for_voice_chat(
                    response_text,
                    selected_voice
                )
            )
            return {
                "text": response_text,
                "audio_bytes": audio_bytes,
                "error": None
            }
        except Exception as e:
            logger.warning(
                f"Audio synthesis failed, returning text only: {e}"
            )
            return {
                "text": response_text,
                "audio_bytes": None,
                "error": f"Audio synthesis failed: {str(e)}"
            }

    except Exception as e:
        logger.error(f"Error processing voice chat query: {e}")
        return {
            "text": None,
            "audio_bytes": None,
            "error": f"Failed to process query: {str(e)}"
        }


async def synthesize_audio_for_voice_chat(
    text: str,
    voice_id: str,
    max_retries: int = 5
) -> bytes:
    """Synthesize audio from text using Microsoft Edge TTS with enhanced retry logic.

    Includes exponential backoff for edge-tts service recovery.
    Does NOT sanitize text - edge-tts handles special characters fine.

    Args:
        text: Text to synthesize (special characters are OK)
        voice_id: edge-tts voice ID (e.g., "en-US-GuyNeural")
        max_retries: Number of times to retry on failure (default 5)

    Returns:
        bytes: MP3 audio data

    Raises:
        Exception: If synthesis fails after all retries
    """
    import random
    
    # Limit length to avoid TTS timeouts (but keep special characters)
    synthesis_text = text
    if len(synthesis_text) > 500:
        synthesis_text = synthesis_text[:500]
        logger.info(f"[SYNTH] Truncated to 500 chars")
    
    if not synthesis_text.strip():
        logger.error("[SYNTH] Text is empty")
        raise Exception("Text is empty")
    
    # Check for text with only punctuation/special chars
    import string
    if all(c in string.punctuation + string.whitespace for c in synthesis_text):
        logger.error(f"[SYNTH] Text has no actual words: '{synthesis_text}'")
        raise Exception("Text contains only punctuation/whitespace")
    
    # Clean up excessive ellipsis/punctuation (edge-tts can struggle with this)
    # Replace multiple periods with single period, limit trailing punctuation
    synthesis_text = synthesis_text.replace("...", ".")
    synthesis_text = synthesis_text.rstrip(".")  # Remove trailing period
    synthesis_text = synthesis_text + "."  # Add single period back if it had punctuation
    
    logger.info(
        f"[SYNTH] Synthesizing {len(synthesis_text)} chars with voice {voice_id}: "
        f'"{synthesis_text}"'
    )
    
    last_error = None
    
    for attempt in range(max_retries):
        tmp_path = None
        try:
            logger.info(f"[SYNTH] Attempt {attempt + 1}/{max_retries}")

            # Create edge-tts communicator (without text sanitization)
            logger.info(f"[SYNTH] Creating Communicate(text={len(synthesis_text)} chars, voice={voice_id})")
            communicate = edge_tts.Communicate(
                text=synthesis_text,
                voice=voice_id
            )
            logger.info(f"[SYNTH] Communicate object created successfully")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.mp3',
                delete=False
            ) as tmp:
                tmp_path = tmp.name
            
            # Save the audio (immediately, no pre-validation)
            try:
                logger.info(f"[SYNTH] Calling save() on Communicate object")
                await communicate.save(tmp_path)
                logger.info(f"[SYNTH] save() succeeded")
            except Exception as save_err:
                logger.error(f"[SYNTH] save() failed: {type(save_err).__name__}: {save_err}")
                raise save_err
            
            # Give file system a moment to finalize
            await asyncio.sleep(0.1)

            # Read the MP3 file immediately (matching pdf_podcast pattern)
            try:
                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()
                
                if not audio_bytes:
                    raise Exception("Failed to read audio bytes")
                
                logger.info(f"[SYNTH] SUCCESS: {len(audio_bytes)} bytes")
                
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logger.warning(f"[SYNTH] Failed to cleanup: {e}")
                
                return audio_bytes
                
            except IOError as io_err:
                logger.error(f"[SYNTH] IOError reading file: {io_err}")
                raise Exception(f"IO error: {io_err}")

        except Exception as e:
            last_error = e
            logger.warning(
                f"[SYNTH] Attempt {attempt + 1} failed: {type(e).__name__}: {e}"
            )
            
            # Log more details about the error
            if "NoAudioReceived" in str(type(e)):
                logger.error(
                    f"[SYNTH] Edge-tts returned no audio. "
                    f"Text: '{synthesis_text[:50]}...' "
                    f"Voice: {voice_id} "
                    f"Attempt: {attempt + 1}/{max_retries}"
                )
            
            # Try to clean up on error
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.info(f"[SYNTH] Cleaned up on error: {tmp_path}")
                except Exception as cleanup_err:
                    logger.warning(f"[SYNTH] Cleanup failed: {cleanup_err}")
            
            # Add exponential backoff before retry (longer waits for service recovery)
            if attempt < max_retries - 1:
                # 1s, 2s, 4s, 8s, 16s with random jitter (±20%)
                base_wait = 2 ** attempt
                jitter = random.uniform(0.8, 1.2)
                wait_time = base_wait * jitter
                logger.info(
                    f"[SYNTH] Retrying in {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
    
    # All retries failed
    logger.error(
        f"[SYNTH] FAILED after {max_retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )
    raise last_error


def handle_voice_chat(st, model_name: str, langfuse_handler=None):
    """Handle Voice Chat interaction with Streamlit UI and session state.

    Features:
    - Speech-to-text via faster-whisper (user-selectable model size)
    - Voice selection via edge-tts
    - Persistent conversation history
    - Text fallback if audio capture fails
    - Auto-playing audio responses

    Args:
        st: Streamlit module
        model_name: Name of the Ollama model to use
        langfuse_handler: Optional Langfuse callback handler for tracing
    """
    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.last_processed_input = None

    if "selected_whisper_model" not in st.session_state:
        st.session_state.selected_whisper_model = "tiny"

    if "selected_voice" not in st.session_state:
        st.session_state.selected_voice = "en-US-AriaNeural"

    # Sidebar configuration
    with st.sidebar:
        st.header("Voice Chat Settings")

        # Whisper model selection
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small"],
            index=["tiny", "base", "small"].index(
                st.session_state.selected_whisper_model
            ),
            help=(
                "tiny: ~1-3s (fast, good for clear audio) | "
                "base: ~5-15s (better accuracy) | "
                "small: ~15-30s (best accuracy, slower)"
            )
        )
        st.session_state.selected_whisper_model = whisper_model

        # Voice selection
        available_voices = get_available_edge_tts_voices()
        voice_index = (
            available_voices.index(st.session_state.selected_voice)
            if st.session_state.selected_voice in available_voices
            else 0
        )
        selected_voice = st.selectbox(
            "Output Voice",
            available_voices,
            index=voice_index,
            help="Select the voice for audio responses"
        )
        st.session_state.selected_voice = selected_voice

        st.info(
            "🔬 **Agentic Mode**: Uses research tools (Wikipedia, Arxiv, Web "
            "Search) for informed responses.\n"
            f"⏱️  Expected latency: Whisper ({whisper_model}) + Agent reasoning "
            "+ TTS\n"
            f"Tiny: ~10-30s | Base: ~15-40s | Small: ~30-60s\n"
            "💡 Shows thinking messages while researching your question."
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If message has audio, display player
            if message.get("audio_bytes"):
                st.audio(message["audio_bytes"], format="audio/mp3")

    # Audio input and transcription
    st.subheader("Record or Type Your Message")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Option 1: Record Audio**")
        audio_data = st.audio_input(
            "Click to record your message",
            key=f"audio_input_{st.session_state.session_id}"
        )

    with col2:
        st.write("**Option 2: Type Text**")
        text_input = st.text_input(
            "Or type your message here (for testing/accessibility)",
            key=f"text_input_{st.session_state.session_id}"
        )

    # Process user input
    user_input = None
    input_source = None

    if audio_data:
        # Try to transcribe audio
        try:
            with st.spinner("Transcribing audio..."):
                # Load Whisper model
                whisper_model = load_whisper_model(whisper_model)

                # Transcribe
                segments, _ = whisper_model.transcribe(audio_data)
                user_input = " ".join([segment.text for segment in segments])
                input_source = "audio"

                # Display transcription
                st.info(f"**Transcribed:** {user_input}")
        except Exception as e:
            st.error(f"Failed to transcribe audio: {str(e)}")
            logger.error(f"Transcription error: {e}")

    elif text_input:
        user_input = text_input
        input_source = "text"

    # Process the user input if available
    if user_input and user_input != st.session_state.last_processed_input:
        # Mark this input as processed to prevent infinite loops
        st.session_state.last_processed_input = user_input

        # Add user message to chat
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "audio_bytes": None,
            "input_source": input_source
        })

        # Show thinking message while agent processes
        thinking_message = random.choice(THINKING_MESSAGES)
        with st.spinner(thinking_message):
            # Synthesize and play thinking message audio for feedback
            try:
                thinking_audio = asyncio.run(
                    synthesize_audio_for_voice_chat(
                        thinking_message,
                        st.session_state.selected_voice
                    )
                )
                if thinking_audio:
                    play_audio_with_autoplay(thinking_audio, "audio/mp3")
                    # Small delay to let user hear the thinking message
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to synthesize thinking message: {e}")
                # Continue even if thinking message fails - don't block processing

            try:
                # Create agent
                agent = create_voice_chat_chain(model_name)

                # Process query
                result = process_voice_chat_query(
                    agent,
                    user_input,
                    selected_voice=st.session_state.selected_voice,
                    langfuse_handler=langfuse_handler
                )

                if result["text"]:
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(result["text"])

                        # Play audio if available
                        if result["audio_bytes"]:
                            play_audio_with_autoplay(
                                result["audio_bytes"],
                                "audio/mp3"
                            )
                        elif result["error"]:
                            st.warning(result["error"])

                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["text"],
                        "audio_bytes": result["audio_bytes"],
                        "error": result["error"]
                    })

                    # No rerun needed - input deduplication prevents loops
                    # User already sees autoplay response above
                else:
                    st.error(
                        result["error"] or "Failed to generate response"
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Voice chat error: {e}")
