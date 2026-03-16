
import logging
import os

import requests
import streamlit as st  # noqa: E401
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

from example.agentic_chat import handle_agentic_chat
from example.arxiv import handle_arxiv
from example.chroma import handle_chroma
from example.city import handle_cities
from example.country import handle_country
from example.deep_agents import handle_deep_agents
from example.mtg import handle_mtg
from example.pgvector import handle_pgvector
from example.simple_chat import handle_simple_chat
from example.state import handle_states
from example.web import handle_web
from example.wikipedia import handle_wikipedia


def main():
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")

    assert OLLAMA_HOST, "OLLAMA_HOST is not set"

    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
    langfuse_enabled = all([
        LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL])
    langfuse_handler = None

    if langfuse_enabled:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

    st.title("Generative AI Demo")

    with st.sidebar:
        type = st.selectbox(
            "Select a Type",
            [
                "Cities", "States", "Countries", "MTG",
                "Chroma", "PG_Vector", "Web", "Wikipedia",
                "Arxiv", "Simple_Chat", "Agentic_Chat",
                "Deep_Agents"
            ]
        )
        available_models = requests.get(f"{OLLAMA_HOST}/api/tags").json()
        model_names = [model["name"] for model in available_models["models"]]

        model_name = st.selectbox("Model Name", sorted(model_names))

        selected_type = type
        selected_model = model_name

    if selected_type == "Cities":
        handle_cities(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "States":
        handle_states(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Countries":
        handle_country(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "MTG":
        handle_mtg(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Chroma":
        handle_chroma(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "PG_Vector":
        handle_pgvector(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Web":
        handle_web(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Wikipedia":
        handle_wikipedia(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Arxiv":
        handle_arxiv(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Simple_Chat":
        handle_simple_chat(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Agentic_Chat":
        handle_agentic_chat(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )

    if selected_type == "Deep_Agents":
        handle_deep_agents(
            st,
            selected_model,
            langfuse_handler=langfuse_handler
        )


if __name__ == "__main__":
    main()
