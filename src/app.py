import logging
from dotenv import load_dotenv
import requests
import os, streamlit as st  # noqa: E401
from example.chroma import handle_chroma
from example.city import handle_cities
from example.country import handle_country
from example.mtg import handle_mtg
from example.pgvector import handle_pgvector
from example.state import handle_states
from example.wikipedia import handle_wikipedia


logging.basicConfig(level=logging.INFO)
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"

st.title("Generative AI Demo")

with st.sidebar:
    type = st.selectbox(
        "Select a Type",
        [
            "Countries", "States", "Cities", "MTG", "Chroma", "PG_Vector",
            "Wikipedia"
        ]
    )
    available_models = requests.get(f"{OLLAMA_HOST}/api/tags").json()
    model_names = [model["name"] for model in available_models["models"]]

    model_name = st.selectbox("Model Name", sorted(model_names))

    selected_type = type
    selected_model = model_name

if selected_type == "Countries":
    handle_country(st, selected_model)

if selected_type == "States":
    handle_states(st, selected_model)

if selected_type == "Cities":
    handle_cities(st, selected_model)

if selected_type == "MTG":
    handle_mtg(st, selected_model)

if selected_type == "Chroma":
    handle_chroma(st, selected_model)

if selected_type == "PG_Vector":
    handle_pgvector(st, selected_model)

if selected_type == "Wikipedia":
    handle_wikipedia(st, selected_model)
