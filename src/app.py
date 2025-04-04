from dotenv import load_dotenv
import requests
import os, streamlit as st  # noqa: E401
from internal.chroma import handle_chroma
from internal.city import handle_cities
from internal.country import handle_country
from internal.mtg import handle_mtg
from internal.pgvector import handle_pgvector
from internal.state import handle_states

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"

st.title("Generative AI Demo")

with st.sidebar:
    type = st.selectbox(
        "Select a Type",
        ["Countries", "States", "Cities", "MTG", "Chroma", "PG_Vector"]
    )
    available_models = requests.get(f"{OLLAMA_HOST}/api/tags").json()
    model_names = [model["name"] for model in available_models["models"]]

    model_name = st.selectbox("Model Name", model_names)

if type == "Countries":
    handle_country(st, model_name)

if type == "States":
    handle_states(st, model_name)

if type == "Cities":
    handle_cities(st, model_name)

if type == "MTG":
    handle_mtg(st, model_name)

if type == "Chroma":
    handle_chroma(st, model_name)

if type == "PG_Vector":
    handle_pgvector(st, model_name)
