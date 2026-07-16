import os
import uuid

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

assert OLLAMA_HOST, "OLLAMA_HOST is not set"

store = {}  # memory is maintained outside the chain


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def handle_simple_chat(st, model_name, langfuse_handler=None):
    llm = ChatOllama(
        temperature=0.0,
        model=model_name,
        base_url=OLLAMA_HOST,
        streaming=True)
    chain = RunnableWithMessageHistory(llm, get_session_history)

    # start a new chat session
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", "content": "How can I help you?"})
        st.session_state.session_id = str(uuid.uuid4())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if input := st.chat_input("Type here..."):
        st.chat_message("user").markdown(input)
        st.session_state.messages.append({
            "role": "user", "content": input})

        with st.spinner("Processing, please wait..."):
            config = {
                "session_id": st.session_state.session_id,
                "callbacks": [langfuse_handler] if langfuse_handler else None
            }
            response = chain.invoke({"input": input}, config=config)

        with st.chat_message("assistant"):
            st.markdown(response.content)

        st.session_state.messages.append({
            "role": "assistant", "content": response.content})
