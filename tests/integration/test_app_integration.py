import os
import pytest
import requests
from dotenv import load_dotenv
from unittest.mock import patch, Mock

import app

load_dotenv()

type_options = [
    "Cities", "States", "Countries", "MTG", "Chroma", "PG_Vector",
    "Web", "Wikipedia", "Arxiv", "Simple_Chat", "Agentic_Chat"
]


class TestAppIntegration:
    @staticmethod
    def _get_first_ollama_model():
        ollama_host = os.getenv("OLLAMA_HOST")
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code != 200:
            pytest.fail("Ollama service not available")

        models = response.json().get("models", [])
        if not models:
            msg = ("No Ollama models found, run 'ollama pull llama3.2' first")
            pytest.fail(msg)

        model_name = models[0]["name"]
        return model_name, models

    @pytest.mark.parametrize("selected_type", [t for t in type_options if t not in ["Simple_Chat", "MTG"]])  # noqa: E501
    def test_app_type_selection(self, selected_type):
        """
        Verify each type selection in the Streamlit dropdown sets up the UI correctly (excluding Simple_Chat and MTG).  # noqa: E501
        """
        model_name, models = self._get_first_ollama_model()

        with patch('app.st') as mock_st:
            mock_st.selectbox.side_effect = [selected_type, model_name]
            mock_st.title = Mock()
            mock_st.sidebar = Mock()
            mock_st.sidebar.selectbox = Mock()
            mock_st.sidebar.__enter__ = lambda s: s
            mock_st.sidebar.__exit__ = lambda s, exc_type, exc_val, exc_tb: None  # noqa: E501

            app.main()

            mock_st.title.assert_called_once_with("Generative AI Demo")
            mock_st.selectbox.assert_any_call(
                "Select a Type",
                type_options
            )
            mock_st.selectbox.assert_any_call(
                "Model Name",
                sorted([model["name"] for model in models])
            )

    def test_app_type_selection_simple_chat(self):
        """
        Special test for Simple_Chat type selection with chat input mocked.
        """
        model_name, models = self._get_first_ollama_model()

        with patch('app.st') as mock_st, \
             patch('app.st.chat_input', return_value="Hello!"):
            mock_st.selectbox.side_effect = ["Simple_Chat", model_name]
            mock_st.title = Mock()
            mock_st.sidebar = Mock()
            mock_st.sidebar.selectbox = Mock()
            mock_st.sidebar.__enter__ = lambda s: s
            mock_st.sidebar.__exit__ = lambda s, exc_type, exc_val, exc_tb: None  # noqa: E501

            app.main()

            mock_st.title.assert_called_once_with("Generative AI Demo")
            mock_st.selectbox.assert_any_call(
                "Select a Type",
                type_options
            )
            mock_st.selectbox.assert_any_call(
                "Model Name",
                sorted([model["name"] for model in models])
            )

    def test_app_type_selection_mtg(self):
        """
        Special test for MTG type selection with card data and columns mocked.
        """
        model_name, models = self._get_first_ollama_model()

        with patch('app.st') as mock_st, \
             patch('app.st.text_input', return_value="Golden"):
            mock_st.selectbox.side_effect = ["MTG", model_name]
            mock_st.title = Mock()
            mock_st.sidebar = Mock()
            mock_st.sidebar.selectbox = Mock()
            mock_st.sidebar.__enter__ = lambda s: s
            mock_st.sidebar.__exit__ = lambda s, exc_type, exc_val, exc_tb: None  # noqa: E501
            col2 = Mock()
            col2.__enter__ = lambda s: s
            col2.__exit__ = lambda s, exc_type, exc_val, exc_tb: None
            mock_st.columns.return_value = (Mock(), col2, Mock())

            app.main()

            mock_st.title.assert_called_once_with("Generative AI Demo")
            mock_st.selectbox.assert_any_call(
                "Select a Type",
                type_options
            )
            mock_st.selectbox.assert_any_call(
                "Model Name",
                sorted([model["name"] for model in models])
            )
