import os
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests
from dotenv import load_dotenv

import app

load_dotenv()

type_options = [
    "Cities", "States", "Countries", "MTG", "Chroma", "PG_Vector",
    "PDF2Audio_Local", "Web", "Wikipedia", "Arxiv", "Simple_Chat",
    "Agentic_Chat", "Deep_Agents", "Pokemon_MCP"
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
            msg = ("No Ollama models found")
            pytest.fail(msg)

        model_name = models[0]["name"]
        return model_name, models

    @staticmethod
    def _create_mock_session_state():
        """Create a proper mock session state object for interactive features"""
        class MockSessionState:
            def __init__(self):
                self._data = {}

            def __contains__(self, key):
                return key in self._data

            def __getattr__(self, key):
                return self._data.get(key)

            def __setattr__(self, key, value):
                if key.startswith('_'):
                    super().__setattr__(key, value)
                else:
                    self._data[key] = value

        return MockSessionState()

    @pytest.mark.parametrize("selected_type", [t for t in type_options if t not in ["Agentic_Chat", "Simple_Chat", "MTG", "PDF2Audio_Local"]])  # noqa: E501
    def test_app_type_selection(self, selected_type):
        """
        Verify each type selection in the Streamlit dropdown sets up the UI correctly.
        Excludes interactive features: Simple_Chat, Agentic_Chat, MTG, PDF2Audio_Local.
        """
        model_name, models = self._get_first_ollama_model()

        with patch('app.st') as mock_st:
            mock_st.selectbox.side_effect = [selected_type, model_name]
            mock_st.title = Mock()
            mock_st.sidebar = Mock()
            mock_st.sidebar.selectbox = Mock()
            mock_st.sidebar.__enter__ = lambda s: s
            mock_st.sidebar.__exit__ = lambda s, exc_type, exc_val, exc_tb: None  # noqa: E501

            # Add mocking for common Streamlit functions used by examples
            mock_st.text_input = Mock(return_value="test query")
            mock_st.button = Mock(return_value=False)  # Don't trigger on_click

            # Proper context manager for spinner
            mock_spinner = MagicMock()
            mock_spinner.__enter__ = Mock(return_value=None)
            mock_spinner.__exit__ = Mock(return_value=False)
            mock_st.spinner = Mock(return_value=mock_spinner)

            mock_st.warning = Mock()
            mock_st.success = Mock()
            mock_st.error = Mock()
            mock_st.info = Mock()
            mock_st.exception = Mock()
            mock_st.chat_input = Mock(return_value=None)
            mock_st.file_uploader = Mock(return_value=None)
            mock_st.subheader = Mock()

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

            # Setup session_state for simple_chat
            mock_st.session_state = self._create_mock_session_state()

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

            # Setup session_state
            mock_st.session_state = self._create_mock_session_state()

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
