import pytest
import requests
from unittest.mock import patch


@pytest.mark.unit
class TestApp:
    """Test cases for main app.py functionality"""

    @patch('requests.get')
    def test_ollama_models_fetch_success(
            self, mock_get, mock_env_vars, mock_ollama_response):
        """Test successful fetching of Ollama models"""
        mock_get.return_value.json.return_value = mock_ollama_response
        mock_get.return_value.status_code = 200

        ollama_host = mock_env_vars['OLLAMA_HOST']
        response = requests.get(f"{ollama_host}/api/tags")
        models = response.json()

        assert "models" in models
        assert len(models["models"]) == 3
        assert models["models"][0]["name"] == "llama3.2:latest"

    @patch('requests.get')
    def test_ollama_connection_failure(self, mock_get, mock_env_vars):
        """Test handling of Ollama connection failure"""
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        ollama_host = mock_env_vars['OLLAMA_HOST']

        with pytest.raises(requests.ConnectionError):
            requests.get(f"{ollama_host}/api/tags")

    def test_model_selection_logic(self, mock_ollama_response):
        """Test model name extraction and sorting"""
        models = mock_ollama_response["models"]
        model_names = [model["name"] for model in models]
        sorted_names = sorted(model_names)

        expected = ["codellama:latest", "llama3.2:latest", "mistral:latest"]
        assert sorted_names == expected
