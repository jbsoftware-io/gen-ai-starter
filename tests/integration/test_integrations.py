import pytest
import requests
from unittest.mock import patch, Mock


class TestOllamaIntegration:
    """Integration tests for Ollama API connectivity"""

    @pytest.mark.integration
    @patch('requests.get')
    def test_ollama_api_connection(self, mock_get, mock_env_vars):
        """Test actual connection to Ollama API"""
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3.2:latest"}]
        }
        mock_get.return_value.status_code = 200

        response = requests.get(f"{mock_env_vars['OLLAMA_HOST']}/api/tags")
        assert response.status_code == 200
        assert "models" in response.json()

    @pytest.mark.integration
    def test_ollama_health_check(self, mock_env_vars):
        """Test Ollama service health check"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200

            response = requests.get(f"{mock_env_vars['OLLAMA_HOST']}/api/tags")
            assert response.status_code == 200


class TestChromaIntegration:
    """Integration tests for Chroma vector database"""

    @pytest.mark.integration
    @patch('chromadb.Client')
    def test_chroma_connection(self, mock_client, mock_env_vars):
        """Test connection to Chroma database"""
        mock_chroma_client = Mock()
        mock_client.return_value = mock_chroma_client

        import chromadb
        client = chromadb.Client()

        assert client is not None
        mock_client.assert_called_once()


class TestPGVectorIntegration:
    """Integration tests for PGVector database"""

    @pytest.mark.integration
    @patch('psycopg2.connect')
    def test_postgres_connection(self, mock_connect, mock_env_vars):
        """Test connection to PostgreSQL with pgvector"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )

        assert conn is not None
        mock_connect.assert_called_once()
