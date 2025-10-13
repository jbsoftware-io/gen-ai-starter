import pytest
import requests
import time
import psycopg2
import chromadb
import os

from langchain_community.llms import Ollama
from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from internal.util import create_llm, format_docs
from internal.prompts import (
    create_question_type_prompt,
    create_summarize_prompt_v2
)
from example.wikipedia import create_wikipedia_chain


class TestOllamaIntegration:
    """Integration tests for Ollama API connectivity"""

    @pytest.mark.integration
    def test_ollama_api_health_check(self):
        """Test actual connection to Ollama API"""
        ollama_host = "http://host.docker.internal:11434"

        # Wait for service to be ready (with timeout)
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    msg = ("Ollama service not available - ensure "
                           "'docker compose up -d' is running")
                    pytest.fail(msg)
                time.sleep(1)

        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    @pytest.mark.integration
    def test_ollama_langchain_integration(self):
        """Test LangChain integration with Ollama"""
        ollama_host = "http://host.docker.internal:11434"

        # Check if any models are available
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.fail("Ollama service not available")

            models = response.json().get("models", [])
            if not models:
                msg = ("No Ollama models available - "
                       "run 'ollama pull llama3.2' first")
                pytest.fail(msg)

            # Use the first available model for testing
            model_name = models[0]["name"]

        except requests.exceptions.RequestException:
            pytest.fail("Ollama service not available")

        # Test LangChain Ollama integration
        llm = Ollama(
            model=model_name,
            base_url=ollama_host
        )

        # Simple test query
        response = llm.invoke("Hello! Respond with just 'OK' if you receive this.")  # noqa: E501
        assert response is not None
        assert len(response.strip()) > 0


class TestChromaIntegration:
    """Integration tests for Chroma vector database"""

    @pytest.mark.integration
    def test_chroma_connection_and_operations(self):
        """Test connection and basic operations with Chroma database"""
        chroma_host = "host.docker.internal"
        chroma_port = 9000

        try:
            # Test connection
            client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )

            # Test basic operations
            collection_name = "test_integration_collection"

            # Clean up any existing test collection
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass  # Collection might not exist

            # Create collection
            collection = client.create_collection(name=collection_name)
            assert collection is not None

            # Add some test documents
            test_docs = [
                "This is a test document about cats.",
                "This is a test document about dogs.",
                "This is a test document about birds."
            ]

            collection.add(
                documents=test_docs,
                ids=["doc1", "doc2", "doc3"]
            )

            # Query the collection
            results = collection.query(
                query_texts=["pets"],
                n_results=2
            )

            assert len(results["documents"][0]) == 2
            assert len(results["ids"][0]) == 2

            # Clean up
            client.delete_collection(name=collection_name)

        except Exception as e:
            if "Connection refused" in str(e) or "Failed to connect" in str(e):
                msg = ("Chroma service not available - ensure "
                       "'docker compose up -d' is running")
                pytest.fail(msg)
            else:
                raise


class TestPGVectorIntegration:
    """Integration tests for PGVector database"""

    @pytest.mark.integration
    def test_postgres_connection_and_pgvector(self):
        """Test connection to PostgreSQL with pgvector extension"""
        db_config = {
            "host": "postgres",
            "port": 5432,
            "database": "api",
            "user": "myuser",
            "password": "ChangeMe"
        }

        try:
            # Test connection
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Test pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

            # Test creating a table with vector column
            test_table = "test_integration_vectors"
            cursor.execute(f"DROP TABLE IF EXISTS {test_table};")
            cursor.execute(f"""
                CREATE TABLE {test_table} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding VECTOR(3)
                );
            """)
            conn.commit()

            # Test inserting vector data
            cursor.execute(f"""
                INSERT INTO {test_table} (content, embedding)
                VALUES (%s, %s);
            """, ("test document", "[1,2,3]"))
            conn.commit()

            # Test querying vector data
            cursor.execute(f"SELECT content, embedding FROM {test_table};")
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0][0] == "test document"

            # Test vector similarity (cosine distance)
            cursor.execute(f"""
                SELECT content, embedding <=> '[1,2,4]' AS distance
                FROM {test_table}
                ORDER BY distance;
            """)
            results = cursor.fetchall()
            assert len(results) == 1
            assert results[0][1] is not None  # Should have a distance value

            # Clean up
            cursor.execute(f"DROP TABLE {test_table};")
            conn.commit()

            cursor.close()
            conn.close()

        except psycopg2.OperationalError as e:
            if "Connection refused" in str(e) or "could not connect" in str(e):
                msg = ("PostgreSQL service not available - ensure "
                       "'docker compose up -d' is running")
                pytest.fail(msg)
            else:
                raise


class TestLLMChainIntegration:
    """Integration tests for LLM chain functionality"""

    @pytest.mark.integration
    def test_simple_llm_chain_integration(self):
        """Test basic LLM chain with real Ollama"""
        ollama_host = "http://host.docker.internal:11434"

        # Check if models are available
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama service not available")

            models = response.json().get("models", [])
            if not models:
                pytest.skip("No Ollama models available")

            model_name = models[0]["name"]

        except requests.exceptions.RequestException:
            pytest.skip("Ollama service not available")

        # Test the LLM chain integration
        llm = create_llm(model_name)
        prompt = create_question_type_prompt()
        chain = prompt | llm | StrOutputParser()

        # Test chain execution
        result = chain.invoke({
            'search_query': 'What is the capital?',
            'selections': ['Texas'],
            'type': 'States'
        })

        assert result is not None
        assert len(result.strip()) > 0
        assert 'austin' in result.lower() or 'texas' in result.lower()

    @pytest.mark.integration
    def test_wikipedia_retriever_integration(self):
        """Test Wikipedia retriever functionality"""
        ollama_host = "http://host.docker.internal:11434"

        # Check if models are available
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama service not available")

            models = response.json().get("models", [])
            if not models:
                pytest.skip("No Ollama models available")

        except requests.exceptions.RequestException:
            pytest.skip("Ollama service not available")

        # Test that we can create the Wikipedia chain without errors
        # This tests the integration points without external API calls
        try:
            from langchain_community.retrievers import WikipediaRetriever

            # Just test that the retriever can be instantiated
            retriever = WikipediaRetriever(
                top_k_results=1,
                doc_content_chars_max=100
            )
            assert retriever is not None

            # Test that our chain creation function works
            chain = create_wikipedia_chain(models[0]["name"])
            assert chain is not None

        except Exception as e:
            pytest.fail(f"Failed to create Wikipedia chain: {e}")

    @pytest.mark.integration
    def test_arxiv_retriever_integration(self):
        """Test ArXiv retriever functionality"""
        ollama_host = "http://host.docker.internal:11434"

        # Check if models are available
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                pytest.skip("Ollama service not available")

            models = response.json().get("models", [])
            if not models:
                pytest.skip("No Ollama models available")

        except requests.exceptions.RequestException:
            pytest.skip("Ollama service not available")

        # Test that we can create ArXiv retriever and chain without errors
        try:
            # Test ArXiv retriever instantiation
            retriever = ArxivRetriever(
                load_max_docs=1,
                get_full_documents=False
            )
            assert retriever is not None

            # Test that we can create LLM chain components
            llm = create_llm(models[0]["name"])
            prompt = create_summarize_prompt_v2()

            # Test that chain components can be combined
            rag_chain = (
                RunnablePassthrough.assign(
                    context=(lambda x: format_docs(x["context"])))
                | prompt
                | llm
                | StrOutputParser()
            )
            assert rag_chain is not None

        except Exception as e:
            pytest.fail(f"Failed to create ArXiv chain components: {e}")


class TestEnvironmentIntegration:
    """Integration tests for environment configuration"""

    @pytest.mark.integration
    def test_all_required_environment_variables(self):
        """Test that all required environment variables are accessible from container"""  # noqa: E501
        # Test OLLAMA_HOST is accessible and valid
        ollama_host = os.getenv("OLLAMA_HOST")
        assert ollama_host is not None, "OLLAMA_HOST not set"

        # Test if Ollama is actually reachable
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.fail("Cannot reach Ollama at OLLAMA_HOST")

        # Test Chroma environment variables
        chroma_host = os.getenv("CHROMA_HOST")
        chroma_port = os.getenv("CHROMA_PORT")
        assert chroma_host is not None, "CHROMA_HOST not set"
        assert chroma_port is not None, "CHROMA_PORT not set"

        # Test PostgreSQL environment variable
        db_url = os.getenv("DB_URL")
        assert db_url is not None, "DB_URL not set"
        expected_prefix = "postgresql://"
        assert expected_prefix in db_url, "DB_URL should be a PostgreSQL connection string"  # noqa: E501

    @pytest.mark.integration
    def test_model_availability_check(self):
        """Test that at least one model is available in Ollama"""
        ollama_host = os.getenv("OLLAMA_HOST")

        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            assert response.status_code == 200

            models = response.json().get("models", [])
            msg = "No models available in Ollama - run 'ollama pull llama3.2'"
            assert len(models) > 0, msg

            # Test that we can get model info
            for model in models:
                assert "name" in model
                assert len(model["name"]) > 0

        except requests.exceptions.RequestException:
            pytest.fail("Cannot connect to Ollama service")


class TestServiceHealthIntegration:
    """Integration tests for overall service health"""

    @pytest.mark.integration
    def test_all_services_healthy(self):
        """Test that all required services are running and healthy"""

        # Test Ollama
        try:
            response = requests.get(
                "http://host.docker.internal:11434/api/tags", timeout=5
            )
            assert response.status_code == 200, "Ollama service unhealthy"
        except requests.exceptions.RequestException:
            pytest.fail("Ollama service not reachable")

        # Test Chroma
        try:
            client = chromadb.HttpClient(
                host="host.docker.internal",
                port=9000
            )
            # Try to list collections (should not fail)
            client.list_collections()
        except Exception as e:
            if "Connection refused" in str(e):
                pytest.fail("Chroma service not reachable")
            # Other errors might be expected (like version mismatches)

        # Test PostgreSQL
        try:
            conn = psycopg2.connect(
                host="postgres",
                port=5432,
                database="api",
                user="myuser",
                password="ChangeMe"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1, "PostgreSQL query failed"
            cursor.close()
            conn.close()
        except psycopg2.OperationalError:
            pytest.fail("PostgreSQL service not reachable")
