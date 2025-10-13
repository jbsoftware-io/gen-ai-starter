import pytest
import requests
import time
import psycopg2
import chromadb
from langchain_community.llms import Ollama


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
                    pytest.fail("Ollama service not available - ensure 'docker compose up -d' is running")
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
                pytest.fail("No Ollama models available - run 'ollama pull llama3.2' first")
                
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
        response = llm.invoke("Hello! Respond with just 'OK' if you receive this.")
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
                pytest.fail("Chroma service not available - ensure 'docker compose up -d' is running")
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
                pytest.fail("PostgreSQL service not available - ensure 'docker compose up -d' is running")
            else:
                raise
