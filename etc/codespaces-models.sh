#!/bin/bash

# Recommended models for 8GB Codespaces
echo "🤖 Installing memory-optimized models for Codespaces..."

# Wait for Ollama to be ready
until curl -s http://localhost:11434/api/version; do
    echo "⏳ Waiting for Ollama to start..."
    sleep 5
done

# Install lightweight models that work well in 8GB
echo "📥 Installing llama3.2:3b (3B parameters, ~2GB memory)..."
docker exec ollama ollama pull llama3.2:3b

echo "📥 Installing phi3:mini (3.8B parameters, ~2.3GB memory)..."
docker exec ollama ollama pull phi3:mini

echo "📥 Installing gemma2:2b (2B parameters, ~1.6GB memory)..."
docker exec ollama ollama pull gemma2:2b

echo "✅ Model installation complete!"
echo ""
echo "💡 Available models:"
echo "  - llama3.2:3b (recommended for chat)"
echo "  - phi3:mini (good for reasoning)"
echo "  - gemma2:2b (fastest, least memory)"
