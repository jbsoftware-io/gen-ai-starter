#!/bin/bash
set -e

echo "🚀 Setting up GenAI Starter in Codespaces (8GB optimized)..."

# Copy codespaces environment file
if [ ! -f .env ]; then
    cp .env.codespaces .env
    echo "✅ Environment file configured"
fi

# Start services with memory limits
echo "🔄 Starting services with memory optimization..."
docker compose --profile=cpu up -d

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/version; do
    sleep 5
done

# Check memory usage
echo "📊 Memory usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo "✅ Setup complete!"
echo ""
echo "🌐 Access your application at:"
echo "  - Streamlit App: https://$CODESPACE_NAME-8501.app.github.dev"
echo "  - Open WebUI: https://$CODESPACE_NAME-3000.app.github.dev"
echo ""

