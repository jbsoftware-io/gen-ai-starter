#!/bin/bash

# Start Ollama in the background.
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA3.2 model..."
ollama pull llama3.2
echo "🟢 Done!"

echo "🔴 Retrieve Mistral model..."
ollama pull mistral
echo "🟢 Done!"

echo "🔴 Retrieve Gemma 3 model..."
ollama pull gemma3:4b
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
