#!/bin/bash

# Start Ollama in the background.
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve llama3.2:3b model..."
ollama pull llama3.2:3b
echo "🟢 Done!"

echo "🔴 Retrieve gemma3:1b model..."
ollama pull gemma3:1b
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
