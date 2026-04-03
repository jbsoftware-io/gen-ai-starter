#!/bin/bash


# Start Ollama in the background.
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
pid=$!

# Pause for Ollama to start.
sleep 5

# Read models from OLLAMA_MODELS env variable, default if not set
MODELS=${OLLAMA_MODELS:-"llama3.2;mistral;gemma4:e4b"}

# Loop through models and pull each
IFS=';' read -ra MODEL_LIST <<< "$MODELS"
for model in "${MODEL_LIST[@]}"; do
	echo "🔴 Retrieve $model model..."
	ollama pull "$model"
	echo "🟢 Done!"
done

# Wait for Ollama process to finish.
wait $pid
