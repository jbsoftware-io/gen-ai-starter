#!/bin/sh
set -o pipefail
for i in $(seq 1 30); do
  curl -s http://localhost:11434/api/tags | grep -o '"name":"[^\"]*"' | grep . && \
  curl -s -X POST -d '{"name":"llama3.2:1b"}' http://localhost:11434/api/show | grep '"details"' && exit 0
  echo 'Waiting for Ollama...'
  sleep 10
done
exit 1
