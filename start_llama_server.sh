#!/usr/bin/env bash
set -euo pipefail
set -a
source /opt/nextcloud-rag/.env
set +a

exec /opt/llama.cpp/build/bin/llama-server \
  -m "$LLAMA_MODEL_PATH" \
  --host "$LLAMA_HOST" \
  --port "$LLAMA_PORT" \
  -c "$LLAMA_CTX_SIZE" \
  -t "$LLAMA_THREADS" \
  -ngl "$LLAMA_N_GPU_LAYERS"
