#!/usr/bin/env bash
# Decode-speed and VRAM sweep for GLM-4.7-Flash against qwen3.5:27b on ollama.
# Prints eval rate and GPU memory at each context length. Pipe to a file and
# tidy into results/decode_speed.csv and results/vram.csv.
set -euo pipefail

MODELS=("glm-4.7-flash" "qwen3.5:27b")
CTX=(4096 16384 32768 65536 131072)
PROMPT="Summarize the history of computing in about 200 words."

for m in "${MODELS[@]}"; do
  for c in "${CTX[@]}"; do
    echo "== ${m} @ ${c} =="
    OLLAMA_CONTEXT_LENGTH=$c ollama run "$m" --verbose "$PROMPT" 2>&1 \
      | grep -E "prompt eval rate|eval rate" || true
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo
  done
done
