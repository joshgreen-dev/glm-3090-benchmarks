#!/usr/bin/env bash
# vLLM tensor-parallel (TP=2) across both RTX 3090s. Single-stream latency at a
# few context sizes, then throughput under concurrency. Needs the two small
# bench_*.py helpers (an OpenAI-compatible client hitting the local server).
set -euo pipefail

MODEL="${MODEL:-cyankiwi/GLM-4.7-Flash-AWQ-4bit}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 90   # weights load + graph capture

echo "# single-stream decode"
for toks in 4096 32768 115000; do
  python bench_latency.py --prompt-tokens "$toks" --output-tokens 128
done

echo "# throughput under concurrency"
for conc in 8 16 32; do
  python bench_throughput.py --concurrency "$conc" --requests 128
done
