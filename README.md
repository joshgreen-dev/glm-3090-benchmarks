# glm-3090-benchmarks

Benchmark scripts and the raw numbers behind running GLM-4.7-Flash on a pair of RTX 3090s, measured against a plain grouped-query model (qwen3.5:27b) as a control. I ran all of this to decide what is actually worth running on my own hardware rather than on rented GPUs, where "does this model earn its place" is a cost question for me.

Full writeup with charts: https://joshgreen-dev.github.io/glm-3090-benchmarks/

## Hardware

- 2x NVIDIA RTX 3090, 24 GB GDDR6X each (48 GB total)
- Ryzen 5 3600X, ASUS PRIME B450-PLUS
- No NVLink, P2P disabled on the consumer cards, measured card-to-card link about 1.4 GB/s
- Driver 580.159.03 / CUDA 13.0, ollama 0.20.5, vLLM 0.23.0 (TP=2)
- Model: `glm-4.7-flash` Q4-class GGUF (arch `glm4moelite`, 29.9B params, roughly 3B active, MLA)

## What is here

- `run_decode.sh` — ollama decode-speed and VRAM sweep across context lengths
- `run_vllm_tp2.sh` — vLLM tensor-parallel (TP=2) single-stream latency and throughput
- `results/` — the raw CSVs behind every number in the writeup

## The short version

GLM-4.7-Flash opens about 2.6x faster than the grouped-query control at short context, then decays with context while the plain model stays flat. They cross around 64K tokens, and past that the boring model is faster. MLA keeps GLM's KV cache small, so it uses less VRAM at long context even while it runs slower there. For a single stream one RTX 3090 wins short context, but both cards under tensor parallel win long context and win throughput about 5 to 6 times over. So the answer is the boring one: it depends on your context length and whether you are chatting or serving.

Numbers, method, and charts are on the [project page](https://joshgreen-dev.github.io/glm-3090-benchmarks/).

## License

MIT. See `LICENSE`.
