# vLLM Structured Tool Calling & Inference Benchmark

A controlled benchmark comparing vLLM 0.9.2 against serial HuggingFace Transformers inference on a single consumer GPU, with structured-output (JSON Schema) compliance testing across 300 tool-calling requests.

## Key Results

| Metric | Serial Transformers | vLLM | Improvement |
|--------|---------------------|------|-------------|
| Throughput (QPS) | 0.80 | 43.57 | **~55×** |
| Generation tokens/s | ~58 | ~3,043 | ~52× |
| 50-request wall time | 62.8s | 1.15s | ~55× |

| Schema Compliance | Raw | response\_format | guided\_json |
|--------------------|-----|------------------|-------------|
| Complex schema (create\_order) | 42% | 42% | **100%** |
| Simple schema (stock\_quote) | 60% | 68% | **100%** |

All numbers are reproducible from [`outputs/throughput_results.json`](outputs/throughput_results.json) and [`outputs/function_call_results.json`](outputs/function_call_results.json).

## Tech Stack

- **Inference**: vLLM 0.9.2, PyTorch 2.7 (CUDA 12.6)
- **Model**: Qwen2-0.5B-Instruct
- **Structured Output**: vLLM guided decoding (guided\_json, guided\_regex, guided\_choice)
- **Validation**: `jsonschema.validate()` (full JSON Schema compliance)
- **Visualisation**: matplotlib

## Environment

- **GPU**: NVIDIA RTX 4060 8 GB (single GPU, controlled local benchmark)
- **OS**: WSL2 Ubuntu
- **vLLM config**: `gpu_memory_utilization=0.6` (~5 GB / 8 GB)

## Quick Start

```bash
pip install -r requirements.txt

# Start vLLM server (OpenAI-compatible)
bash start_server.sh

# Run throughput benchmark
python src/bench_throughput.py

# Run structured tool-calling benchmark
python src/demo_function_call.py

# Results saved to outputs/
```

## Key Findings

1. **vLLM provides ~55× throughput improvement** over serial Transformers on a single consumer GPU (0.80 → 43.57 QPS)
2. **Guided JSON constrained decoding raises schema compliance from 42% to 100%** — the gap is in semantic field validity (enum, regex, range), not JSON syntax
3. **Constrained decoding adds negligible latency** (±4% in this small-model benchmark)
4. **Schema compliance ≠ model accuracy** — the 0.5B model may pick wrong values even at 100% schema-valid output

## Project Structure

```
├── src/
│ ├── bench_throughput.py # Throughput benchmark (294 lines)
│ ├── demo_function_call.py # Tool-calling structured output (428 lines)
│ ├── demo_guided_json.py # Guided JSON schema decoding
│ ├── demo_guided_choice.py # Guided choice decoding
│ ├── demo_guided_regex.py # Guided regex decoding
│ ├── demo_response_format.py # Response format enforcement
│ └── start_server.sh # vLLM server launch script
├── outputs/
│ ├── throughput_results.json # Benchmark numbers
│ ├── function_call_results.json # Schema compliance data
│ └── throughput_comparison.png # Bar chart visualisation
├── ARCHITECTURE.md
├── USAGE_GUIDE.md
└── requirements.txt
```

## License

MIT
