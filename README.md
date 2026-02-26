# mlc-graph-cli

CLI for extracting knowledge graph triples from product reviews using local LLM inference (MLC-LLM) and fine-tuning a small model to match larger model quality.

## Pipeline

1. **Extract triples (baseline)** — Run Ministral-3B locally via MLC-LLM to extract `(subject, predicate, object)` triples from review text
2. **Extract finetune triples** — Use a larger cloud model (Gemini via OpenRouter) to produce higher-quality triples as training targets
3. **Fine-tune** — SFT Ministral-3B on the cloud-extracted triples so the small model learns to produce better extractions

## Project Structure

```
main.py          # CLI entrypoint — interactive prompts + argparse
engine.py        # Engine (local MLC-LLM) and CloudEngine (OpenRouter) wrappers
utils.py         # Shared extraction pipeline (resume, concurrency, retries)
finetune.py      # SFT training with preset configs
graph.py         # Knowledge graph data structures
normalize.py     # Triple normalization (WIP)
PGraphRAG/       # Input datasets (downloaded via scripts/download.sh)
train/           # Training data (.jsonl) for fine-tuning
models/          # Model weights (HF + MLC-compiled)
wheels/          # MLC-LLM and TVM wheel files
```

## Setup

Requires Python 3.13.

```bash
# Install dependencies (uv)
uv sync

# Download datasets and base model
bash scripts/download.sh

# Copy .env and add your OpenRouter key (only needed for extract-finetune)
cp .env.example .env
```

MLC-LLM and TVM wheels go in `wheels/` — update paths in `pyproject.toml` if needed.
The MLC-compiled model goes in `models/`.

## Usage

Run interactively (no args, prompts for everything):

```bash
python main.py
```

Or fully scripted:

```bash
# Extract triples locally with Ministral-3B
python main.py --task extract-triples --device metal --extract-file amazon_train.json

# Extract higher-quality triples via cloud model
python main.py --task extract-finetune --extract-file amazon_train.json

# Fine-tune Ministral-3B on cloud-extracted triples
python main.py --task finetune --train-config balanced --dataset train/data.jsonl
```

### CLI Arguments

| Arg | Description |
|-----|-------------|
| `--task` | `extract-triples`, `extract-finetune`, `finetune`, `normalize-data`, `query` |
| `--device` | `cuda`, `metal`, `cpu` |
| `--extract-file` | Input JSON file name (from `PGraphRAG/`) |
| `--output-path` | Output JSONL path |
| `--model-path` | Path to model weights |
| `--dataset` | Training data JSONL path (for finetune) |
| `--concurrency` | Number of concurrent extraction tasks |
| `--train-config` | `conservative`, `balanced`, `aggressive` |

All arguments are optional — the CLI will prompt interactively for anything not provided.
