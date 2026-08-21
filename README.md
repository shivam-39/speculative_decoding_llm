# Speculative Decoding
** (Work-In-Progress -> Adding more features)

This project focuses on optimizing speculative decoding for faster language model inference. 
The method uses a smaller, faster draft model to generate candidate tokens, which are then verified by a larger target model in a single forward pass.

## Project Layout

```text
speculative_decoding/
├── README.md
├── pyproject.toml
└── src/
    ├── config.yaml
    ├── decoder.py
    └── generate.py
```

## Setup

Create a virtual environment and install the dependencies with `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Run 

Speculative decoding:

```bash
uv run python src/generate.py
```
