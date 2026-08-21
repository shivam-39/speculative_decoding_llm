# Speculative Decoding
** (Work-In-Progress -> Adding more features)

This project focuses on optimizing speculative decoding for faster language model inference. 
The method uses a smaller, faster draft model to generate candidate tokens, which are then verified by a larger target model in a single forward pass.

## Repo layout

```text
.
├── README.md
├── pyproject.toml
├── src/
│   ├── config.yaml
│   ├── decoder.py
│   ├── generate.py
│   └── prompts.txt
├── results/
└── model_cache/
```

## Setup

Create a virtual environment and install the dependencies with `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Run

```bash
python src/generate.py
```

This reads model settings from [src/config.yaml](src/config.yaml) and prompt entries from [src/prompts.txt](src/prompts.txt). The config includes the prompt filename via `prompts_file`, and the generator loads that file automatically.

