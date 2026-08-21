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

## Run default prompt set

Add few prompts in the `prompts.txt` file and run:

```bash
python src/generate.py
```

This reads model settings from [src/config.yaml](src/config.yaml) and prompt entries from [src/prompts.txt](src/prompts.txt). The config includes the prompt filename via `prompts_file`, and the generator loads that file automatically.

## Run compact mixed-length benchmark

The repo now includes a built-in compact benchmark with short, medium, and longer prompts of varying complexity.

```bash
python src/generate.py --use-compact-benchmark --max-prompts 12
```

This loads the prompts from [src/benchmark_prompts.py](src/benchmark_prompts.py) and processes only the first N prompts if you limit them with `--max-prompts`.

## Run a public benchmark dataset

You can also benchmark against a public dataset directly. Supported choices are `gsm8k`, `humaneval`, and `mt-bench`.

```bash
python src/generate.py --benchmark gsm8k --max-prompts 20
python src/generate.py --benchmark humaneval --max-prompts 20
python src/generate.py --benchmark mt-bench --max-prompts 20
```

## Run from a custom prompt file

```bash
python src/generate.py --prompt-file /path/to/prompts.txt --max-prompts 50
```

Each prompt should be on its own line, and the generator will benchmark baseline decoding and speculative decoding for each entry.

## Output

The script writes:
- a text file under the `results/` directory containing prompt-by-prompt generated text and metrics
- a JSON metrics file with one entry per prompt and decode mode

