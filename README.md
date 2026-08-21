# Speculative Decoding
** (Work-In-Progress -> Adding more features)

This project focuses on optimizing speculative decoding for faster language model inference. 
The method uses a smaller, faster draft model to generate candidate tokens, which are then verified by a larger target model in a single forward pass.  
Summary of a current benchmark can be found in [docs/BENCHMARK.md](docs/BENCHMARK.md) 

## Repo layout

```text

|── README.md
├── main.py
├── pyproject.toml
├── configs/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── benchmark_prompts.py
│   ├── decoder.py
│   ├── main.py
│   ├── prompts.txt
│   └── utils.py
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

## Run from the repo root

```bash
python main.py
```

Or pass optional custom config and prompt files using:

```bash
python main.py ----config /path/to/config.yaml --prompt-file /path/to/prompts.txt
```

Each prompt should be on its own line, and the generator will benchmark baseline decoding and speculative decoding for each entry.

# Benachmarks

## Run compact mixed-length benchmark

The repo includes a built-in compact benchmark with short, medium, and longer prompts of varying complexity.

```bash
python main.py --use-compact-benchmark --max-prompts 12
```

This loads the prompts from [src/benchmark_prompts.py](src/benchmark_prompts.py) and processes only the first N prompts if you limit them with `--max-prompts`.

## Run a public benchmark dataset

You can also benchmark against a public dataset directly. Supported choices are:

- `gsm8k`: a dataset of grade-school math word problems used to evaluate numerical reasoning.
- `humaneval`: a collection of programming tasks with reference unit tests for evaluating code generation.
- `mt-bench`: a multi-task instruction-following benchmark used to compare model performance on diverse prompts.

```bash
python main.py --benchmark gsm8k --max-prompts 20
python main.py --benchmark humaneval --max-prompts 20
python main.py --benchmark mt-bench --max-prompts 20
```

## Output

The script writes:
- a text file under the `results/` directory containing prompt-by-prompt generated text and metrics
- a JSON metrics file with one entry per prompt and decode mode, including the aggregate summary with average speedup and model names

