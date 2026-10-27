from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency for public benchmark loading
    load_dataset = None

COMPACT_BENCHMARK_PROMPTS: list[str] = [
    "Explain recursion in one sentence.",
    "Write a haiku about a lighthouse.",
    "What is the difference between a list and a tuple in Python?",
    "Write a short story about a robot learning to feel emotions for the first time.",
    "A train leaves Boston at 8:15 AM traveling at 72 km/h. Another train leaves New York at 9:00 AM traveling at 88 km/h. If the cities are 350 km apart, when do they meet?",
    "Summarize this in two bullet points: 'The team shipped a faster inference engine that reduced latency by 30%, cut memory usage, and improved throughput while keeping accuracy stable.'",
    "Rewrite this sentence to sound more concise and professional: 'The project team is currently in the process of evaluating several different approaches to improve the speed of the system, and they are doing so with a high degree of caution.'",
    "Provide a Python function that returns the median of a list of numbers without using the statistics module.",
    "Explain why the sky appears blue to a 10-year-old using a simple analogy.",
    "Create a SQL query that returns the top 5 customers by total spend from a table called orders with columns customer_id, customer_name, amount, and order_date.",
    "Describe the trade-offs between using a smaller draft model and a larger target model in speculative decoding, and explain when one is preferable to the other.",
    "Give three examples of common bugs in Python and show one-line fixes for each.",
    "Draft a short meeting agenda for a team planning a model benchmarking sprint, including goals, constraints, and success criteria.",
    "Write a JSON schema for a task-tracking application with title, assignee, due_date, status, priority, and tags fields, and include example values.",
    "A bakery sells muffins for $3 each and cookies for $2 each. If a customer buys 4 muffins and 6 cookies, how much change do they get from $30?",
    "Explain in plain English how a language model predicts the next token during generation, including the role of context, tokenization, and probability distribution.",
    "Write a one-paragraph product launch email for a solar-powered backpack startup, aimed at eco-conscious commuters, and include a clear CTA for early access signup.",
]

PUBLIC_BENCHMARKS: dict[str, dict[str, str]] = {
    "gsm8k": {"name": "openai/gsm8k", "config": "main", "split": "test", "field": "question"},
    "humaneval": {"name": "openai/openai_humaneval", "split": "test", "field": "prompt"},
    "mt-bench": {"name": "HuggingFaceH4/mt_bench_prompts", "split": "train", "field": "prompt"},
}


def _normalize_prompt(prompt: object) -> str:
    if prompt is None:
        raise ValueError("Prompt cannot be None")
    text = str(prompt).strip()
    if not text:
        raise ValueError("Prompt cannot be empty")
    return text


def load_prompts(path: str | Path | None = None, max_prompts: int | None = None) -> list[str]:
    if path is not None:
        prompt_file = Path(path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        prompts: list[str] = []
        with prompt_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                prompt = raw_line.strip()
                if prompt:
                    prompts.append(prompt)
        if max_prompts is not None:
            prompts = prompts[:max_prompts]
        return prompts

    prompts = list(COMPACT_BENCHMARK_PROMPTS)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def load_public_benchmark(benchmark_name: str, max_prompts: int | None = None) -> list[str]:
    normalized_name = benchmark_name.lower().replace("_", "-")
    if normalized_name not in PUBLIC_BENCHMARKS:
        available = ", ".join(sorted(PUBLIC_BENCHMARKS))
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {available}")

    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is required to load public benchmark prompts. Install it with: uv add datasets")

    dataset_config = PUBLIC_BENCHMARKS[normalized_name]
    config_name = dataset_config.get("config")
    if config_name:
        dataset = load_dataset(dataset_config["name"], config_name, split=dataset_config["split"])
    else:
        dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
    prompts: list[str] = []
    for example in dataset:
        value = example.get(dataset_config["field"])
        if isinstance(value, list):
            prompts.extend(_normalize_prompt(item) for item in value)
        else:
            prompts.append(_normalize_prompt(value))
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts
