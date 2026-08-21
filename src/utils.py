from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_prompts import COMPACT_BENCHMARK_PROMPTS, load_prompts, load_public_benchmark


def summarize_metrics(
    results: list[dict],
    draft_model_name: str | None = None,
    target_model_name: str | None = None,
    prompt_source: str | None = None,
) -> dict:
    by_prompt: dict[str, dict[str, float]] = {}
    for entry in results:
        prompt = entry["prompt"]
        decode = entry["decode"]
        elapsed = entry["metrics"].get("elapsed_time", 0.0)

        if prompt not in by_prompt:
            by_prompt[prompt] = {}
        by_prompt[prompt][decode] = elapsed

    speedups: list[float] = []
    for pair in by_prompt.values():
        baseline_time = pair.get("baseline")
        speculative_time = pair.get("speculative")
        if baseline_time is None or speculative_time is None:
            continue
        if speculative_time <= 0:
            continue
        speedups.append(baseline_time / speculative_time)

    return {
        "num_prompts": len(by_prompt),
        "avg_speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        "speedups": speedups,
        "draft_model_name": draft_model_name,
        "target_model_name": target_model_name,
        "prompt_source": prompt_source,
    }


def get_prompts(args: argparse.Namespace) -> list[str]:
    if args.benchmark:
        try:
            return load_public_benchmark(args.benchmark, max_prompts=args.max_prompts)
        except (ConnectionError, OSError, RuntimeError, ValueError):
            # Public benchmark downloads are optional; if the Hub is unreachable or the
            # dataset cannot be loaded, fall back to the built-in benchmark prompts.
            if args.max_prompts is not None:
                return list(COMPACT_BENCHMARK_PROMPTS[: args.max_prompts])
            return list(COMPACT_BENCHMARK_PROMPTS)
    if args.prompt_file:
        return load_prompts(args.prompt_file, max_prompts=args.max_prompts)
    if args.use_compact_benchmark:
        return load_prompts(max_prompts=args.max_prompts)

    prompt_file = Path(__file__).with_name("prompts.txt")
    if prompt_file.exists():
        return load_prompts(prompt_file, max_prompts=args.max_prompts)
    return list(COMPACT_BENCHMARK_PROMPTS[: args.max_prompts])
