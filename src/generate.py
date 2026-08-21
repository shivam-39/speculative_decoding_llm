from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from omegaconf import OmegaConf

try:
    from .benchmark_prompts import COMPACT_BENCHMARK_PROMPTS, load_prompts, load_public_benchmark
    from .decoder import SpeculativeDecoder
except ImportError:  # pragma: no cover - allow execution as a script from src/
    from benchmark_prompts import COMPACT_BENCHMARK_PROMPTS, load_prompts, load_public_benchmark
    from decoder import SpeculativeDecoder

CONFIG = OmegaConf.load(Path(__file__).with_name("config.yaml"))
DEVICE = CONFIG.device
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPTS_FILE = Path(__file__).with_name(CONFIG.prompts_file)


def summarize_metrics(
    results: list[dict],
    draft_model_name: str | None = None,
    target_model_name: str | None = None,
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
    for prompt, pair in by_prompt.items():
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
    }


def get_prompts(args: argparse.Namespace) -> list[str]:
    if args.benchmark:
        return load_public_benchmark(args.benchmark, max_prompts=args.max_prompts)
    if args.prompt_file:
        return load_prompts(args.prompt_file, max_prompts=args.max_prompts)
    if args.use_compact_benchmark:
        return load_prompts(max_prompts=args.max_prompts)
    if PROMPTS_FILE.exists():
        return load_prompts(PROMPTS_FILE, max_prompts=args.max_prompts)
    return list(COMPACT_BENCHMARK_PROMPTS[: args.max_prompts])


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding on a set of prompts.")
    parser.add_argument("--benchmark", choices=["gsm8k", "humaneval", "mt-bench"], help="Load a public benchmark dataset.")
    parser.add_argument("--prompt-file", type=str, help="Path to a file with one prompt per line.")
    parser.add_argument("--use-compact-benchmark", action="store_true", help="Use the built-in compact public-style benchmark set.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit the number of prompts processed.")
    args = parser.parse_args()

    decoder = SpeculativeDecoder(
        target_model_name=CONFIG.target_model_name,
        draft_model_name=CONFIG.draft_model_name,
        device=DEVICE,
    )

    # ensure results directory exists at repository root: ../results
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # timestamp for this run; used in both filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"result_{timestamp}.txt"
    metrics_file = results_dir / f"metrics_{timestamp}.json"

    metrics_list: list[dict] = []
    prompts = get_prompts(args)

    for prompt in prompts:
        # print(f"Prompt: {prompt}\n")

        result1 = decoder.baseline_decode(prompt=prompt, max_tokens=CONFIG.max_tokens)

        result2 = decoder.speculative_decode(
            prompt=prompt,
            max_tokens=CONFIG.max_tokens,
            num_speculative_tokens=CONFIG.num_speculative_tokens,
        )

        # print("Generated text result1 - Baseline Decode:\n")
        # print(result1.text)
        # print("\nMetrics - Baseline Decode:\n")
        # print(json.dumps(result1.metrics.to_dict(), indent=2))
        # print("Generated text result2 - Speculative Decode:\n")
        # print(result2.text)
        # print("\nMetrics - Speculative Decode:\n")
        # print(json.dumps(result2.metrics.to_dict(), indent=2))

        # Append textual results to the run's result file
        try:
            with open(result_file, "a", encoding="utf-8") as rf:
                rf.write(f"Prompt: {prompt}\n\n")
                rf.write("--- Baseline Decode ---\n")
                rf.write(result1.text + "\n\n")
                rf.write("--- Baseline Metrics ---\n")
                rf.write(json.dumps(result1.metrics.to_dict(), indent=2) + "\n\n")
                rf.write("--- Speculative Decode ---\n")
                rf.write(result2.text + "\n\n")
                rf.write("--- Speculative Metrics ---\n")
                rf.write(json.dumps(result2.metrics.to_dict(), indent=2) + "\n")
                rf.write("=" * 80 + "\n\n")
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"Warning: failed to write result file {result_file}: {exc}")

        # collect metrics for JSON output
        metrics_list.append({"prompt": prompt, "decode": "baseline", "metrics": result1.metrics.to_dict()})
        metrics_list.append({"prompt": prompt, "decode": "speculative", "metrics": result2.metrics.to_dict()})

    summary = summarize_metrics(
        metrics_list,
        draft_model_name=CONFIG.draft_model_name,
        target_model_name=CONFIG.target_model_name,
    )

    # write metrics JSON for the run
    try:
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump({"summary": summary, "results": metrics_list}, mf, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Warning: failed to write metrics file {metrics_file}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
