from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .decoder import SpeculativeDecoder
from .utils import get_prompts, summarize_metrics

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
CONFIG = OmegaConf.load(CONFIG_PATH)
DEVICE = CONFIG.device
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding on a set of prompts.")
    parser.add_argument("--benchmark", choices=["gsm8k", "humaneval", "mt-bench"], help="Load a public benchmark dataset.")
    parser.add_argument("--prompt-file", type=str, help="Path to a file with one prompt per line.")
    parser.add_argument("--use-compact-benchmark", action="store_true", help="Use the built-in compact public-style benchmark set.")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit the number of prompts processed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    decoder = SpeculativeDecoder(
        target_model_name=CONFIG.target_model_name,
        draft_model_name=CONFIG.draft_model_name,
        device=DEVICE,
    )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"result_{timestamp}.txt"
    metrics_file = results_dir / f"metrics_{timestamp}.json"

    metrics_list: list[dict] = []
    prompts = get_prompts(args)

    for prompt in prompts:
        result1 = decoder.baseline_decode(prompt=prompt, max_tokens=CONFIG.max_tokens)
        result2 = decoder.speculative_decode(
            prompt=prompt,
            max_tokens=CONFIG.max_tokens,
            num_speculative_tokens=CONFIG.num_speculative_tokens,
        )

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

        metrics_list.append({"prompt": prompt, "decode": "baseline", "metrics": result1.metrics.to_dict()})
        metrics_list.append({"prompt": prompt, "decode": "speculative", "metrics": result2.metrics.to_dict()})

    summary = summarize_metrics(
        metrics_list,
        draft_model_name=CONFIG.draft_model_name,
        target_model_name=CONFIG.target_model_name,
    )

    try:
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump({"summary": summary, "results": metrics_list}, mf, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Warning: failed to write metrics file {metrics_file}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
