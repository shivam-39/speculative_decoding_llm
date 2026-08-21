from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from omegaconf import OmegaConf

from decoder import SpeculativeDecoder

CONFIG = OmegaConf.load(Path(__file__).with_name("config.yaml"))
DEVICE = CONFIG.device
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPTS = [
    "The future of Artificial Intelligence is",
    "Write a short story about a robot learning to feel emotions:",
    "Write the lyrics to the song 'Happy Birthday'.",
]


def main() -> int:
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

    for prompt in PROMPTS:
        print(f"Prompt: {prompt}\n")

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

    # write metrics JSON for the run
    try:
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump(metrics_list, mf, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Warning: failed to write metrics file {metrics_file}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
