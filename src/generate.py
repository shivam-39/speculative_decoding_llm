from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    for prompt in PROMPTS:
        print(f"Prompt: {prompt}\n")
        
        result1 = decoder.baseline_decode(prompt=prompt, max_tokens=CONFIG.max_tokens)
    
        result2 = decoder.speculative_decode(
                prompt=prompt,
                max_tokens=CONFIG.max_tokens,
                num_speculative_tokens=CONFIG.num_speculative_tokens,
        )

        print("Generated text result1 - Baseline Decode:\n")
        print(result1.text)
        print("\nMetrics - Baseline Decode:\n")
        print(json.dumps(result1.metrics.to_dict(), indent=2))
        print("Generated text result2 - Speculative Decode:\n")
        print(result2.text)
        print("\nMetrics - Speculative Decode:\n")
        print(json.dumps(result2.metrics.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
