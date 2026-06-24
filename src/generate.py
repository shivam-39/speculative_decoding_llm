from __future__ import annotations

import argparse
import json

from decoder import DEFAULT_PROMPTS, DEFAULT_DRAFT_MODEL, DEVICE, DEFAULT_TARGET_MODEL, MAX_TOKENS, NUM_SPECULATIVE_TOKENS, SpeculativeDecoder

def main() -> int:
    decoder = SpeculativeDecoder(
        target_model_name=DEFAULT_TARGET_MODEL,
        draft_model_name=DEFAULT_DRAFT_MODEL,
        device=DEVICE,
    )

    for prompt in DEFAULT_PROMPTS:
        print(f"Prompt: {prompt}\n")
        
        # result1 = decoder.baseline_decode(prompt=DEFAULT_PROMPTS, max_tokens=MAX_TOKENS)
    
        result2 = decoder.speculative_decode(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        )

        # print("Generated text result1 - Baseline Decode\n")
        # print(result1.text)
        # print("\nMetrics:\n")
        # print(json.dumps(result1.metrics.to_dict(), indent=2))
        print("Generated text result2 - Speculative Decode\n")
        print(result2.text)
        print("\nMetrics:\n")
        print(json.dumps(result2.metrics.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
