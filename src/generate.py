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
    
    result = decoder.speculative_decode(
            prompt=DEFAULT_PROMPTS,
            max_tokens=MAX_TOKENS,
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
    )

    print("Generated text:\n")
    print(result.text)
    print("\nMetrics:\n")
    print(json.dumps(result.metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
