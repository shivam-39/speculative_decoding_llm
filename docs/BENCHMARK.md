# Benchmark Summary

[TODO]: run of full dataset and other models.

This document summarizes the sample benchmarks.


## Models

- Draft model: `EleutherAI/pythia-160m-deduped`
- Target model: `EleutherAI/pythia-1.4b-deduped`

## Summary table

| Draft model name | Target model name | Benchmark type | Number of prompts | Avg speedup |
| --- | --- | --- | ---: | ---: |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | gsm8k | 4 | 0.9020x |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | humaneval | 3 | 0.9638x |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | mt-bench | 2 | 1.3829x |
