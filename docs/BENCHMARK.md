# Benchmark Summary

[TODO]: run of full dataset and other models.

This document summarizes the sample benchmarks.


## Models

- Draft model: `EleutherAI/pythia-160m-deduped`
- Target model: `EleutherAI/pythia-1.4b-deduped`

## Summary table

| Draft model name | Target model name | Benchmark type | Number of prompts | Avg speedup |
| --- | --- | --- | ---: | ---: |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | gsm8k | 2 | 1.4617x |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | humaneval | 2 | 1.2903x |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | mt-bench | 2 | 1.1132x |
