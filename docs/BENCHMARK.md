# Benchmark Summary

[TODO]: eval with different params to find the optimal settings.

This document summarizes the sample benchmarks.

## Summary table


| Draft model name                 | Target model name                | Acceptance Rate | Benchmark type | Avg speedup |
| -------------------------------- | -------------------------------- | -------------- | -------------- | ----------- |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | gsm8k          | 0.803          | 1.4617x     |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | humaneval      | 0.756          | 1.2903x     |
| `EleutherAI/pythia-160m-deduped` | `EleutherAI/pythia-1.4b-deduped` | mt-bench       | 0.855          | 1.1132x     |
| `google/gemma-2-2b-it`           | `google/gemma-2-9b-it`           | gsm8k          | 0.751          | 1.7191x     |
| `google/gemma-2-2b-it`           | `google/gemma-2-9b-it`           | humaneval      | 0.823          | 1.8114x     |
| `google/gemma-2-2b-it`           | `google/gemma-2-9b-it`           | mt-bench       | 0.337          | 0.5995x     |
| `Qwen/Qwen2.5-0.5B-Instruct`     | `Qwen/Qwen2.5-32B-Instruct`      | gsm8k          | 0.795          | 1.6818x     |
| `Qwen/Qwen2.5-0.5B-Instruct`     | `Qwen/Qwen2.5-32B-Instruct`      | humaneval      | 0.749          | 1.1132x     |
| `Qwen/Qwen2.5-0.5B-Instruct`     | `Qwen/Qwen2.5-32B-Instruct`      | mt-bench       | 0.443          | 1.1212x     |
