from pathlib import Path

from src.benchmark_prompts import COMPACT_BENCHMARK_PROMPTS, load_prompts
from src.utils import summarize_metrics


def test_compact_benchmark_has_variable_prompt_lengths():
    prompts = COMPACT_BENCHMARK_PROMPTS
    assert len(prompts) >= 12
    lengths = [len(p.split()) for p in prompts]
    assert max(lengths) > min(lengths) * 5


def test_load_prompts_truncates_and_filters_empty_lines(tmp_path):
    prompt_file = tmp_path / "custom_prompts.txt"
    prompt_file.write_text("\n\nalpha\n\n beta \n gamma delta\n", encoding="utf-8")

    prompts = load_prompts(prompt_file)

    assert prompts == ["alpha", "beta", "gamma delta"]
    assert load_prompts(prompt_file, max_prompts=2) == ["alpha", "beta"]


def test_summarize_metrics_computes_avg_speedup():
    results = [
        {"prompt": "p1", "decode": "baseline", "metrics": {"elapsed_time": 10.0}},
        {"prompt": "p1", "decode": "speculative", "metrics": {"elapsed_time": 5.0}},
        {"prompt": "p2", "decode": "baseline", "metrics": {"elapsed_time": 8.0}},
        {"prompt": "p2", "decode": "speculative", "metrics": {"elapsed_time": 4.0}},
    ]

    summary = summarize_metrics(results, draft_model_name="draft", target_model_name="target")

    assert summary["num_prompts"] == 2
    assert summary["avg_speedup"] == 2.0
    assert summary["draft_model_name"] == "draft"
    assert summary["target_model_name"] == "target"
