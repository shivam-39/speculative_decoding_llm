from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TARGET_MODEL = "EleutherAI/pythia-1.4b-deduped"
DEFAULT_DRAFT_MODEL = "EleutherAI/pythia-160m-deduped"
ALTERNATIVE_TARGET_MODEL = "EleutherAI/pythia-410m-deduped"
DEFAULT_PROMPTS = [
    "The future of Artificial Intelligence is",
    "Write a short story about a robot learning to feel emotions:",
    "Write the lyrics to the song 'Happy Birthday'.",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 100
NUM_SPECULATIVE_TOKENS = 15


@dataclass
class DecodeMetrics:
    generated_tokens: int
    elapsed_time: float
    tokens_per_second: float
    acceptance_rate: float
    draft_tokens_proposed: int
    draft_tokens_accepted: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecodeResult:
    text: str
    token_ids: List[int]
    metrics: DecodeMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "metrics": self.metrics.to_dict(),
        }


class SpeculativeDecoder:
    def __init__(self, target_model_name: str, draft_model_name: str, device: Optional[str] = None):
        self.device = resolve_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._last_target_predictions: Optional[torch.Tensor] = None

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)
        self.draft_model, self.draft_tokenizer = self.initialize_draft_model(draft_model_name)

        if self.target_tokenizer.get_vocab() != self.draft_tokenizer.get_vocab():
            raise ValueError("Target and draft tokenizers must use the same vocabulary.")

    def _configure_tokenizer(self, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer must expose an eos_token or unk_token to use as pad_token.")
        tokenizer.padding_side = "left"
        return tokenizer

    def _load_model(self, model_name: str, tokenizer, *, use_lower_precision: bool):
        base_kwargs: Dict[str, Any] = {}
        if use_lower_precision:
            base_kwargs["torch_dtype"] = self.dtype

        load_attempts = [
            {**base_kwargs, "low_cpu_mem_usage": True, "attn_implementation": "sdpa"},
            {**base_kwargs, "low_cpu_mem_usage": True},
            dict(base_kwargs),
        ]

        model = None
        last_error = None
        for load_kwargs in load_attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                break
            except (ImportError, TypeError, ValueError) as exc:
                last_error = exc

        if model is None:
            raise RuntimeError(f"Failed to load model {model_name}") from last_error

        model.to(self.device)
        model.eval()
        model.config.use_cache = True
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        return model

    def initialize_target_model(self, model_name: str):
        tokenizer = self._configure_tokenizer(AutoTokenizer.from_pretrained(model_name))
        model = self._load_model(model_name, tokenizer, use_lower_precision=(self.device == "cuda"))
        return model, tokenizer

    def initialize_draft_model(self, model_name: str):
        tokenizer = self._configure_tokenizer(AutoTokenizer.from_pretrained(model_name))
        model = self._load_model(model_name, tokenizer, use_lower_precision=(self.device == "cuda"))
        return model, tokenizer

    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_speculative_tokens: int = 10,
    ) -> torch.Tensor:
        if num_speculative_tokens <= 0:
            return input_ids.new_empty((input_ids.shape[0], 0))

        with torch.inference_mode():
            generated = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_speculative_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.draft_tokenizer.pad_token_id,
                eos_token_id=self.draft_tokenizer.eos_token_id,
            )

        return generated[:, input_ids.shape[1] :]

    def verify_tokens_vectorized(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[List[int], int]:
        draft_length = draft_tokens.shape[1]
        if draft_length == 0:
            self._last_target_predictions = input_ids.new_empty((0,))
            return [], 0

        combined_input_ids = torch.cat([input_ids, draft_tokens], dim=1)
        combined_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], draft_length),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        )

        with torch.inference_mode():
            outputs = self.target_model(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                use_cache=True,
            )

        start_idx = input_ids.shape[1] - 1
        verification_logits = outputs.logits[:, start_idx : start_idx + draft_length + 1, :]
        target_predictions = verification_logits.argmax(dim=-1)
        draft_matches = target_predictions[:, :draft_length].eq(draft_tokens)

        mismatch_locations = (~draft_matches[0]).nonzero(as_tuple=False)
        accepted_position = draft_length if mismatch_locations.numel() == 0 else mismatch_locations[0, 0].item()
        accepted_tokens = draft_tokens[0, :accepted_position].tolist()
        self._last_target_predictions = target_predictions[0].detach()
        return accepted_tokens, accepted_position

    def speculative_decode(
        self,
        prompt: str,
        max_tokens: int = 100,
        num_speculative_tokens: int = 15,
    ) -> DecodeResult:
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]
        eos_token_id = self.target_tokenizer.eos_token_id

        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        start_time = time.time()

        while input_ids.shape[1] - prompt_length < max_tokens:
            remaining_tokens = max_tokens - (input_ids.shape[1] - prompt_length)
            speculative_window = min(num_speculative_tokens, remaining_tokens)

            draft_tokens = self.generate_draft_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_speculative_tokens=speculative_window,
            )

            if draft_tokens.shape[1] == 0:
                with torch.inference_mode():
                    outputs = self.target_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                candidate_tokens = [int(outputs.logits[:, -1, :].argmax(dim=-1).item())]
            else:
                total_draft_tokens_proposed += draft_tokens.shape[1]
                accepted_tokens, accepted_position = self.verify_tokens_vectorized(
                    input_ids=input_ids,
                    draft_tokens=draft_tokens,
                    attention_mask=attention_mask,
                )
                total_draft_tokens_accepted += accepted_position

                candidate_tokens = list(accepted_tokens)
                if accepted_position < draft_tokens.shape[1]:
                    candidate_tokens.append(int(self._last_target_predictions[accepted_position].item()))
                elif len(candidate_tokens) < remaining_tokens:
                    candidate_tokens.append(int(self._last_target_predictions[draft_tokens.shape[1]].item()))

            if not candidate_tokens:
                break

            candidate_tokens = candidate_tokens[:remaining_tokens]
            if eos_token_id is not None and eos_token_id in candidate_tokens:
                candidate_tokens = candidate_tokens[: candidate_tokens.index(eos_token_id) + 1]

            new_tokens = torch.tensor([candidate_tokens], dtype=input_ids.dtype, device=self.device)
            new_attention = torch.ones(
                (attention_mask.shape[0], new_tokens.shape[1]),
                dtype=attention_mask.dtype,
                device=self.device,
            )
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, new_attention], dim=1)

            if eos_token_id is not None and candidate_tokens[-1] == eos_token_id:
                break

        elapsed_time = time.time() - start_time
        generated_tokens = input_ids.shape[1] - prompt_length
        acceptance_rate = (
            total_draft_tokens_accepted / total_draft_tokens_proposed
            if total_draft_tokens_proposed > 0
            else 0.0
        )
        metrics = DecodeMetrics(
            generated_tokens=generated_tokens,
            elapsed_time=elapsed_time,
            tokens_per_second=generated_tokens / max(elapsed_time, 1e-8),
            acceptance_rate=acceptance_rate,
            draft_tokens_proposed=total_draft_tokens_proposed,
            draft_tokens_accepted=total_draft_tokens_accepted,
        )
        return DecodeResult(
            text=self.target_tokenizer.decode(input_ids[0], skip_special_tokens=True),
            token_ids=input_ids[0].tolist(),
            metrics=metrics,
        )
