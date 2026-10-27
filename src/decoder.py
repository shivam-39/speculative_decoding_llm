from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_CACHE_DIR = (Path(__file__).resolve().parents[1] / "model_cache").resolve()


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
    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        device: Optional[str] = None,
        model_cache_dir: Optional[Path] = None,
    ):
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.device == "mps":
            torch.backends.mps.is_available = True

        self.model_cache_dir = Path(model_cache_dir).resolve() if model_cache_dir is not None else DEFAULT_MODEL_CACHE_DIR

        self.draft_model, self.draft_tokenizer = self.initialize_draft_model(draft_model_name)
        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)

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
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.model_cache_dir,
                    **load_kwargs,
                )
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
        tokenizer = self._configure_tokenizer(AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_cache_dir))
        model = self._load_model(model_name, tokenizer, use_lower_precision=(self.device == "cuda"))
        return model, tokenizer

    def initialize_draft_model(self, model_name: str):
        tokenizer = self._configure_tokenizer(AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_cache_dir))
        model = self._load_model(model_name, tokenizer, use_lower_precision=(self.device == "cuda"))
        return model, tokenizer

    def _cached_forward(self, model, token_ids: torch.Tensor, past):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        outputs = model(
            input_ids=token_ids,
            past_key_values=past,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

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

        with torch.inference_mode():
            target_prefill = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            draft_prefill = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        target_past = target_prefill.past_key_values
        draft_past = draft_prefill.past_key_values
        target_next_logits = target_prefill.logits[:, -1, :]
        draft_next_logits = draft_prefill.logits[:, -1, :]

        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        start_time = time.time()

        while input_ids.shape[1] - prompt_length < max_tokens:
            remaining_tokens = max_tokens - (input_ids.shape[1] - prompt_length)
            gamma = min(num_speculative_tokens, remaining_tokens)
            if gamma <= 0:
                break

            draft_token_list: list[int] = []
            with torch.inference_mode():
                for _ in range(gamma):
                    token = int(draft_next_logits.argmax(dim=-1).item())
                    draft_token_list.append(token)
                    step_logits, draft_past = self._cached_forward(
                        self.draft_model,
                        torch.tensor([[token]], dtype=input_ids.dtype, device=self.device),
                        draft_past,
                    )
                    draft_next_logits = step_logits[:, -1, :]

                verify_logits, target_past = self._cached_forward(
                    self.target_model,
                    torch.tensor([draft_token_list], dtype=input_ids.dtype, device=self.device),
                    target_past,
                )

            first_pred = int(target_next_logits.argmax(dim=-1).item())
            if gamma == 1:
                predictions = [first_pred]
            else:
                later_predictions = verify_logits[:, :-1, :].argmax(dim=-1)[0].tolist()
                predictions = [first_pred, *later_predictions]
            bonus_token = int(verify_logits[:, -1, :].argmax(dim=-1).item())

            accepted = 0
            for prediction, drafted in zip(predictions, draft_token_list):
                if prediction != drafted:
                    break
                accepted += 1

            total_draft_tokens_proposed += gamma
            total_draft_tokens_accepted += accepted

            if accepted < gamma:
                committed = draft_token_list[:accepted] + [predictions[accepted]]
            else:
                committed = draft_token_list[:accepted] + [bonus_token]

            if eos_token_id is not None and eos_token_id in committed:
                committed = committed[: committed.index(eos_token_id) + 1]
            committed = committed[:remaining_tokens]
            if not committed:
                break

            kept_draft = 0
            for token in committed:
                if kept_draft < gamma and token == draft_token_list[kept_draft]:
                    kept_draft += 1
                else:
                    break
            to_forward = committed[kept_draft:]
            rejected = gamma - kept_draft
            if rejected:
                target_past.crop(-rejected)
                draft_past.crop(-rejected)

            if to_forward:
                forward_ids = torch.tensor([to_forward], dtype=input_ids.dtype, device=self.device)
                with torch.inference_mode():
                    step_logits, target_past = self._cached_forward(self.target_model, forward_ids, target_past)
                    target_next_logits = step_logits[:, -1, :]
                    step_logits, draft_past = self._cached_forward(self.draft_model, forward_ids, draft_past)
                    draft_next_logits = step_logits[:, -1, :]

            new_tokens = torch.tensor([committed], dtype=input_ids.dtype, device=self.device)
            new_attention = torch.ones(
                (attention_mask.shape[0], new_tokens.shape[1]),
                dtype=attention_mask.dtype,
                device=self.device,
            )
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, new_attention], dim=1)

            if eos_token_id is not None and committed[-1] == eos_token_id:
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

    def baseline_decode(self, prompt: str, max_tokens: int = 100) -> DecodeResult:
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]
        eos_token_id = self.target_tokenizer.eos_token_id

        start_time = time.time()

        with torch.inference_mode():
            output_ids = self.target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.target_tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        elapsed_time = time.time() - start_time
        generated_tokens = output_ids.shape[1] - prompt_length
        metrics = DecodeMetrics(
            generated_tokens=generated_tokens,
            elapsed_time=elapsed_time,
            tokens_per_second=generated_tokens / max(elapsed_time, 1e-8),
            acceptance_rate=0.0,
            draft_tokens_proposed=0,
            draft_tokens_accepted=0,
        )
        return DecodeResult(
            text=self.target_tokenizer.decode(output_ids[0], skip_special_tokens=True),
            token_ids=output_ids[0].tolist(),
            metrics=metrics,
        )
