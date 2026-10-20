"""Local sequential Transformers generation with explicit per-call token caps."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verisearch.config import GenerationConfig
from verisearch.data.schema import Candidate, Query
from verisearch.generation.answer_extraction import extract_answer

SYSTEM_PROMPT = "Please reason step by step, and put your final numeric answer within \\boxed{}."


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def candidate_seed(seed: int, problem_id: str, candidate_id: int) -> int:
    # Independent of traversal order, output directory and Python hash randomization.
    return int(hashlib.sha256(f"{seed}:{problem_id}:{candidate_id}".encode()).hexdigest()[:8], 16)


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; never silently substitute CPU")
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            config.checkpoint, revision=config.revision
        )
        # Transformers' decorated .to() has incomplete static typing.
        self.model: Any = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                config.checkpoint,
                revision=config.revision,
                torch_dtype=torch.bfloat16 if config.device == "cuda" else torch.float32,
                attn_implementation="sdpa",
            ),
        )
        self.model.to(config.device)
        self.model.eval()
        self.resolved_revision: str | None = getattr(self.model.config, "_commit_hash", None)

    def hardware(self) -> dict[str, Any]:
        return {
            "device": self.config.device,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if self.config.device == "cuda" else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
            if self.config.device == "cuda"
            else None,
        }

    def generate(
        self,
        query: Query,
        candidate_id: int,
        seed: int,
        token_cap: int,
        *,
        greedy: bool = False,
        temperature: float | None = None,
    ) -> Candidate:
        if token_cap < 1:
            raise ValueError("Token cap must be positive")
        cap = min(token_cap, self.config.max_new_tokens)
        actual_seed = candidate_seed(seed, query.problem_id, candidate_id)
        seed_everything(actual_seed)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query.question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": cap,
            "do_sample": not greedy,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if not greedy:
            kwargs.update(
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - start
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Count actual new IDs, including EOS if emitted, never re-tokenize text.
        return Candidate(
            query.problem_id,
            candidate_id,
            text,
            extract_answer(text),
            len(generated),
            latency,
            actual_seed,
            self.config.checkpoint,
            {
                **asdict(self.config),
                "temperature": temperature or self.config.temperature,
                "greedy": greedy,
                "actual_max_new_tokens": cap,
                "resolved_revision": self.resolved_revision,
                "prompt": prompt,
            },
        )

    def stream(
        self,
        query: Query,
        seed: int,
        budget: int,
        max_candidates: int | None = None,
        *,
        temperature: float | None = None,
    ) -> Iterator[Candidate]:
        remaining = budget
        for index in range(max_candidates or self.config.candidates):
            if remaining <= 0:
                break
            candidate = self.generate(query, index, seed, remaining, temperature=temperature)
            remaining -= candidate.tokens
            yield candidate
