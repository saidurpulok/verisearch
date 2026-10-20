"""Small fully fine-tuned encoder; logits mapped to correctness probabilities."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verisearch.data.schema import Candidate, Query
from verisearch.verifier.dataset import verifier_text


class Verifier:
    def __init__(self, checkpoint: str | Path, device: str, max_length: int = 384) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device, self.max_length = device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # type: ignore[no-untyped-call]
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        )
        self.seconds = 0.0

    def score(self, query: Query, candidate: Candidate) -> float:
        if query.problem_id != candidate.problem_id:
            raise ValueError("Query/candidate mismatch")
        encoded = self.tokenizer(
            verifier_text(query, candidate),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            score = self.model(**encoded).logits.softmax(-1)[0, 1].item()
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - start
        return float(score)
