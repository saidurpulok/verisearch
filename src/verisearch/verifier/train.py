"""Inspectable full fine-tuning loop with validation-only checkpoint selection."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verisearch.config import VerifierConfig
from verisearch.data.schema import LabeledCandidate
from verisearch.generation.generator import seed_everything
from verisearch.io import digest, file_sha256, provenance, write_json
from verisearch.verifier.dataset import balance, validate_partitions, verifier_text


def train_verifier(
    config: VerifierConfig,
    train: list[LabeledCandidate],
    validation: list[LabeledCandidate],
    output: Path,
) -> dict[str, Any]:
    validate_partitions(train, validation)
    if len({r.correct for r in train}) < 2:
        raise ValueError("Training needs both correct and incorrect candidates")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seed_everything(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        config.checkpoint, revision=config.revision
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        revision=config.revision,
        num_labels=2,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    def batches(rows: list[LabeledCandidate], order: list[int]) -> Any:
        for start in range(0, len(order), config.batch_size):
            batch = [rows[index] for index in order[start : start + config.batch_size]]
            encoded = tokenizer(
                [verifier_text(r.query, r.candidate) for r in batch],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(config.device)
            labels = torch.tensor([int(r.correct) for r in batch], device=config.device)
            yield encoded, labels

    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    truncation = {}
    for name, rows in (("train", train), ("validation", validation)):
        lengths = [len(tokenizer.encode(verifier_text(r.query, r.candidate))) for r in rows]
        truncation[name] = {
            "fraction_truncated": float(np.mean(np.array(lengths) > config.max_length)),
            "max_input_tokens": max(lengths),
        }
    for epoch in range(config.epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(config.seed + epoch).shuffle(order)
        total = 0.0
        for encoded, labels in batches(train, order):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for encoded, labels in batches(validation, list(range(len(validation)))):
                validation_total += float(model(**encoded, labels=labels).loss) * len(labels)
        validation_loss = validation_total / len(validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train),
                "validation_loss": validation_loss,
            }
        )
        print(history[-1], flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
    metadata = {
        **provenance(),
        "config": asdict(config),
        "base_resolved_revision": getattr(model.config, "_commit_hash", None),
        "train_balance": balance(train),
        "validation_balance": balance(validation),
        "train_problem_ids": sorted({r.query.problem_id for r in train}),
        "validation_problem_ids": sorted({r.query.problem_id for r in validation}),
        "train_sha256": digest([r.to_dict() for r in train]),
        "validation_sha256": digest([r.to_dict() for r in validation]),
        "history": history,
        "selection": "minimum validation cross-entropy",
        "truncation": truncation,
        "device": config.device,
        "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_path": str(output),
        "weights_sha256": file_sha256(output / "model.safetensors"),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if config.device == "cuda"
        else None,
    }
    write_json(output / "metadata.json", metadata)
    return metadata
