"""One saved YAML defines a complete experiment; unknown keys are rejected."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "openai/gsm8k"
    revision: str = "main"
    split_seed: int = 42
    train_problems: int = 128
    validation_problems: int = 32
    test_problems: int = 48


@dataclass(frozen=True)
class GenerationConfig:
    checkpoint: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    revision: str = "main"
    device: str = "cuda"
    max_new_tokens: int = 256
    candidates: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    seeds: list[int] = field(default_factory=lambda: [17, 29])
    temperatures: list[float] = field(default_factory=lambda: [0.6, 1.0])
    budgets: list[int] = field(default_factory=lambda: [256, 512, 1024])


@dataclass(frozen=True)
class VerifierConfig:
    checkpoint: str = "distilbert/distilroberta-base"
    revision: str = "main"
    device: str = "cuda"
    max_length: int = 384
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    seed: int = 101
    fractions: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 2026
    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 1.0])
    # Select on validation: maximize accuracy - this * mean tokens / budget.
    token_penalty: float = 0.02


@dataclass(frozen=True)
class Config:
    name: str = "pilot"
    output: str = "runs/pilot"
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if (
            min(self.data.train_problems, self.data.validation_problems, self.data.test_problems)
            < 1
        ):
            raise ValueError("Every partition must contain problems")
        g, v, e = self.generation, self.verifier, self.evaluation
        if g.max_new_tokens < 1 or g.candidates < 1 or not g.budgets or min(g.budgets) < 1:
            raise ValueError("Positive generation limits required")
        if not g.seeds or len(set(g.seeds)) != len(g.seeds):
            raise ValueError("Distinct nonempty generation seeds required")
        if g.temperature <= 0 or any(t <= 0 for t in g.temperatures):
            raise ValueError("Sampling temperatures must be positive")
        if not 0 < g.top_p <= 1 or g.top_k < 0:
            raise ValueError("Invalid top-p/top-k")
        if v.max_length < 32 or min(v.batch_size, v.epochs) < 1 or v.learning_rate <= 0:
            raise ValueError("Invalid verifier configuration")
        if not v.fractions or any(not 0 < f <= 1 for f in v.fractions):
            raise ValueError("Fractions must be in (0,1]")
        if 1.0 not in v.fractions:
            raise ValueError("Full-data verifier required")
        if e.bootstrap_samples < 1 or any(not 0 <= t <= 1 for t in e.thresholds):
            raise ValueError("Invalid evaluation configuration")
        if not e.thresholds or e.token_penalty < 0:
            raise ValueError("Threshold grid and nonnegative token penalty required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> Config:
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")
    sections = {
        "data": DataConfig,
        "generation": GenerationConfig,
        "verifier": VerifierConfig,
        "evaluation": EvaluationConfig,
    }
    for key, cls in sections.items():
        if key in raw:
            raw[key] = cls(**raw[key])
    return Config(**raw)
