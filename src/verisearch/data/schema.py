"""Labels and gold answers are absent from the policy-facing types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from verisearch.generation.answer_extraction import is_correct

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class Query:
    problem_id: str
    split: Split
    question: str


@dataclass(frozen=True)
class Problem:
    query: Query
    gold_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Problem:
        return cls(Query(**row["query"]), row["gold_answer"])


@dataclass(frozen=True)
class Candidate:
    problem_id: str
    candidate_id: int
    text: str
    answer: str | None
    tokens: int
    latency_seconds: float
    seed: int
    checkpoint: str
    generation: dict[str, Any]

    def __post_init__(self) -> None:
        if self.tokens < 1 or self.latency_seconds < 0:
            raise ValueError("Candidates must contain tokens and nonnegative latency")


@dataclass(frozen=True)
class LabeledCandidate:
    query: Query
    candidate: Candidate
    gold_answer: str
    correct: bool

    def __post_init__(self) -> None:
        if self.query.problem_id != self.candidate.problem_id:
            raise ValueError("Candidate/query mismatch")
        if self.correct != is_correct(self.candidate.answer, self.gold_answer):
            raise ValueError("Inconsistent correctness label")

    @classmethod
    def label(cls, problem: Problem, candidate: Candidate) -> LabeledCandidate:
        return cls(
            problem.query,
            candidate,
            problem.gold_answer,
            is_correct(candidate.answer, problem.gold_answer),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LabeledCandidate:
        return cls(
            Query(**row["query"]), Candidate(**row["candidate"]), row["gold_answer"], row["correct"]
        )
