"""Label-safe verifier inputs and problem-level training subsets."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence

from verisearch.data.schema import Candidate, LabeledCandidate, Query
from verisearch.data.split import question_key


def verifier_text(query: Query, candidate: Candidate) -> str:
    # Never reference gold or correctness. Put answer first so it survives truncation.
    return (
        f"Question: {query.question}\nCandidate answer: {candidate.answer or '[missing]'}"
        f"\nReasoning: {candidate.text}"
    )


def validate_partitions(
    train: Sequence[LabeledCandidate],
    validation: Sequence[LabeledCandidate],
) -> None:
    if not train or not validation:
        raise ValueError("Nonempty train and validation required")
    if any(row.query.split != "train" for row in train):
        raise ValueError("Training accepts only train problems")
    if any(row.query.split != "validation" for row in validation):
        raise ValueError("Validation accepts only validation problems")
    for attr in ("problem_id", "question"):
        a = {getattr(row.query, attr) for row in train}
        b = {getattr(row.query, attr) for row in validation}
        if attr == "question":
            a, b = {question_key(q) for q in a}, {question_key(q) for q in b}
        if a & b:
            raise ValueError("Problem leakage")
    for rows in (train, validation):
        keys = [(r.query.problem_id, r.candidate.candidate_id) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate verifier examples")


def subset_problems(
    rows: list[LabeledCandidate],
    fraction: float,
    seed: int,
) -> list[LabeledCandidate]:
    if not 0 < fraction <= 1:
        raise ValueError("Invalid fraction")
    ids = sorted(
        {r.query.problem_id for r in rows},
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
    )
    keep = set(ids[: max(1, int(len(ids) * fraction))])
    return [row for row in rows if row.query.problem_id in keep]


def balance(rows: Sequence[LabeledCandidate]) -> dict[str, int | float]:
    counts = Counter(row.correct for row in rows)
    return {
        "candidates": len(rows),
        "problems": len({r.query.problem_id for r in rows}),
        "correct": counts[True],
        "incorrect": counts[False],
        "positive_fraction": counts[True] / len(rows) if rows else 0.0,
    }
