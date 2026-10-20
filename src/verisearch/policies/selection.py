"""Gold-free selection over lazily generated, strictly budgeted candidate streams.

Ties always prefer the earliest candidate. Malformed answers never form a
self-consistency voting bloc; if all are malformed return the first trace.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from verisearch.data.schema import Candidate

Policy = Literal["greedy", "single", "self_consistency", "verifier", "adaptive"]


@dataclass(frozen=True)
class Selection:
    candidate: Candidate | None
    tokens: int
    candidates: int
    generation_seconds: float
    verifier_seconds: float | None = None  # Generic callbacks are not timed here.


def majority(pool: list[Candidate]) -> Candidate | None:
    votes = Counter(c.answer for c in pool if c.answer is not None)
    if not votes:
        return pool[0] if pool else None
    most = max(votes.values())
    return next(c for c in pool if c.answer is not None and votes[c.answer] == most)


def select(
    pool: Iterable[Candidate],
    policy: Policy,
    budget: int,
    max_candidates: int,
    scorer: Callable[[Candidate], float] | None = None,
    threshold: float | None = None,
) -> Selection:
    if budget < 1 or max_candidates < 1:
        raise ValueError("Positive budget/count required")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a probability")
    if policy in ("verifier", "adaptive") and scorer is None:
        raise ValueError("Learned scorer required")
    seen: list[Candidate] = []
    tokens, latency = 0, 0.0
    best: Candidate | None = None
    best_score = float("-inf")
    iterator = iter(pool)
    # Check limits before next(): a live generator must never do unused work.
    while len(seen) < max_candidates and tokens < budget:
        try:
            candidate = next(iterator)
        except StopIteration:
            break
        if tokens + candidate.tokens > budget:
            raise ValueError("Pool violates token cap; regenerate with the remaining budget")
        if seen and candidate.problem_id != seen[0].problem_id:
            raise ValueError("Mixed problems in selection pool")
        seen.append(candidate)
        tokens += candidate.tokens
        latency += candidate.latency_seconds
        if policy in ("greedy", "single"):
            best = candidate
            break
        if policy in ("verifier", "adaptive"):
            assert scorer is not None
            score = scorer(candidate)
            if not 0 <= score <= 1:
                raise ValueError("Verifier score must be a finite probability")
            if score > best_score:
                best, best_score = candidate, score
            if policy == "adaptive" and threshold is not None and best_score >= threshold:
                break
    if policy == "self_consistency":
        best = majority(seen)
    return Selection(best, tokens, len(seen), latency)
