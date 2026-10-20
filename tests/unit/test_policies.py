from collections.abc import Iterator

import pytest

from verisearch.data.schema import Candidate
from verisearch.policies.selection import majority, select


def candidate(index: int, answer: str | None = "2", tokens: int = 10) -> Candidate:
    return Candidate("p", index, str(answer), answer, tokens, 0.01, 17, "test", {})


def test_majority_and_ties() -> None:
    pool = [candidate(0, "1"), candidate(1, "2"), candidate(2, "2")]
    assert majority(pool) == pool[1]
    assert majority(pool[:2]) == pool[0]
    assert majority([candidate(0, None), candidate(1, "2")]) == candidate(1, "2")
    assert majority([candidate(0, None)]) == candidate(0, None)
    assert majority([]) is None


def test_greedy_single_and_ranking() -> None:
    pool = [candidate(i) for i in range(4)]
    for policy in ("greedy", "single"):
        result = select(pool, policy, 40, 4)  # type: ignore[arg-type]
        assert result.candidate == pool[0] and result.tokens == 10

    def score(c: Candidate) -> float:
        return [0.3, 0.9, 0.9, 0.1][c.candidate_id]

    result = select(pool, "verifier", 40, 4, score)
    assert result.candidate == pool[1] and result.candidates == 4


def test_adaptive_does_not_generate_unused_candidates() -> None:
    generated = []

    def stream() -> Iterator[Candidate]:
        for i in range(4):
            generated.append(i)
            yield candidate(i)

    result = select(stream(), "adaptive", 40, 4, lambda c: 0.8, threshold=0.7)
    assert generated == [0] and result.tokens == 10
    generated.clear()
    result = select(stream(), "adaptive", 40, 4, lambda c: 0.8, threshold=None)
    assert generated == [0, 1, 2, 3] and result.tokens == 40
    generated.clear()
    select(stream(), "adaptive", 10, 4, lambda c: 0.1)
    assert generated == [0]


def test_budget_violations() -> None:
    with pytest.raises(ValueError, match="violates"):
        select([candidate(0, tokens=11)], "single", 10, 4)
    with pytest.raises(ValueError):
        select([], "single", 0, 4)
    with pytest.raises(ValueError, match="scorer"):
        select([], "verifier", 10, 4)
    with pytest.raises(ValueError, match="finite"):
        select([candidate(0)], "verifier", 10, 4, lambda c: float("nan"))


def test_no_label_based_selection_and_missing_verifier_answer() -> None:
    pool = [candidate(0, None), candidate(1, "2")]
    # A verifier may unfortunately prefer a malformed trace: do not repair it
    # with an evaluation label or silently change the policy.
    result = select(pool, "verifier", 20, 2, lambda c: 0.9 if c.answer is None else 0.1)
    assert result.candidate == pool[0] and result.candidate.answer is None
    # With confidence off, adaptive is exactly best-of-N.
    full = select(pool, "adaptive", 20, 2, lambda c: 0.2, threshold=None)
    ordinary = select(pool, "verifier", 20, 2, lambda c: 0.2)
    assert full == ordinary


def test_candidate_limit_prevents_iterator_side_effects() -> None:
    calls = []

    def stream() -> Iterator[Candidate]:
        for i in range(4):
            calls.append(i)
            yield candidate(i)

    result = select(stream(), "self_consistency", 100, 2)
    assert calls == [0, 1] and result.tokens == 20
