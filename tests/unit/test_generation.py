"""CPU mechanism checks. Controlled traces are never presented as GPU evidence."""

from verisearch.config import GenerationConfig
from verisearch.data.schema import Candidate, Query
from verisearch.generation.generator import Generator, candidate_seed


class ControlledGenerator(Generator):
    def __init__(self) -> None:
        self.config = GenerationConfig(device="cpu", max_new_tokens=10, candidates=4)
        self.calls: list[int] = []

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
        self.calls.append(token_cap)
        tokens = min(token_cap, self.config.max_new_tokens)
        return Candidate(
            query.problem_id,
            candidate_id,
            "#### 2",
            "2",
            tokens,
            0.01,
            candidate_seed(seed, query.problem_id, candidate_id),
            "controlled",
            {},
        )


def test_stream_caps_last_call_at_remaining_budget() -> None:
    generator = ControlledGenerator()
    traces = list(generator.stream(Query("q", "train", "one plus one"), 17, 25))
    assert generator.calls == [25, 15, 5]
    assert [c.tokens for c in traces] == [10, 10, 5]
    assert sum(c.tokens for c in traces) == 25


def test_candidate_cap_and_stable_per_problem_seeds() -> None:
    generator = ControlledGenerator()
    traces = list(generator.stream(Query("q", "train", "one plus one"), 17, 100, max_candidates=2))
    assert len(traces) == 2 and sum(c.tokens for c in traces) == 20
    assert candidate_seed(17, "q", 0) == candidate_seed(17, "q", 0)
    assert candidate_seed(17, "q", 0) != candidate_seed(17, "different", 0)
    assert candidate_seed(17, "q", 0) != candidate_seed(17, "q", 1)
