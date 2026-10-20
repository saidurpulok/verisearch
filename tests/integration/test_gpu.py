"""Real pretrained GPU integration. Disabled in CPU CI; never substituted by mocks."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from verisearch.config import load_config
from verisearch.data.schema import LabeledCandidate, Problem, Query
from verisearch.policies.selection import select

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.pretrained,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("VERISEARCH_GPU_TESTS") != "1", reason="Enable real GPU tests explicitly"
    ),
]


def test_real_generation_scoring_and_end_to_end() -> None:
    from verisearch.generation.generator import Generator
    from verisearch.verifier.model import Verifier

    config = load_config(Path(os.environ.get("VERISEARCH_CONFIG", "configs/pilot.yaml")))
    generator = Generator(replace(config.generation, max_new_tokens=64))
    query = Query(
        "gpu-integration-arithmetic",
        "test",
        "I have 2 apples and buy 3 more. How many apples do I have?",
    )
    a = generator.generate(query, 0, 17, 64)
    b = generator.generate(query, 0, 17, 64)
    assert a.text == b.text and a.tokens == b.tokens
    assert 0 < a.tokens <= 64
    other = generator.generate(query, 1, 17, 64)
    greedy_a = generator.generate(query, 0, 17, 64, greedy=True)
    greedy_b = generator.generate(query, 0, 29, 64, greedy=True)
    assert greedy_a.text == greedy_b.text
    verifier = Verifier(
        Path(config.output) / "verifiers/fraction_1", "cuda", config.verifier.max_length
    )
    score = verifier.score(query, a)
    assert 0 <= score <= 1
    result = select([a, other], "verifier", 128, 2, lambda c: verifier.score(query, c))
    assert result.candidate is not None and result.tokens <= 128
    label = LabeledCandidate.label(Problem(query, "5"), result.candidate)
    assert isinstance(label.correct, bool)
    # A real live stream must stop before generating an unused second candidate.
    early = select(
        generator.stream(query, 17, 128, max_candidates=2),
        "adaptive",
        128,
        2,
        lambda c: verifier.score(query, c),
        threshold=0.0,
    )
    assert early.candidates == 1 and early.tokens <= 64
    full = select(
        generator.stream(query, 17, 128, max_candidates=2),
        "adaptive",
        128,
        2,
        lambda c: verifier.score(query, c),
        threshold=None,
    )
    assert full.candidates == 2 and full.tokens <= 128
