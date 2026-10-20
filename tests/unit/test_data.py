from dataclasses import asdict

import pytest

from verisearch.data.schema import Candidate, LabeledCandidate, Problem, Query
from verisearch.data.split import assert_disjoint, prepare_splits
from verisearch.generation.answer_extraction import extract_answer, is_correct
from verisearch.verifier.dataset import balance, subset_problems, validate_partitions, verifier_text


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("work\n#### 1,234", "1234"),
        ("#### -25", "-25"),
        ("#### 3.50", "7/2"),
        (r"the result is \boxed{42}", "42"),
        (r"\boxed{\frac{3}{2}}", "3/2"),
        ("Final answer: $12.", "12"),
        ("answer is 7.", "7"),
        ("9", "9"),
        ("We compute 2 + 3 = 5", None),
        ("#### nope", None),
        ("", None),
        (r"\boxed{2", None),
        ("#### 1,23", None),
        ("#### 1/0", None),
        (r"\boxed{8} then \boxed{9}", "9"),
        ("#### 7 cats", None),
        ("#### 3\nmore explanation", "3"),
    ],
)
def test_extraction(text: str, answer: str | None) -> None:
    assert extract_answer(text) == answer


def test_exact_labels() -> None:
    assert is_correct("7/2", "3.5")
    assert not is_correct("12", "2")
    assert not is_correct(None, "2")
    with pytest.raises(ValueError):
        is_correct("2", "abc")


def rows(offset: int, count: int) -> list[dict[str, str]]:
    return [
        {"question": f"What is {i}?", "answer": f"#### {i}"} for i in range(offset, offset + count)
    ]


def test_deterministic_disjoint_splits() -> None:
    a = prepare_splits(rows(0, 20), rows(30, 10), 10, 5, 5, 42)
    b = prepare_splits(list(reversed(rows(0, 20))), rows(30, 10), 10, 5, 5, 42)
    assert a == b
    assert_disjoint(a)
    assert {p.query.split for p in a} == {"train", "validation", "test"}
    with pytest.raises(ValueError, match="overlap"):
        prepare_splits(rows(0, 20), rows(0, 10), 10, 5, 5, 42)
    with pytest.raises(ValueError, match="reused"):
        assert_disjoint([a[0], a[0]])


def labeled(split: str = "train", pid: str = "a", cid: int = 0) -> LabeledCandidate:
    query = Query(pid, split, f"question {pid}")  # type: ignore[arg-type]
    candidate = Candidate(pid, cid, "#### 2", "2", 4, 0.01, 17, "test", {})
    return LabeledCandidate.label(Problem(query, "2"), candidate)


def test_serialization_no_label_in_input() -> None:
    row = labeled()
    assert LabeledCandidate.from_dict(row.to_dict()) == row
    assert "gold_answer" not in asdict(row.candidate)
    assert "correct" not in asdict(row.query)
    text = verifier_text(row.query, row.candidate)
    assert "Question:" in text and "Reasoning:" in text
    assert "gold" not in text
    assert balance([row])["positive_fraction"] == 1.0


def test_partition_guards_and_problem_subset() -> None:
    train = [labeled(pid=str(i), cid=j) for i in range(8) for j in range(4)]
    validation = [labeled("validation", "v")]
    validate_partitions(train, validation)
    small = subset_problems(train, 0.25, 17)
    assert len(small) == 8
    for pid in {r.query.problem_id for r in small}:
        assert sum(r.query.problem_id == pid for r in small) == 4
    assert subset_problems(train, 0.25, 17) == small
    with pytest.raises(ValueError, match="only train"):
        validate_partitions([labeled("test")], validation)
    with pytest.raises(ValueError, match="leakage"):
        validate_partitions(train, [labeled("validation", "0")])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_partitions(train + [train[0]], validation)
