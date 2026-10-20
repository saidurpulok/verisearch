"""Stable hash ranking of unique normalized questions before trace generation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from verisearch.data.schema import Problem, Query, Split
from verisearch.generation.answer_extraction import extract_answer


def question_key(question: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", question.strip()).encode()).hexdigest()


def prepare_splits(
    official_train: Sequence[dict[str, str]],
    official_test: Sequence[dict[str, str]],
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> list[Problem]:
    def unique(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            key = question_key(row["question"])
            if key in result and result[key]["answer"] != row["answer"]:
                raise ValueError("Duplicate question has inconsistent answers")
            result[key] = row
        return result

    train, test = unique(official_train), unique(official_test)
    if train.keys() & test.keys():
        raise ValueError("Official train/test question overlap")
    if train_count + validation_count > len(train) or test_count > len(test):
        raise ValueError("Requested sample exceeds unique official split")

    def ranked(rows: dict[str, dict[str, str]]) -> list[str]:
        return sorted(rows, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())

    train_ids, test_ids = ranked(train), ranked(test)
    partitions: list[tuple[Split, list[str], dict[str, dict[str, str]]]] = [
        ("train", train_ids[:train_count], train),
        ("validation", train_ids[train_count : train_count + validation_count], train),
        ("test", test_ids[:test_count], test),
    ]
    problems = []
    for split, keys, source in partitions:
        for key in keys:
            answer = extract_answer(source[key]["answer"])
            if answer is None:
                raise ValueError("Official answer cannot be parsed")
            problems.append(Problem(Query(f"gsm8k-{key}", split, source[key]["question"]), answer))
    assert_disjoint(problems)
    return problems


def assert_disjoint(problems: Sequence[Problem]) -> None:
    ids: set[str] = set()
    questions: set[str] = set()
    for problem in problems:
        key = question_key(problem.query.question)
        if problem.query.problem_id in ids or key in questions:
            raise ValueError("Problem identity or normalized question reused")
        ids.add(problem.query.problem_id)
        questions.add(key)
