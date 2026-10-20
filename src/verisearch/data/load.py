"""Official GSM8K loading and immutable split manifests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from verisearch.config import Config
from verisearch.data.schema import Problem
from verisearch.data.split import assert_disjoint, prepare_splits
from verisearch.io import digest, provenance, read_json, read_jsonl, write_json, write_jsonl


def prepare_data(config: Config) -> None:
    from datasets import load_dataset

    directory = Path(config.output) / "data"
    if directory.exists():
        raise FileExistsError("Refusing to overwrite existing splits; use a new output directory")
    dataset: Any = load_dataset(config.data.dataset, "main", revision=config.data.revision)
    problems = prepare_splits(
        list(dataset["train"]),
        list(dataset["test"]),
        config.data.train_problems,
        config.data.validation_problems,
        config.data.test_problems,
        config.data.split_seed,
    )
    rows = [p.to_dict() for p in problems]
    write_jsonl(directory / "problems.jsonl", rows)
    write_json(
        directory / "manifest.json",
        {
            **provenance(),
            "dataset": config.data.dataset,
            "revision": config.data.revision,
            "official_fingerprints": {key: dataset[key]._fingerprint for key in dataset},
            "official_sizes": {key: len(dataset[key]) for key in dataset},
            "config": config.to_dict(),
            "counts": dict(Counter(p.query.split for p in problems)),
            "problems_sha256": digest(rows),
            "problem_ids": {
                s: [p.query.problem_id for p in problems if p.query.split == s]
                for s in ("train", "validation", "test")
            },
        },
    )
    write_json(Path(config.output) / "config.json", config.to_dict())


def load_problems(directory: Path) -> list[Problem]:
    rows = read_jsonl(directory / "data/problems.jsonl")
    manifest = read_json(directory / "data/manifest.json")
    if digest(rows) != manifest["problems_sha256"]:
        raise ValueError("Prepared problems no longer match their immutable manifest")
    problems = [Problem.from_dict(row) for row in rows]
    assert_disjoint(problems)
    return problems
