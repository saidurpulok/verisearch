from dataclasses import replace
from pathlib import Path

import pytest

from verisearch.config import Config
from verisearch.data.schema import Candidate, LabeledCandidate, Problem, Query
from verisearch.evaluation.error_analysis import error_report
from verisearch.experiment import check_config, decide, summarize
from verisearch.io import read_json, read_jsonl, write_json, write_jsonl


def pool() -> dict:  # type: ignore[type-arg]
    query = Query("p", "test", "How much is 1+1?")
    candidates = [
        Candidate("p", i, f"#### {answer}", answer, 10, 0.1, 17, "test", {})
        for i, answer in enumerate(["1", "2", "2"])
    ]
    return {
        "pool_id": "pool",
        "problem_id": "p",
        "split": "test",
        "budget": 30,
        "temperature": 0.8,
        "seed": 17,
        "kind": "stochastic",
        "candidates": [
            LabeledCandidate.label(Problem(query, "2"), c).to_dict() for c in candidates
        ],
    }


def test_selection_evaluation_and_clustered_summary() -> None:
    p = pool()
    scores = {0: 0.9, 1: 0.8, 2: 0.7}
    ranking = decide(p, scores, "verifier", 3)
    voting = decide(p, scores, "self_consistency", 3)
    assert not ranking["correct"] and voting["correct"]
    assert ranking["tokens"] == voting["tokens"] == 30
    adaptive = decide(p, scores, "adaptive", 3, 0.7)
    assert adaptive["tokens"] == 10
    summary = summarize([ranking, {**ranking, "seed": 29}], Config())
    assert summary[0]["problems"] == 1 and summary[0]["trials"] == 2
    assert summary[0]["accuracy_ci95"] == [0.0, 0.0]


def test_config_freezing_and_atomic_io(tmp_path: Path) -> None:
    config = replace(Config(), output=str(tmp_path))
    write_json(tmp_path / "config.json", config.to_dict())
    assert check_config(config) == tmp_path
    with pytest.raises(ValueError, match="differs"):
        check_config(replace(config, name="changed"))
    write_jsonl(tmp_path / "pools.jsonl", [pool()])
    assert read_jsonl(tmp_path / "pools.jsonl") == [pool()]
    assert read_json(tmp_path / "config.json")["output"] == str(tmp_path)


def test_unfavorable_error_examples(tmp_path: Path) -> None:
    output = tmp_path / "errors.md"
    error_report([pool()], {"pool": {0: 0.95, 1: 0.1, 2: 0.2}}, output)
    text = output.read_text()
    assert "Self-consistency succeeds; verifier fails (1 problems; 1 pool cases)" in text
    assert "High-confidence (>=0.9) false positive (1 problems; 1 pool cases)" in text
    assert "Candidate 1" in text and "#### 2" in text


def test_manifest_tampering_is_detected(tmp_path: Path) -> None:
    from verisearch.experiment import load_pools
    from verisearch.io import digest

    config = replace(Config(), output=str(tmp_path))
    write_json(tmp_path / "config.json", config.to_dict())
    write_jsonl(tmp_path / "fit_pools.jsonl", [pool()])
    write_json(
        tmp_path / "fit_generation.json", {"sha256": digest([pool()]), "config": config.to_dict()}
    )
    assert load_pools(config, "fit") == [pool()]
    altered = pool()
    altered["budget"] = 100
    write_jsonl(tmp_path / "fit_pools.jsonl", [altered])
    with pytest.raises(ValueError, match="manifest"):
        load_pools(config, "fit")


def test_paired_comparisons_require_identical_candidate_pools() -> None:
    from verisearch.experiment import paired_comparisons

    config = Config()
    config = replace(config, generation=replace(config.generation, budgets=[30], candidates=3))
    p = pool()
    scores = {0: 0.9, 1: 0.8, 2: 0.7}
    ranking = decide(p, scores, "verifier", 3)
    voting = decide(p, scores, "self_consistency", 3)
    adaptive = {**decide(p, scores, "adaptive", 3, 0.7), "policy": "adaptive_tuned"}
    comparisons = paired_comparisons([ranking, voting, adaptive], config)
    assert comparisons[0]["accuracy_difference"] == -1
    assert comparisons[0]["difference_ci95"] == [-1, -1]
    assert comparisons[1]["accuracy_difference"] == 0
    with pytest.raises(ValueError, match="identical"):
        paired_comparisons([ranking, {**voting, "pool_id": "different"}, adaptive], config)


def test_error_analysis_counts_unique_problems_across_seeds(tmp_path: Path) -> None:
    first = pool()
    second = {**first, "pool_id": "other-seed", "seed": 29}
    scores = {0: 0.95, 1: 0.1, 2: 0.2}
    output = tmp_path / "errors.md"
    error_report([second, first], {"pool": scores, "other-seed": scores}, output)
    text = output.read_text()
    assert "Self-consistency succeeds; verifier fails (1 problems; 2 pool cases)" in text
    assert "seed=17" in text and "seed=29" not in text
    assert "cap=30" in text
