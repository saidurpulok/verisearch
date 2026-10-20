"""File-backed experiment stages with explicit validation/test gates.

Candidate generation and policy decisions are label-free. Labeled artifacts
are used for verifier training and *after* decisions for measured outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from verisearch.config import Config
from verisearch.data.load import load_problems
from verisearch.data.schema import LabeledCandidate, Problem
from verisearch.evaluation.metrics import (
    answer_presence_diagnostics,
    binary_metrics,
    bootstrap_accuracy,
    bootstrap_difference,
)
from verisearch.io import (
    digest,
    file_sha256,
    provenance,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from verisearch.policies.selection import Policy, select
from verisearch.verifier.dataset import subset_problems


class ThresholdTrial(TypedDict):
    threshold: float | None
    accuracy: float
    mean_fraction_budget: float
    objective: float


def check_config(config: Config) -> Path:
    root = Path(config.output)
    if read_json(root / "config.json") != config.to_dict():
        raise ValueError("Config differs from prepared experiment; use a new output directory")
    return root


def pool_id(split: str, pid: str, budget: int, temperature: float, seed: int, kind: str) -> str:
    return digest([split, pid, budget, temperature, seed, kind])


def generate(config: Config, partition: str) -> None:
    root = check_config(config)
    if partition not in ("fit", "test"):
        raise ValueError("Partition must be fit or test")
    if partition == "test" and not (root / "tuning.json").exists():
        raise ValueError("Freeze validation thresholds before generating test candidates")
    path = root / f"{partition}_pools.jsonl"
    if path.exists():
        raise FileExistsError("Pools already exist; use resume rather than overwrite")
    from verisearch.generation.generator import Generator

    generator = Generator(config.generation)
    problems = load_problems(root)
    g = config.generation
    # Every complete pool is checkpointed atomically; interrupted runs can resume.
    partial = root / f"{partition}_partial.jsonl"
    pools = read_jsonl(partial) if partial.exists() else []
    done = {p["pool_id"] for p in pools}
    selected = [p for p in problems if (p.query.split == "test") == (partition == "test")]
    tasks: list[tuple[Problem, int, float, int, str]] = []
    for problem in selected:
        if problem.query.split == "train":
            tasks.append(
                (problem, g.max_new_tokens * g.candidates, g.temperature, g.seeds[0], "stochastic")
            )
        else:
            for budget in g.budgets:
                for seed in g.seeds:
                    tasks.append((problem, budget, g.temperature, seed, "stochastic"))
                tasks.append((problem, budget, g.temperature, g.seeds[0], "greedy"))
            # Predeclared temperature ablation: middle budget, first seed only.
            if partition == "test":
                middle = g.budgets[len(g.budgets) // 2]
                for temperature in g.temperatures:
                    if temperature != g.temperature:
                        tasks.append((problem, middle, temperature, g.seeds[0], "stochastic"))
    for index, (problem, budget, temperature, seed, kind) in enumerate(tasks):
        pid = pool_id(
            problem.query.split, problem.query.problem_id, budget, temperature, seed, kind
        )
        if pid in done:
            continue
        if kind == "greedy":
            candidates = [generator.generate(problem.query, 0, seed, budget, greedy=True)]
        else:
            candidates = list(
                generator.stream(problem.query, seed, budget, temperature=temperature)
            )
        rows = [LabeledCandidate.label(problem, c).to_dict() for c in candidates]
        pools.append(
            {
                "pool_id": pid,
                "split": problem.query.split,
                "problem_id": problem.query.problem_id,
                "budget": budget,
                "temperature": temperature,
                "seed": seed,
                "kind": kind,
                "candidates": rows,
            }
        )
        write_jsonl(partial, pools)
        print(
            f"{partition} pool {index + 1}/{len(tasks)}: {problem.query.split} "
            f"{kind} budget={budget} tokens={sum(c.tokens for c in candidates)}",
            flush=True,
        )
    write_jsonl(path, pools)
    write_json(
        root / f"{partition}_generation.json",
        {
            **provenance(),
            "config": config.to_dict(),
            "pools": len(pools),
            "sha256": digest(pools),
            "resolved_revision": generator.resolved_revision,
            "device": g.device,
            "hardware": generator.hardware(),
            "note": "Per-call synchronized latency; prefix replay costs, no batching.",
        },
    )
    partial.unlink(missing_ok=True)


def load_training(config: Config) -> tuple[list[LabeledCandidate], list[LabeledCandidate]]:
    pools = load_pools(config, "fit")
    train, validation = [], []
    full_budget = max(config.generation.budgets)
    for pool in pools:
        rows = [LabeledCandidate.from_dict(r) for r in pool["candidates"]]
        if pool["split"] == "train":
            train.extend(rows)
        elif (
            pool["split"] == "validation"
            and pool["budget"] == full_budget
            and pool["seed"] == config.generation.seeds[0]
            and pool["kind"] == "stochastic"
        ):
            validation.extend(rows)
    return train, validation


def train(config: Config) -> None:
    from verisearch.verifier.train import train_verifier

    root = check_config(config)
    train_rows, validation = load_training(config)
    for fraction in config.verifier.fractions:
        output = root / "verifiers" / f"fraction_{fraction:g}"
        if (output / "metadata.json").exists():
            continue
        train_verifier(
            config.verifier,
            subset_problems(train_rows, fraction, config.verifier.seed),
            validation,
            output,
        )


def score_pools(
    config: Config,
    pools: list[dict[str, Any]],
    fraction: float = 1.0,
) -> dict[str, dict[int, float]]:
    from verisearch.verifier.model import Verifier

    root = Path(config.output)
    checkpoint = root / "verifiers" / f"fraction_{fraction:g}"
    metadata = read_json(checkpoint / "metadata.json")
    if file_sha256(checkpoint / "model.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Verifier weights no longer match training metadata")
    verifier = Verifier(
        checkpoint,
        config.verifier.device,
        config.verifier.max_length,
    )
    scores: dict[str, dict[int, float]] = {}
    for pool in pools:
        entries = [LabeledCandidate.from_dict(r) for r in pool["candidates"]]
        scores[pool["pool_id"]] = {
            r.candidate.candidate_id: verifier.score(r.query, r.candidate) for r in entries
        }
    return scores


def decide(
    pool: dict[str, Any],
    scores: dict[int, float],
    policy: Policy,
    count: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    records = [LabeledCandidate.from_dict(r) for r in pool["candidates"]]
    candidates = [r.candidate for r in records]
    # The selector cannot access labels. Evaluation attaches correctness afterwards.
    selection = select(
        candidates,
        policy,
        pool["budget"],
        count,
        lambda candidate: scores[candidate.candidate_id],
        threshold,
    )
    selected = selection.candidate
    correct = next((r.correct for r in records if r.candidate == selected), False)
    return {
        "problem_id": pool["problem_id"],
        "pool_id": pool["pool_id"],
        "seed": pool["seed"],
        "policy": policy,
        "budget": pool["budget"],
        "temperature": pool["temperature"],
        "max_candidates": count,
        "threshold": threshold,
        "correct": correct,
        "tokens": selection.tokens,
        "candidates": selection.candidates,
        "generation_seconds": selection.generation_seconds,
        "selected_candidate": selected.candidate_id if selected else None,
        "answer": selected.answer if selected else None,
    }


def tune(config: Config) -> None:
    root = check_config(config)
    if (root / "test_pools.jsonl").exists():
        raise ValueError("Cannot retune after test generation")
    if (root / "tuning.json").exists():
        raise FileExistsError("Tuning already frozen")
    pools = [
        p
        for p in load_pools(config, "fit")
        if p["split"] == "validation" and p["kind"] == "stochastic"
    ]
    scores = score_pools(config, pools)
    trials: list[ThresholdTrial] = []
    for threshold in [None, *config.evaluation.thresholds]:
        rows = [
            decide(p, scores[p["pool_id"]], "adaptive", config.generation.candidates, threshold)
            for p in pools
        ]
        accuracy = float(np.mean([r["correct"] for r in rows]))
        fraction = float(np.mean([r["tokens"] / r["budget"] for r in rows]))
        trials.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "mean_fraction_budget": fraction,
                "objective": accuracy - config.evaluation.token_penalty * fraction,
            }
        )
    chosen = max(trials, key=lambda r: (r["objective"], -r["mean_fraction_budget"]))
    write_json(
        root / "tuning.json",
        {
            **provenance(),
            "split": "validation",
            "trials": trials,
            "chosen_threshold": chosen["threshold"],
            "config": config.to_dict(),
            "rule": "max validation accuracy - token_penalty * mean fraction budget; "
            "ties use fewer tokens, then first threshold in predefined grid",
            "validation_pool_sha256": digest(pools),
        },
    )


def summarize(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys = ("policy", "budget", "temperature", "max_candidates", "threshold")
    for row in rows:
        groups[digest([row[k] for k in keys])].append(row)
    output = []
    for group in groups.values():
        outcomes: dict[str, list[bool]] = defaultdict(list)
        for row in group:
            outcomes[row["problem_id"]].append(row["correct"])
        interval = bootstrap_accuracy(
            list(outcomes.values()),
            config.evaluation.bootstrap_samples,
            config.evaluation.bootstrap_seed,
        )
        output.append(
            {
                **{key: group[0][key] for key in keys},
                "problems": len(outcomes),
                "trials": len(group),
                "seeds": sorted({r["seed"] for r in group}),
                "accuracy": float(np.mean([r["correct"] for r in group])),
                "accuracy_ci95": list(interval),
                "mean_tokens": float(np.mean([r["tokens"] for r in group])),
                "median_tokens": float(np.median([r["tokens"] for r in group])),
                "mean_candidates": float(np.mean([r["candidates"] for r in group])),
                "mean_generation_seconds": float(np.mean([r["generation_seconds"] for r in group])),
            }
        )
    return output


def paired_comparisons(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    """Compare predeclared policies on matching test pools; never select a policy."""
    output = []
    for budget in config.generation.budgets:
        eligible = [
            row
            for row in rows
            if row["budget"] == budget
            and row["temperature"] == config.generation.temperature
            and row["max_candidates"] == config.generation.candidates
        ]
        for first, second in (("verifier", "self_consistency"), ("adaptive_tuned", "verifier")):
            left = {r["pool_id"]: r for r in eligible if r["policy"] == first}
            right = {r["pool_id"]: r for r in eligible if r["policy"] == second}
            if not left or left.keys() != right.keys():
                raise ValueError("Paired comparison requires identical nonempty pool identities")
            outcomes: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
            for identity in sorted(left):
                a, b = left[identity], right[identity]
                outcomes[a["problem_id"]].append((a["correct"], b["correct"]))
            difference, low, high = bootstrap_difference(
                list(outcomes.values()),
                config.evaluation.bootstrap_samples,
                config.evaluation.bootstrap_seed,
            )
            output.append(
                {
                    "first": first,
                    "second": second,
                    "budget": budget,
                    "problems": len(outcomes),
                    "paired_trials": len(left),
                    "accuracy_difference": difference,
                    "difference_ci95": [low, high],
                    "method": "paired problem-cluster percentile bootstrap",
                }
            )
    return output


def evaluate(config: Config) -> None:
    root = check_config(config)
    tuning = read_json(root / "tuning.json")
    pools = load_pools(config, "test")
    if any(p["split"] != "test" for p in pools):
        raise ValueError("Final evaluation accepts test pools only")
    scores = score_pools(config, pools)
    rows = []
    counts = sorted({1, min(2, config.generation.candidates), config.generation.candidates})
    for pool in pools:
        pool_scores = scores[pool["pool_id"]]
        if pool["kind"] == "greedy":
            rows.append(decide(pool, pool_scores, "greedy", 1))
            continue
        rows.append(decide(pool, pool_scores, "single", 1))
        for count in counts:
            for policy in ("self_consistency", "verifier"):
                rows.append(decide(pool, pool_scores, policy, count))
            oracle = decide(pool, pool_scores, "self_consistency", count)
            oracle.update(
                policy="oracle",
                correct=any(r["correct"] for r in pool["candidates"][:count]),
                selected_candidate=None,
                answer=None,
            )
            rows.append(oracle)
        for threshold in [None, *config.evaluation.thresholds]:
            rows.append(
                decide(pool, pool_scores, "adaptive", config.generation.candidates, threshold)
            )
        adaptive = decide(
            pool, pool_scores, "adaptive", config.generation.candidates, tuning["chosen_threshold"]
        )
        adaptive["policy"] = "adaptive_tuned"
        rows.append(adaptive)
    # Candidate-level metrics on a predeclared single pool per problem, avoiding
    # repeated prefixes from different budgets/seeds.
    primary = [
        p
        for p in pools
        if p["kind"] == "stochastic"
        and p["budget"] == max(config.generation.budgets)
        and p["temperature"] == config.generation.temperature
        and p["seed"] == config.generation.seeds[0]
    ]
    labels = [r["correct"] for p in primary for r in p["candidates"]]
    probabilities = [
        scores[p["pool_id"]][r["candidate"]["candidate_id"]]
        for p in primary
        for r in p["candidates"]
    ]
    verifier_metrics = binary_metrics(labels, probabilities)
    answers = [r["candidate"]["answer"] for p in primary for r in p["candidates"]]
    shortcut_diagnostic = answer_presence_diagnostics(labels, probabilities, answers)
    fit_train, fit_validation = load_training(config)
    fit_answer_composition = {
        name: {
            "count": len(entries),
            "answer_missing": sum(r.candidate.answer is None for r in entries),
            "present_correct": sum(r.correct for r in entries if r.candidate.answer is not None),
            "present_incorrect": sum(
                not r.correct for r in entries if r.candidate.answer is not None
            ),
        }
        for name, entries in (("train", fit_train), ("validation", fit_validation))
    }
    training_ablation = []
    for fraction in config.verifier.fractions:
        subset_scores = scores if fraction == 1.0 else score_pools(config, primary, fraction)
        probabilities_fraction = [
            subset_scores[p["pool_id"]][r["candidate"]["candidate_id"]]
            for p in primary
            for r in p["candidates"]
        ]
        selected = [
            decide(p, subset_scores[p["pool_id"]], "verifier", config.generation.candidates)
            for p in primary
        ]
        training_ablation.append(
            {
                "fraction": fraction,
                "metrics": binary_metrics(labels, probabilities_fraction),
                "selection": summarize(selected, config),
                "training": read_json(
                    root / "verifiers" / f"fraction_{fraction:g}" / "metadata.json"
                ),
            }
        )
    write_jsonl(root / "decisions.jsonl", rows)
    write_json(
        root / "results.json",
        {
            **provenance(),
            "config": config.to_dict(),
            "dataset": config.data.dataset,
            "split": "test",
            "summaries": summarize(rows, config),
            "paired_comparisons": paired_comparisons(rows, config),
            "verifier_metrics": verifier_metrics,
            "answer_presence_diagnostic": shortcut_diagnostic,
            "fit_answer_composition": fit_answer_composition,
            "failure_analysis_scope": {
                "temperature": config.generation.temperature,
                "budgets": config.generation.budgets,
                "seeds": config.generation.seeds,
                "note": "All primary-temperature stochastic pools; coverage expanded in "
                "final artifact review to include both seeds and policy disagreements. "
                "No training, tuning or policy change.",
            },
            "verifier_labels": labels,
            "verifier_scores": probabilities,
            "training_ablation": training_ablation,
            "tuning": tuning,
            "generation_manifest": read_json(root / "test_generation.json"),
            "data_manifest": read_json(root / "data/manifest.json"),
            "test_pools_sha256": digest(pools),
            "latency_scope": "Recorded synchronized generation latency summed over used prefix. "
            "Verifier latency not included; adaptive outcomes are offline replay.",
        },
    )
    from verisearch.evaluation.error_analysis import error_report

    analysis_pools = [
        p
        for p in pools
        if p["kind"] == "stochastic" and p["temperature"] == config.generation.temperature
    ]
    error_report(analysis_pools, scores, root / "errors.md")


def evaluate_verifier(config: Config, split: str) -> None:
    root = check_config(config)
    if split == "test" and not (root / "tuning.json").exists():
        raise ValueError("Test evaluation requires frozen tuning")
    partition = "fit" if split == "validation" else "test"
    pools = [
        p
        for p in load_pools(config, partition)
        if p["split"] == split
        and p["kind"] == "stochastic"
        and p["budget"] == max(config.generation.budgets)
        and p["seed"] == config.generation.seeds[0]
    ]
    scores = score_pools(config, pools)
    labels = [r["correct"] for p in pools for r in p["candidates"]]
    probabilities = [
        scores[p["pool_id"]][r["candidate"]["candidate_id"]] for p in pools for r in p["candidates"]
    ]
    write_json(
        root / f"verifier_{split}.json",
        {
            **provenance(),
            "split": split,
            "config": config.to_dict(),
            "metrics": binary_metrics(labels, probabilities),
        },
    )


def load_pools(config: Config, partition: str) -> list[dict[str, Any]]:
    root = check_config(config)
    pools = read_jsonl(root / f"{partition}_pools.jsonl")
    manifest = read_json(root / f"{partition}_generation.json")
    if manifest["sha256"] != digest(pools) or manifest["config"] != config.to_dict():
        raise ValueError("Candidate pools no longer match generation manifest/config")
    return pools
