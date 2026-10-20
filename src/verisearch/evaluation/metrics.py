"""Binary correctness metrics; undefined ranking metrics are null, never invented."""

from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np


def reliability(
    labels: list[bool],
    scores: list[float],
    bins: int = 10,
) -> list[dict[str, float | int]]:
    if len(labels) != len(scores) or not labels or bins < 1:
        raise ValueError("Nonempty paired labels/scores and positive bins required")
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores must be finite probabilities")
    y, p = np.asarray(labels, dtype=float), np.asarray(scores)
    assignment = np.minimum((p * bins).astype(int), bins - 1)
    result = []
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "confidence": float(p[mask].mean()) if count else 0.0,
                "accuracy": float(y[mask].mean()) if count else 0.0,
            }
        )
    return result


def binary_metrics(labels: list[bool], scores: list[float]) -> dict[str, Any]:
    calibration = reliability(labels, scores)
    y, p = np.asarray(labels, dtype=float), np.asarray(scores)
    both = len(set(labels)) == 2
    auroc, auprc = ranking_metrics(labels, scores) if both else (None, None)
    return {
        "count": len(labels),
        "positive_count": int(y.sum()),
        "positive_fraction": float(y.mean()),
        "accuracy_at_0.5": float(((p >= 0.5) == y).mean()),
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean((p - y) ** 2)),
        "ece": sum(
            float(b["count"]) * abs(float(b["accuracy"]) - float(b["confidence"]))
            for b in calibration
        )
        / len(labels),
        "reliability": calibration,
    }


def bootstrap_accuracy(
    values: list[list[bool]],
    samples: int = 2000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Resample PROBLEMS, retaining all paired generation seeds for each query."""
    if not values or samples < 1 or any(not row for row in values):
        raise ValueError("Nonempty problem outcomes and positive bootstrap count required")
    problem_means = np.array([np.mean(row) for row in values])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(samples, len(values)))
    estimates = problem_means[draws].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def ranking_metrics(labels: list[bool], scores: list[float]) -> tuple[float, float]:
    """AUROC with half credit for ties; average precision at distinct thresholds.

    Grouping equal scores is essential: arbitrary tie order must not change
    either metric. Complexity is O(n log n), with no pairwise score matrix.
    """
    if len(labels) != len(scores) or len(set(labels)) != 2:
        raise ValueError("Paired data with both classes required")
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores must be finite probabilities")
    pairs = sorted(zip(scores, labels, strict=True))
    positives = sum(labels)
    negatives = len(labels) - positives
    credit, negatives_seen = 0.0, 0
    groups: list[tuple[int, int]] = []
    for _, group in groupby(pairs, key=lambda pair: pair[0]):
        outcomes = [label for _, label in group]
        positive, negative = sum(outcomes), len(outcomes) - sum(outcomes)
        credit += positive * (negatives_seen + 0.5 * negative)
        negatives_seen += negative
        groups.append((positive, negative))
    seen_positive, seen_total, ap = 0, 0, 0.0
    for positive, negative in reversed(groups):
        seen_positive += positive
        seen_total += positive + negative
        ap += (positive / positives) * (seen_positive / seen_total)
    return credit / (positives * negatives), ap


def bootstrap_difference(
    pairs: list[list[tuple[bool, bool]]],
    samples: int = 2000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Mean paired accuracy difference and CI, resampling whole problems.

    Each inner list retains all generation-seed pairs for one problem. The
    estimand weights problems equally and compares the first policy minus the
    second; shared candidate pools are paired rather than treated as independent.
    """
    if not pairs or samples < 1 or any(not row for row in pairs):
        raise ValueError("Nonempty paired problem outcomes and positive samples required")
    differences = np.array(
        [np.mean([int(first) - int(second) for first, second in row]) for row in pairs]
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(pairs), size=(samples, len(pairs)))
    low, high = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def answer_presence_diagnostics(
    labels: list[bool],
    scores: list[float],
    answers: list[str | None],
) -> dict[str, Any]:
    """Exploratory shortcut diagnostic, evaluated only after policy decisions.

    A perfect classifier on missing-vs-present answers is not evidence of
    mathematical verification. Compare that trivial feature with the learned
    scores and also evaluate scores conditional on an extracted answer.
    """
    if len(answers) != len(labels):
        raise ValueError("Answer presence and labels must be paired")
    binary_metrics(labels, scores)  # Validate paired, nonempty probabilities.
    present = [index for index, answer in enumerate(answers) if answer is not None]
    missing = [index for index, answer in enumerate(answers) if answer is None]
    return {
        "count": len(labels),
        "answer_present": len(present),
        "answer_missing": len(missing),
        "present_correct": sum(labels[index] for index in present),
        "present_incorrect": sum(not labels[index] for index in present),
        "missing_correct": sum(labels[index] for index in missing),
        "presence_heuristic_metrics": binary_metrics(
            labels, [float(answer is not None) for answer in answers]
        ),
        "verifier_metrics_on_present_answers": binary_metrics(
            [labels[index] for index in present], [scores[index] for index in present]
        )
        if present
        else None,
        "scope": "Exploratory diagnostic motivated by validation-only error composition; "
        "not used for training, tuning, candidate selection or configuration changes.",
    }
