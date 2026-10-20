"""Simple exportable matplotlib figures directly from measured JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verisearch.io import read_json


def plot_results(result_path: Path, destination: Path) -> None:
    result = read_json(result_path)
    destination.mkdir(parents=True, exist_ok=True)
    config = result["config"]
    g = config["generation"]
    summaries = result["summaries"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png")
        fig.savefig(destination / f"{name}.pdf")
        plt.close(fig)

    colors = {
        "greedy": "black",
        "single": "gray",
        "self_consistency": "C0",
        "verifier": "C1",
        "adaptive_tuned": "C2",
        "oracle": "C3",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for policy, color in colors.items():
        count = 1 if policy in ("greedy", "single") else g["candidates"]
        points = [
            r
            for r in summaries
            if r["policy"] == policy
            and r["temperature"] == g["temperature"]
            and r["max_candidates"] == count
        ]
        points.sort(key=lambda r: r["budget"])
        if not points:
            continue
        ax.plot(
            [r["budget"] for r in points],
            [r["accuracy"] for r in points],
            marker="o",
            color=color,
            label="oracle upper bound (analysis)"
            if policy == "oracle"
            else policy.replace("_", " "),
            linestyle="--" if policy == "oracle" else "-",
        )
        ax.fill_between(
            [r["budget"] for r in points],
            [r["accuracy_ci95"][0] for r in points],
            [r["accuracy_ci95"][1] for r in points],
            alpha=0.08,
            color=color,
        )
    ax.set(xlabel="Generated-token cap per problem", ylabel="Answer accuracy", ylim=(0, 1))
    ax.set_xticks(g["budgets"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "accuracy_vs_budget")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for policy in ("self_consistency", "verifier", "oracle"):
        points = [
            r
            for r in summaries
            if r["policy"] == policy
            and r["temperature"] == g["temperature"]
            and r["budget"] == max(g["budgets"])
        ]
        points.sort(key=lambda r: r["max_candidates"])
        ax.plot(
            [r["max_candidates"] for r in points],
            [r["accuracy"] for r in points],
            marker="o",
            label="oracle upper bound (analysis)"
            if policy == "oracle"
            else policy.replace("_", " "),
            linestyle="--" if policy == "oracle" else "-",
        )
    ax.set(xlabel="Maximum candidate count", ylabel="Answer accuracy", ylim=(0, 1))
    ax.set_xticks(sorted({r["max_candidates"] for r in points}))
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "accuracy_vs_candidates")

    fig, ax = plt.subplots(figsize=(5, 4))
    bins = [b for b in result["verifier_metrics"]["reliability"] if b["count"]]
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="Mean predicted correctness", ylabel="Observed correctness", xlim=(0, 1), ylim=(0, 1)
    )
    save(fig, "reliability")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels, scores = np.asarray(result["verifier_labels"]), np.asarray(result["verifier_scores"])
    for label, name in ((False, "Incorrect"), (True, "Correct")):
        ax.hist(
            scores[labels == label],
            bins=[index / 20 for index in range(21)],
            alpha=0.55,
            label=name,
        )
    ax.set(xlabel="Verifier correctness probability", ylabel="Candidate count", xlim=(0, 1))
    ax.legend()
    save(fig, "score_distribution")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for budget in g["budgets"]:
        points = [
            r
            for r in summaries
            if r["policy"] == "adaptive"
            and r["budget"] == budget
            and r["temperature"] == g["temperature"]
        ]
        points.sort(key=lambda r: r["mean_tokens"])
        ax.plot(
            [r["mean_tokens"] for r in points],
            [r["accuracy"] for r in points],
            "o-",
            label=f"cap {budget}",
        )
        # Collapse thresholds with identical outcomes rather than overlapping labels.
        annotations: dict[tuple[float, float], list[str]] = {}
        for r in points:
            position = (round(r["mean_tokens"], 8), round(r["accuracy"], 12))
            annotations.setdefault(position, []).append(
                "off" if r["threshold"] is None else str(r["threshold"])
            )
        for index, (position, labels_at_point) in enumerate(annotations.items()):
            ax.annotate(
                ", ".join(labels_at_point),
                position,
                xytext=(4, -16 if index % 2 == 0 else 8),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set(xlabel="Mean generated tokens (prefix replay)", ylabel="Answer accuracy", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "adaptive_accuracy_vs_tokens")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    ablation = result["training_ablation"]
    axes[0].plot([r["fraction"] for r in ablation], [r["metrics"]["brier"] for r in ablation], "o-")
    axes[0].set(xlabel="Training problem fraction", ylabel="Candidate Brier score")
    axes[1].plot(
        [r["fraction"] for r in ablation], [r["selection"][0]["accuracy"] for r in ablation], "o-"
    )
    axes[1].set(xlabel="Training problem fraction", ylabel="Best-of-N accuracy", ylim=(0, 1))
    save(fig, "training_size")
