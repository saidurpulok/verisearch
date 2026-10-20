"""Deterministic unfavorable as well as favorable representative examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verisearch.data.schema import LabeledCandidate
from verisearch.policies.selection import majority


def error_report(
    pools: list[dict[str, Any]],
    scores: dict[str, dict[int, float]],
    output: Path,
) -> None:
    categories: dict[str, list[tuple[str, str]]] = {
        "Verifier rejects available correct solution": [],
        "Self-consistency succeeds; verifier fails": [],
        "Verifier succeeds; self-consistency fails": [],
        "All candidates incorrect": [],
        "High-confidence (>=0.9) false positive": [],
    }
    for pool in sorted(
        pools, key=lambda p: (p["problem_id"], p["budget"], p["seed"], p["pool_id"])
    ):
        records = [LabeledCandidate.from_dict(r) for r in pool["candidates"]]
        probabilities = scores[pool["pool_id"]]
        best = max(records, key=lambda r: probabilities[r.candidate.candidate_id])
        vote = majority([r.candidate for r in records])
        vote_correct = next((r.correct for r in records if r.candidate == vote), False)
        any_correct = any(r.correct for r in records)
        flags = [
            any_correct and not best.correct,
            vote_correct and not best.correct,
            best.correct and not vote_correct,
            not any_correct,
            not best.correct and probabilities[best.candidate.candidate_id] >= 0.9,
        ]
        description = (
            f"### {pool['problem_id']}\n\n{records[0].query.question}\n\n"
            f"Pool `{pool['pool_id']}`; cap={pool['budget']}; "
            f"seed={pool['seed']}; temperature={pool['temperature']}.\n\n"
            f"Gold (evaluation only): {records[0].gold_answer}; "
            f"verifier chose {best.candidate.candidate_id}; "
            f"self-consistency chose {vote.candidate_id if vote else None}.\n\n"
        )
        for record in records:
            c = record.candidate
            description += (
                f"- Candidate {c.candidate_id}: answer={c.answer}, "
                f"correct={record.correct}, score={probabilities[c.candidate_id]:.8f}, "
                f"tokens={c.tokens}\n\n"
                f"```text\n{c.text}\n```\n\n"
            )
        for name, flag in zip(categories, flags, strict=True):
            if flag:
                categories[name].append((pool["problem_id"], description))
    lines = [
        "# Held-out failure analysis\n",
        f"Scope: {len(pools)} supplied stochastic pools. First three distinct problem IDs "
        "per category, sorted by problem ID, budget, seed and pool ID. Counts distinguish "
        "unique problems from pool cases; categories may overlap. "
        "All traces shown, including truncated/malformed answers. "
        "Confidence 0.9 is a predeclared diagnostic, not a tuned decision.\n",
    ]
    for name, examples in categories.items():
        identities = {pid for pid, _ in examples}
        lines.append(f"## {name} ({len(identities)} problems; {len(examples)} pool cases)\n")
        representatives = []
        shown: set[str] = set()
        for pid, description in examples:
            if pid not in shown and len(shown) < 3:
                representatives.append(description)
                shown.add(pid)
        lines.extend(representatives or ["No examples observed.\n"])
    output.write_text("\n".join(lines))
