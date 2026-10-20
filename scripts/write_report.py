"""Render measured pilot tables; never synthesize or estimate missing results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from verisearch.config import load_config
from verisearch.experiment import check_config, load_pools, summarize
from verisearch.io import read_json, read_jsonl


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def metric(value: float | None) -> str:
    return "undefined (single class)" if value is None else f"{value:.4f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_report(config_path: Path, destination: Path) -> None:
    config = load_config(config_path)
    root = check_config(config)
    result = read_json(root / "results.json")
    gpu_report = ElementTree.parse(root / "gpu_tests.xml")
    suites = list(gpu_report.iter("testsuite"))
    if not suites or sum(int(s.get("tests", "0")) for s in suites) < 1:
        raise ValueError("No real GPU test execution is recorded")
    if any(int(s.get(key, "0")) for s in suites for key in ("failures", "errors", "skipped")):
        raise ValueError("GPU validation has not passed without skips")
    fit, test = load_pools(config, "fit"), load_pools(config, "test")
    if result["config"] != config.to_dict():
        raise ValueError("Report configuration differs from measured results")
    summaries, g = result["summaries"], config.generation
    counts = result["data_manifest"]["counts"]
    hardware = result["generation_manifest"]["hardware"]
    ablation = result["training_ablation"]
    full = next(a for a in ablation if a["fraction"] == 1.0)["training"]
    verifier = result["verifier_metrics"]
    fit_composition = result["fit_answer_composition"]
    tuned = result["tuning"]["chosen_threshold"]
    tuning_text = "disabled" if tuned is None else str(tuned)
    sections = [
        "# VeriSearch v0.1 measured pilot results",
        "**OBSERVED RESULT.** These measurements come from real local model generation and "
        "full encoder fine-tuning. This is a small, predeclared GSM8K pilot, not a full "
        "benchmark evaluation. All percentages below concern exact extracted final answers, "
        "not independently checked reasoning steps.",
        f"Run timestamp (UTC): `{result['timestamp_utc']}`; evaluation source commit: "
        f"`{result['git_commit']}`. Frozen configuration: [pilot.yaml](configs/pilot.yaml). "
        "Machine-readable evidence and checksums: [results/pilot](results/pilot/).",
        "**LIMITATION: answer-format shortcut.** Fit data includes "
        f"{fit_composition['train']['present_incorrect']} extracted incorrect training answers "
        f"and {fit_composition['train']['answer_missing']} missing training answers. "
        f"Validation includes {fit_composition['validation']['present_incorrect']} extracted "
        f"incorrect answers and {fit_composition['validation']['answer_missing']} missing answers. "
        "The diagnostic below checks this confound without changing the frozen policies. "
        "High candidate-level scores alone cannot establish mathematical verification.",
        "## Environment and checkpoints",
        f"Python {result['python']}; {result['platform']}. GPU: {hardware['gpu']}; "
        f"Torch {hardware['torch']}, CUDA runtime {hardware['cuda_runtime']}. "
        f"Generator peak allocated memory: {hardware['peak_allocated_mib']:.1f} MiB. "
        "Peak allocation excludes driver reservations and other processes.",
        table(
            ["Component", "Checkpoint", "Immutable revision"],
            [
                ["GSM8K", config.data.dataset, config.data.revision],
                ["Generator", g.checkpoint, g.revision],
                ["Verifier base", config.verifier.checkpoint, config.verifier.revision],
            ],
        ),
        f"Problem-level partitions: {counts['train']} train / {counts['validation']} validation / "
        f"{counts['test']} official-test problems; split seed {config.data.split_seed}. "
        f"Generated {len(fit)} fit pools ({sum(len(p['candidates']) for p in fit)} traces) "
        f"and {len(test)} test pools ({sum(len(p['candidates']) for p in test)} traces). "
        "Pools at different caps/seeds can contain repeated prefixes; these counts are not "
        "independent examples. Verifier fitting uses only one declared pool per problem.",
        f"Generation: at most {g.candidates} traces/problem, at most {g.max_new_tokens} new "
        f"tokens/trace, temperature {g.temperature}, top-p {g.top_p}, top-k {g.top_k}, "
        f"sampling seeds {g.seeds}, total token caps {g.budgets}. Greedy uses one trace. "
        "Every recorded pool respects its total generated-token cap.",
        "## Verifier learning and candidate-level evaluation",
        table(
            ["Partition", "Problems", "Traces", "Correct", "Incorrect", "Positive fraction"],
            [
                [
                    name,
                    b["problems"],
                    b["candidates"],
                    b["correct"],
                    b["incorrect"],
                    percent(b["positive_fraction"]),
                ]
                for name, b in [
                    ("Train", full["train_balance"]),
                    ("Validation", full["validation_balance"]),
                ]
            ],
        ),
        f"Full fine-tuning uses AdamW, learning rate {config.verifier.learning_rate:g}, "
        f"batch size {config.verifier.batch_size}, {config.verifier.epochs} epochs, seed "
        f"{config.verifier.seed}, maximum input length {config.verifier.max_length}. "
        "Minimum validation cross-entropy chooses the checkpoint. "
        f"Input truncation: train {percent(full['truncation']['train']['fraction_truncated'])}, "
        f"validation {percent(full['truncation']['validation']['fraction_truncated'])}.",
        f"Test verifier metrics use {verifier['count']} traces from one maximum-budget pool "
        f"per test problem at seed {g.seeds[0]}; correctness prevalence "
        f"{percent(verifier['positive_fraction'])}.",
        table(
            ["Metric", "Observed value"],
            [
                [name, metric(verifier[key])]
                for name, key in [
                    ("Accuracy at 0.5", "accuracy_at_0.5"),
                    ("AUROC", "auroc"),
                    ("AUPRC (average precision)", "auprc"),
                    ("Brier score", "brier"),
                    ("ECE (10 bins)", "ece"),
                ]
            ],
        ),
        "[Reliability plot](results/pilot/plots/reliability.png) and "
        "[score distributions](results/pilot/plots/score_distribution.png). "
        "**INTERPRETATION.** Ranking, probability quality and downstream selection measure "
        "different properties. Classification accuracy alone is insufficient under imbalance.",
        "## Fixed generated-token budgets",
        "**OBSERVED RESULT.** Stochastic rows average both declared generation seeds on the "
        "same test problems. Intervals are 95% percentile bootstrap intervals from 2000 "
        "problem resamples, retaining all seed outcomes within each sampled problem. "
        "They are individual accuracy intervals, not significance tests of policy differences.",
    ]
    diagnostic = result["answer_presence_diagnostic"]
    present_metrics = diagnostic["verifier_metrics_on_present_answers"]
    sections.insert(
        sections.index("## Fixed generated-token budgets"),
        "### Exploratory answer-presence diagnostic\n\n"
        + table(
            ["Test trace category", "Count"],
            [
                ["Extracted and correct", diagnostic["present_correct"]],
                ["Extracted and incorrect", diagnostic["present_incorrect"]],
                ["Missing extracted answer", diagnostic["answer_missing"]],
            ],
        )
        + "\n\nA trivial classifier assigning probability 1 to extracted answers and 0 "
        "to missing answers has accuracy "
        + percent(diagnostic["presence_heuristic_metrics"]["accuracy_at_0.5"])
        + ", AUROC "
        + metric(diagnostic["presence_heuristic_metrics"]["auroc"])
        + ". This exploratory diagnostic was motivated by validation composition and "
        "was not used for policy selection, threshold tuning or hyperparameters. "
        + (
            "Among extracted answers only, learned-verifier AUROC is "
            + metric(present_metrics["auroc"])
            + ", Brier is "
            + metric(present_metrics["brier"])
            + "."
            if present_metrics
            else "No extracted answers are available for conditional verifier metrics."
        ),
    )
    policies = ("greedy", "single", "self_consistency", "verifier", "adaptive_tuned", "oracle")
    main = [
        r
        for r in summaries
        if r["policy"] in policies
        and r["temperature"] == g.temperature
        and r["max_candidates"] == (1 if r["policy"] in ("greedy", "single") else g.candidates)
    ]
    main.sort(key=lambda r: (r["budget"], policies.index(r["policy"])))
    sections.append(
        table(
            [
                "Cap",
                "Policy",
                "Accuracy [95% CI]",
                "Mean / median tokens",
                "Mean candidates",
                "Generation s",
            ],
            [
                [
                    r["budget"],
                    r["policy"],
                    f"{percent(r['accuracy'])} "
                    f"[{percent(r['accuracy_ci95'][0])}, "
                    f"{percent(r['accuracy_ci95'][1])}]",
                    f"{r['mean_tokens']:.1f} / {r['median_tokens']:.1f}",
                    f"{r['mean_candidates']:.2f}",
                    f"{r['mean_generation_seconds']:.2f}",
                ]
                for r in main
            ],
        )
    )
    sections.append(
        "Paired policy accuracy differences (first minus second), in percentage points, "
        "with problem-cluster bootstrap intervals. These are predeclared comparisons; "
        "the report makes no statistical significance claim."
    )
    sections.append(
        table(
            ["Cap", "First policy", "Second policy", "Difference [95% CI] (pp)"],
            [
                [
                    r["budget"],
                    r["first"],
                    r["second"],
                    f"{100 * r['accuracy_difference']:+.1f} "
                    f"[{100 * r['difference_ci95'][0]:+.1f}, {100 * r['difference_ci95'][1]:+.1f}]",
                ]
                for r in result["paired_comparisons"]
            ],
        )
    )
    sections.extend(
        [
            "**LIMITATION.** Oracle is an analysis-only any-correct upper bound and cannot "
            "be deployed. Adaptive token/latency figures are offline prefix replay. Generation "
            "latencies are synchronized per-call measurements; verifier time, prompt prefill "
            "accounting and model loading are not included in the token budget. No end-to-end "
            "speedup or equal-FLOPs claim follows from these figures.",
            "[Accuracy vs token cap](results/pilot/plots/accuracy_vs_budget.png); "
            "[accuracy vs candidate count](results/pilot/plots/accuracy_vs_candidates.png).",
            "## Predeclared ablations",
            "### Training data size",
            table(
                [
                    "Problem fraction",
                    "Train problems",
                    "Train traces",
                    "AUROC",
                    "Brier",
                    "Best-of-N accuracy (seed 17)",
                ],
                [
                    [
                        a["fraction"],
                        a["training"]["train_balance"]["problems"],
                        a["training"]["train_balance"]["candidates"],
                        metric(a["metrics"]["auroc"]),
                        metric(a["metrics"]["brier"]),
                        percent(a["selection"][0]["accuracy"]),
                    ]
                    for a in ablation
                ],
            ),
            "**LIMITATION.** Nested problem subsets use the same initialization seed and three "
            "epochs. Larger subsets also receive more optimizer updates; this does not isolate "
            "a causal data-size effect. One training seed cannot establish monotonic scaling. "
            "[Training-size plot](results/pilot/plots/training_size.png).",
            "### Sampling temperature",
            "The following table uses the middle budget and **only seed 17 for every "
            "temperature**, including the primary temperature, so the comparison is matched.",
        ]
    )
    decisions = read_jsonl(root / "decisions.jsonl")
    temperature_rows = [
        r
        for r in decisions
        if r["seed"] == g.seeds[0]
        and r["budget"] == g.budgets[len(g.budgets) // 2]
        and r["policy"] in ("self_consistency", "verifier", "oracle")
        and r["max_candidates"] == g.candidates
    ]
    temperature_summaries = summarize(temperature_rows, config)
    temperature_summaries.sort(key=lambda r: (r["temperature"], r["policy"]))
    sections.append(
        table(
            ["Temperature", "Policy", "Accuracy", "Mean tokens"],
            [
                [r["temperature"], r["policy"], percent(r["accuracy"]), f"{r['mean_tokens']:.1f}"]
                for r in temperature_summaries
            ],
        )
    )
    sections.extend(
        [
            "### Adaptive threshold",
            f"Validation froze threshold **{tuning_text}** before any test generation, using "
            f"accuracy minus {config.evaluation.token_penalty} times mean "
            "consumed budget fraction. "
            "The full validation grid and selection are in "
            "[tuning.json](results/pilot/tuning.json). "
            "The test grid below is descriptive; no test-optimal threshold is selected.",
        ]
    )
    adaptive = [
        r
        for r in summaries
        if r["policy"] == "adaptive"
        and r["temperature"] == g.temperature
        and r["budget"] == max(g.budgets)
    ]
    sections.append(
        table(
            ["Threshold", "Accuracy", "Mean tokens", "Mean candidates"],
            [
                [
                    "disabled" if r["threshold"] is None else r["threshold"],
                    percent(r["accuracy"]),
                    f"{r['mean_tokens']:.1f}",
                    f"{r['mean_candidates']:.2f}",
                ]
                for r in adaptive
            ],
        )
    )
    sections.extend(
        [
            "[Adaptive accuracy vs consumed tokens]"
            "(results/pilot/plots/adaptive_accuracy_vs_tokens.png). "
            "Candidate-count results for N=1,2,4 and all budgets are included in results.json.",
            "## Failure analysis and interpretation",
            "[Structured failure report](results/pilot/errors.md) selects the first examples "
            "in stable problem/budget/seed order for each category across all primary-temperature "
            "budgets and both seeds, including unfavorable cases. Categories "
            "may overlap; the report includes every candidate from each selected pool.",
            "**INTERPRETATION.** Policy comparisons describe this fixed pilot, generator, "
            "prompt and verifier. They do not establish algorithmic novelty, a universal benefit "
            "of verification, or a statistically significant improvement. A verifier can prefer "
            "incorrect traces even when a correct candidate is available; oracle headroom alone "
            "does not show that a learnable verifier can recover it.",
            f"**LIMITATION.** Only {counts['test']} held-out problems, short traces, one domain, "
            f"{len(g.seeds)} generation "
            "seeds and one training seed limit precision and generalization. Exact-answer labels "
            "can reward invalid reasoning, and the verifier can learn stylistic shortcuts. Local "
            "problem-level isolation does not rule out contamination in pretrained models. "
            "See [LIMITATIONS.md](docs/LIMITATIONS.md).",
            "## Reproduction and verification",
            "See [README.md](README.md) for stage commands and "
            "[METHODOLOGY.md](docs/METHODOLOGY.md) "
            "for data boundaries, metrics and uncertainty. Weights and full trace pools remain "
            "outside Git; hashes and configuration are tracked. Compressed per-problem policy "
            "outcomes permit reanalysis without rerunning inference. "
            "[GPU test evidence](results/pilot/gpu_tests.xml) records actual "
            "pretrained generation, "
            "scoring, seeded repeatability and live adaptive stopping. Final engineering checks "
            "are recorded in [VALIDATION.md](docs/VALIDATION.md); this report alone is not "
            "a release acceptance declaration.",
        ]
    )
    destination.write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.config, args.output)
