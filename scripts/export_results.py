"""Export a small, reviewable pilot artifact without weights or full trace pools.

Run only after evaluation and plotting. JSON manifests retain the configuration,
split identities and checksums needed to audit the omitted large artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

from verisearch.config import load_config
from verisearch.experiment import check_config, load_pools
from verisearch.io import file_sha256, read_json, write_json


def export(config_path: Path, destination: Path) -> None:
    config = load_config(config_path)
    root = check_config(config)
    # Verify original pool integrity before publishing its summaries/manifests.
    load_pools(config, "fit")
    load_pools(config, "test")
    result = read_json(root / "results.json")
    if result["config"] != config.to_dict() or result["split"] != "test":
        raise ValueError("Results do not match the completed test experiment")
    if destination.exists():
        raise FileExistsError("Use a new destination; do not silently replace published evidence")
    files = [
        "config.json",
        "data/manifest.json",
        "fit_generation.json",
        "test_generation.json",
        "tuning.json",
        "verifier_validation.json",
        "results.json",
        "errors.md",
        "gpu_tests.xml",
    ]
    for fraction in config.verifier.fractions:
        files.append(f"verifiers/fraction_{fraction:g}/metadata.json")
    for name in (
        "accuracy_vs_budget",
        "accuracy_vs_candidates",
        "reliability",
        "score_distribution",
        "adaptive_accuracy_vs_tokens",
        "training_size",
    ):
        for extension in ("png", "pdf"):
            files.append(f"plots/{name}.{extension}")
    # Fail before creating a partial export if any required evidence is missing.
    for relative in [*files, "decisions.jsonl"]:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    # Deterministic gzip container; retains every per-problem policy outcome.
    with (root / "decisions.jsonl").open("rb") as source:
        with (destination / "decisions.jsonl.gz").open("wb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", mtime=0, filename="") as compressed:
                shutil.copyfileobj(source, compressed)
    write_json(
        destination / "artifact_manifest.json",
        {
            "source": config.output,
            "result_source_commit": result["git_commit"],
            "files_sha256": {
                str(path.relative_to(destination)): file_sha256(path)
                for path in sorted(destination.rglob("*"))
                if path.is_file()
            },
            "omitted": "Model weights, model caches, prepared questions and full trace pools. "
            "Failure examples and all policy decisions are included.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.config, arguments.output)
