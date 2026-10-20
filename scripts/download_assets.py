"""Resolve immutable public revisions and download only required model assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
# Refuse to alter a run once data are prepared.
if (Path(config["output"]) / "config.json").exists():
    raise FileExistsError("Pin assets before preparing data; config is already frozen")
api = HfApi()
config["data"]["revision"] = api.dataset_info(
    config["data"]["dataset"], revision=config["data"].get("revision", "main")
).sha
for name in ("generation", "verifier"):
    section = config[name]
    section["revision"] = api.model_info(
        section["checkpoint"], revision=section.get("revision", "main")
    ).sha
args.config.write_text(yaml.safe_dump(config, sort_keys=False))
for name in ("generation", "verifier"):
    section = config[name]
    print(f"Downloading {section['checkpoint']} @ {section['revision']}", flush=True)
    snapshot_download(
        section["checkpoint"],
        revision=section["revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        max_workers=2,
    )
