import subprocess
import sys
from pathlib import Path

import pytest

from verisearch.cli import main
from verisearch.config import load_config
from verisearch.experiment import generate
from verisearch.io import write_json


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "verisearch.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "verifier" in result.stdout and "generate" in result.stdout


def test_config_and_test_gate(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"output: {tmp_path / 'run'}\n")
    config = load_config(cfg)
    write_json(Path(config.output) / "config.json", config.to_dict())
    with pytest.raises(ValueError, match="Freeze"):
        generate(config, "test")


def test_unknown_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("generation:\n  typo: 1\n")
    with pytest.raises(TypeError):
        load_config(cfg)
    with pytest.raises(SystemExit):
        main(["generate", "--partition", "test"])
