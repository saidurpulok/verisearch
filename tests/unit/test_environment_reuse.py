import base64
import hashlib
import runpy
from pathlib import Path

import pytest


def test_cache_export_rejects_modified_contents(tmp_path: Path) -> None:
    verify = runpy.run_path("scripts/export_cached_wheels.py")["verify_record"]
    data = b"verified content"
    (tmp_path / "data.bin").write_bytes(data)
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    record = tmp_path / "RECORD"
    record.write_text(f"data.bin,sha256={encoded},{len(data)}\nRECORD,,\n")
    verify(tmp_path, record)
    (tmp_path / "data.bin").write_bytes(b"modified content")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path, record)


def test_cache_export_rejects_path_escape(tmp_path: Path) -> None:
    verify = runpy.run_path("scripts/export_cached_wheels.py")["verify_record"]
    root = tmp_path / "wheel"
    root.mkdir()
    (tmp_path / "outside").write_text("outside")
    record = root / "RECORD"
    record.write_text("../outside,,\n")
    with pytest.raises(ValueError, match="Invalid"):
        verify(root, record)
