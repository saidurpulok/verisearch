"""Read verified unpacked uv wheels into a local wheelhouse without network IO.

Fallback for upstream artifact-URL changes, not the normal uv cache workflow.
Only exports large GPU packages pinned in uv.lock and compatible with this Python.
Never modifies uv's cache. An exported ZIP is a new container, not byte-identical
to the original download; its installed contents must pass the original RECORD.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import platform
import sys
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote

GPU_NAMES = {"torch", "triton", "cuda-bindings", "cuda-pathfinder"}


def verify_record(root: Path, record: Path) -> None:
    for relative, encoded, size in csv.reader(io.StringIO(record.read_text())):
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid/missing wheel member: {relative}")
        if encoded:
            algorithm, expected = encoded.split("=", 1)
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, algorithm).digest()
            actual_text = base64.urlsafe_b64encode(actual).decode().rstrip("=")
            if actual_text != expected:
                raise ValueError(f"Cached content checksum mismatch: {relative}")
        if size and path.stat().st_size != int(size):
            raise ValueError(f"Cached content size mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path, default=Path(".cache/wheelhouse"))
    args = parser.parse_args()
    locked = tomllib.loads(args.lock.read_text())["package"]
    args.output.mkdir(parents=True, exist_ok=True)
    minor = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine()
    exported = 0
    for package in locked:
        name, version = package["name"], package["version"]
        if name not in GPU_NAMES and not name.startswith("nvidia-"):
            continue
        for wheel in package.get("wheels", []):
            filename = unquote(wheel["url"].rsplit("/", 1)[-1])
            if machine not in filename or f"-{minor}t-" in filename:
                continue
            if f"-{minor}-{minor}-" not in filename and "-py3-none-" not in filename:
                continue
            target = args.output / filename
            if target.exists():
                continue
            cached_name = filename.split("-", 1)[1].removesuffix(".whl")
            sources = sorted((args.cache / "wheels-v6/index").glob(f"*/{name}/{cached_name}"))
            for source in sources:
                if not source.is_dir():
                    continue
                metadata_dirs = list(source.glob("*.dist-info"))
                if len(metadata_dirs) != 1:
                    continue
                metadata_dir = metadata_dirs[0]
                metadata = Parser().parsestr((metadata_dir / "METADATA").read_text())
                if metadata["Version"] != version:
                    raise ValueError("Cache version differs from locked version")
                verify_record(source, metadata_dir / "RECORD")
                temporary = target.with_suffix(".tmp")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                    for path in sorted(source.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(source))
                temporary.replace(target)
                print(f"Exported verified cached contents: {filename}", flush=True)
                exported += 1
                break
    print(f"Exported {exported} wheels; missing/incompatible entries are left alone.", flush=True)


if __name__ == "__main__":
    main()
