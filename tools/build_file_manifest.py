"""Create SHA-256 file manifests for the code and companion data packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXCLUDED_DIRS = {".git", "__pycache__", "external_data", "outputs", ".pytest_cache"}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if not path.is_file() or relative.as_posix() == "FILE_MANIFEST.csv":
            continue
        yield path, relative


def write_manifest(root: Path) -> int:
    output = root / "FILE_MANIFEST.csv"
    rows = []
    for path, relative in iter_files(root):
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "neural-pde-decision-reliability-data",
    )
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    data_root = args.data_root.resolve()
    print(f"code files: {write_manifest(code_root)}")
    print(f"data files: {write_manifest(data_root)}")


if __name__ == "__main__":
    main()

