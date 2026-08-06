#!/usr/bin/env python3
"""Create deterministic upload archives for the code and companion data records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path


FIXED_TIMESTAMP = (2026, 8, 6, 0, 0, 0)


def manifested_files(root: Path) -> list[Path]:
    manifest = root / "FILE_MANIFEST.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Build the manifest before archiving: {manifest}")
    paths = [manifest]
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = root / row["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifested file is missing: {path}")
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def write_archive(root: Path, output: Path) -> int:
    files = manifested_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{root.name}/{relative}", date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    return len(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "neural-pde-decision-reliability-data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "release_upload",
    )
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    code_archive = output_dir / f"{code_root.name}-{args.version}.zip"
    data_archive = output_dir / f"{data_root.name}-{args.version}.zip"

    code_count = write_archive(code_root, code_archive)
    data_count = write_archive(data_root, data_archive)
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (code_archive, data_archive)),
        encoding="ascii",
    )
    print(f"code archive: {code_archive} ({code_count} files)")
    print(f"data archive: {data_archive} ({data_count} files)")
    print(f"checksums: {checksum_path}")


if __name__ == "__main__":
    main()
