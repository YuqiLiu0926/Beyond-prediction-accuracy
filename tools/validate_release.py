"""Validate the structure, metadata and optional hashes of the public release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_ANALYSES = [
    "analysis/critical_events/critical_event_features.jsonl",
    "analysis/risk_geometry/decision_features.jsonl",
    "analysis/operator_defect_features/decision_features.jsonl",
    "analysis/recoverability/decision_outcomes.json",
    "analysis/recoverability/surface/surface_summary.csv",
    "analysis/predictive_safety_filter/comparison_summary.json",
    "analysis/static_pde_design/decisions.csv",
    "closed_loop/gray_scott/independent_evaluation_summary.json",
    "closed_loop/thermal/independent_evaluation_summary.json",
    "numerical_qualification/thermal/one_sided_allowance.json",
]

FORBIDDEN_METADATA = [
    re.compile(r"\bGORC-\d+", re.IGNORECASE),
    re.compile(r"/home/lyq/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\(?:code|paper)\\", re.IGNORECASE),
    re.compile(r"nmi_experiments", re.IGNORECASE),
    re.compile(r"candidate prefix", re.IGNORECASE),
    re.compile(r"archived_source", re.IGNORECASE),
    re.compile(r"\bresults_exp", re.IGNORECASE),
    re.compile(r"\bexp[ef]\b", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\"),
]

MANIFEST_EXCLUDED_DIRS = {".git", "__pycache__", "external_data", "outputs", ".pytest_cache"}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_json_files(data_root: Path, errors: list[str]) -> tuple[int, int]:
    json_count = 0
    jsonl_count = 0
    for path in data_root.rglob("*.json"):
        json_count += 1
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid JSON: {path.relative_to(data_root)} ({exc})")
    for path in data_root.rglob("*.jsonl"):
        jsonl_count += 1
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except Exception as exc:
                    errors.append(
                        f"Invalid JSONL: {path.relative_to(data_root)}:{line_number} ({exc})"
                    )
                    break
    return json_count, jsonl_count


def validate_decision_archives(data_root: Path, errors: list[str]) -> Counter:
    archive_root = data_root / "decision_archives"
    counts: Counter = Counter()
    paths = sorted(archive_root.rglob("decision_data.npz"))
    require(len(paths) == 36, f"Expected 36 decision archives, found {len(paths)}", errors)
    required_arrays = {
        "z_sur", "z_hf", "margin_sur", "margin_hf",
        "J_sur", "J_hf", "U_all", "z_sur_ts_all", "z_hf_ts_all",
    }
    for path in paths:
        relative = path.relative_to(archive_root)
        split = relative.parts[0]
        folder = path.parent
        for name in ("decision_summary.csv", "summary.json", "run_manifest.json"):
            require((folder / name).is_file(), f"Missing {relative.parent / name}", errors)
        try:
            with np.load(path, allow_pickle=False) as archive:
                missing = required_arrays.difference(archive.files)
                if missing:
                    errors.append(f"{relative} lacks arrays {sorted(missing)}")
                    continue
                n = len(archive["z_hf"])
                require(len(archive["z_sur"]) == n, f"Risk length mismatch in {relative}", errors)
        except Exception as exc:
            errors.append(f"Cannot read {relative}: {exc}")
            continue
        counts[(split, n)] += 1
    expected = Counter({("discovery", 200): 12, ("detector_development", 100): 12, ("evaluation", 100): 12})
    require(counts == expected, f"Unexpected archive composition: {dict(counts)}", errors)
    return counts


def validate_metadata_terms(data_root: Path, errors: list[str]) -> None:
    extensions = {".json", ".jsonl", ".csv", ".md", ".txt"}
    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_METADATA:
            match = pattern.search(text)
            if match:
                errors.append(
                    f"Internal label or absolute path in {path.relative_to(data_root)}: {match.group(0)!r}"
                )
                break


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == "FILE_MANIFEST.csv":
            continue
        if any(part in MANIFEST_EXCLUDED_DIRS for part in relative.parts):
            continue
        files.add(relative.as_posix())
    return files


def verify_manifest(root: Path, errors: list[str]) -> int:
    manifest = root / "FILE_MANIFEST.csv"
    if not manifest.is_file():
        errors.append(f"Missing manifest: {manifest}")
        return 0
    checked = 0
    listed: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = row["relative_path"]
            if relative_path in listed:
                errors.append(f"Duplicate manifest entry: {relative_path}")
                continue
            listed.add(relative_path)
            path = root / relative_path
            if not path.is_file():
                errors.append(f"Manifest file missing: {relative_path}")
                continue
            if path.stat().st_size != int(row["size_bytes"]):
                errors.append(f"Manifest size mismatch: {relative_path}")
                continue
            if file_hash(path) != row["sha256"]:
                errors.append(f"Manifest hash mismatch: {relative_path}")
                continue
            checked += 1

    expected = manifest_files(root)
    missing_entries = sorted(expected - listed)
    stale_entries = sorted(listed - expected)
    for relative_path in missing_entries:
        errors.append(f"File absent from manifest: {relative_path}")
    for relative_path in stale_entries:
        errors.append(f"Manifest entry has no release file: {relative_path}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external_data",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    errors: list[str] = []

    require(data_root.is_dir(), f"Data package not found: {data_root}", errors)
    if errors:
        raise SystemExit("\n".join(errors))

    checkpoints = sorted((data_root / "checkpoints").rglob("*.pt"))
    datasets = sorted((data_root / "datasets").rglob("*.npz"))
    require(len(checkpoints) == 13, f"Expected 13 checkpoints, found {len(checkpoints)}", errors)
    require(len(datasets) == 4, f"Expected 4 trajectory datasets, found {len(datasets)}", errors)
    for relative in REQUIRED_ANALYSES:
        require((data_root / relative).is_file(), f"Missing required analysis: {relative}", errors)
    source_workbooks = sorted((data_root / "source_data").glob("*.xlsx"))
    require(len(source_workbooks) == 11, f"Expected 11 Source Data workbooks, found {len(source_workbooks)}", errors)

    archive_counts = validate_decision_archives(data_root, errors)
    json_count, jsonl_count = validate_json_files(data_root, errors)
    validate_metadata_terms(data_root, errors)

    checked_hashes = 0
    if args.verify_hashes:
        checked_hashes += verify_manifest(code_root, errors)
        checked_hashes += verify_manifest(data_root, errors)

    report = {
        "checkpoints": len(checkpoints),
        "trajectory_datasets": len(datasets),
        "decision_archives": sum(archive_counts.values()),
        "json_files": json_count,
        "jsonl_files": jsonl_count,
        "source_data_workbooks": len(source_workbooks),
        "verified_manifest_entries": checked_hashes,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
