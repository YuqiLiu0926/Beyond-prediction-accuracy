"""Attach the separately archived data package as ``external_data``."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Path to the extracted data package")
    parser.add_argument(
        "--link",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external_data",
        help="Link location (default: repository/external_data)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the data package instead of creating a directory link",
    )
    args = parser.parse_args()
    source = args.data_root.expanduser().resolve()
    link = args.link.expanduser().absolute()
    if not source.is_dir():
        raise SystemExit(f"Data package not found: {source}")
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)

    if args.copy:
        shutil.copytree(source, link)
    else:
        try:
            os.symlink(source, link, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(source)],
                check=True,
            )
    print(f"external_data -> {source}")


if __name__ == "__main__":
    main()

