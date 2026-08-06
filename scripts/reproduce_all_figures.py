"""Regenerate all quantitative main and Extended Data figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = [
    "figure_2_margin_plane.py",
    "figure_3_tail_risk.py",
    "figure_4_recoverability.py",
    "figure_5_gray_scott_closed_loop.py",
    "figure_6_thermal_transfer.py",
]


def run(script: Path, *arguments: str) -> None:
    command = [sys.executable, str(script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    figure_dir = ROOT / "scripts" / "figures"
    for name in FIGURE_SCRIPTS:
        run(figure_dir / name)
    run(
        figure_dir / "reproduce_extended_data_figures.py",
        "--output-dir",
        str(ROOT / "outputs" / "extended_data"),
        "--suffix",
        "_reproduced",
    )

    expected = [ROOT / "outputs" / "figures" / f"Figure_{index}.pdf" for index in range(2, 7)]
    expected.extend(
        ROOT / "outputs" / "extended_data" / f"Extended_Data_Fig_{index}_reproduced.pdf"
        for index in range(1, 7)
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing reproduced figures: {missing}")
    print(f"Reproduced {len(expected)} quantitative figures.")


if __name__ == "__main__":
    main()

