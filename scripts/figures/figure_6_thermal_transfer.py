"""Generate the final thermal transfer figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import figure_6_refined_panels as refined_panels
import figure_6_certification_time as certification_time
import figure_6_source_data as source_data


OUT_STEM = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "Figure_6"
_independent_endpoint_panel = refined_panels.heldout_endpoint_panel
source_data.SOURCE_DIR = (
    Path(__file__).resolve().parents[2] / "external_data" / "source_data" / "csv"
)


def independent_evaluation_panel(fig, spec) -> None:
    _independent_endpoint_panel(fig, spec)
    for ax in fig.axes:
        for label in ax.texts:
            if label.get_text() == "Held-out validation set":
                label.set_text("Independent evaluation episodes")


def save_figure_stable(fig, output):
    """Export the large TIFF without the unstable LZW encoder on Windows."""
    path = Path(output)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(fig)
    with Image.open(path.with_suffix(".png")) as image:
        image.convert("RGB").save(
            path.with_suffix(".tiff"), dpi=(600, 600), compression="raw"
        )
    return path


def main() -> None:
    refined_panels.heldout_endpoint_panel = independent_evaluation_panel
    certification_time.OUT_STEM = OUT_STEM
    certification_time.save_figure = save_figure_stable
    certification_time.main()


if __name__ == "__main__":
    main()
