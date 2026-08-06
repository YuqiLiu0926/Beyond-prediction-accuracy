"""Load the decision and recoverability evidence used in Figure 4."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    MAX_HEIGHT_IN,
    PDE_COLORS,
    clean_axes,
    panel_label,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "Figure_4_data_check.pdf"

EVIDENCE_DIR = ROOT / "external_data" / "analysis" / "recoverability"
FLOOR_PATH = EVIDENCE_DIR / "decision_outcomes.json"
POSTHOC_PATH = EVIDENCE_DIR / "hf_replay_audit.json"
CONTINUATION_ROWS_PATH = EVIDENCE_DIR / "continuation_rows.jsonl"
SPLIT_PATH = EVIDENCE_DIR / "state_split.json"
RECOVERY_ROWS_PATH = EVIDENCE_DIR / "reference_search_rows.jsonl"
SATURATION_PATH = EVIDENCE_DIR / "search_saturation.json"
ABLATION_PATH = EVIDENCE_DIR / "sequence_definition_ablation.json"

PDE_LABELS = {"burgers": "Burgers", "grayscott": "Gray-Scott", "kolmogorov": "Kolmogorov"}
PDE_ORDER = ("burgers", "grayscott", "kolmogorov")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_rows() -> list[dict]:
    floor = read_json(FLOOR_PATH)
    labels = {
        "raw_surrogate": "Raw surrogate",
        "residual19": "Residual evidence",
        "hf_candidate_oracle": "HF reference labels",
    }
    rows = []
    for policy, label in labels.items():
        values = floor["policies"][policy]["overall"]
        unsafe = int(values["executed_unsafe"])
        unavailable = int(values["continuation_uncertified"])
        rows.append(
            {
                "policy": policy,
                "label": label,
                "unsafe_execution": unsafe,
                "continuation_unavailable": unavailable,
                "overall_failure": unsafe + unavailable,
                "n_scenarios": int(values["n"]),
            }
        )
    assert [row["unsafe_execution"] for row in rows] == [17, 0, 0]
    assert [row["continuation_unavailable"] for row in rows] == [33, 48, 44]
    assert [row["overall_failure"] for row in rows] == [50, 48, 44]
    return rows


def continuation_margin_rows() -> list[dict]:
    rows = read_jsonl(CONTINUATION_ROWS_PATH)
    split = read_json(SPLIT_PATH)["assignments"]
    posthoc = read_json(POSTHOC_PATH)
    mechanism = posthoc["unique_continuation_test_mechanism"]
    oracle = posthoc["forced_execution_counterfactual"]["hf_candidate_oracle"]

    unsafe_keys = {
        (pde, int(index))
        for pde in PDE_ORDER
        for index in mechanism[pde]["unsafe_ic_indices"]
    }
    exact_safe_abstention_keys: set[tuple[str, int]] = set()
    output = []
    for pde in PDE_ORDER:
        test_ids = set(int(value) for value in split[pde]["test_ic_idx"])
        pde_rows = [row for row in rows if row["pde"] == pde and int(row["ic_idx"]) in test_ids]
        safe_uncertified = [
            row for row in pde_rows
            if not int(row["continuation_certified"]) and int(row["continuation_hf_safe"])
        ]
        exact_safe_count = int(oracle[pde]["uncertified_but_hf_safe"])
        assert exact_safe_count % 4 == 0
        assert exact_safe_count // 4 <= len(safe_uncertified)
        for row in safe_uncertified[: exact_safe_count // 4]:
            exact_safe_abstention_keys.add((pde, int(row["ic_idx"])))

        for row in pde_rows:
            key = (pde, int(row["ic_idx"]))
            normalized_margin = 100.0 * (
                float(row["z_limit_phys"]) - float(row["z_hf_selected"])
            ) / float(row["z_limit_phys"])
            role = "background"
            if key in unsafe_keys:
                role = "unsafe_continuation"
            elif key in exact_safe_abstention_keys:
                role = "safe_certificate_abstention"
            output.append(
                {
                    "pde": pde,
                    "ic_idx": int(row["ic_idx"]),
                    "z_limit": float(row["z_limit_phys"]),
                    "selected_continuation_hf_risk": float(row["z_hf_selected"]),
                    "normalized_hf_margin_percent": normalized_margin,
                    "continuation_certified": int(row["continuation_certified"]),
                    "continuation_hf_safe": int(row["continuation_hf_safe"]),
                    "exact_label_floor_role": role,
                }
            )

    assert len(output) == 180
    assert sum(row["exact_label_floor_role"] == "unsafe_continuation" for row in output) == 10
    assert sum(row["exact_label_floor_role"] == "safe_certificate_abstention" for row in output) == 1
    assert sum(row["normalized_hf_margin_percent"] < 0 for row in output) == 10
    return output


def action_stress_rows() -> list[dict]:
    recovery = read_jsonl(RECOVERY_ROWS_PATH)
    saturation = read_json(SATURATION_PATH)
    saturation_by_key = {(row["pde"], int(row["ic_idx"])): row for row in saturation["rows"]}
    challenge = [row for row in recovery if not int(row["continuation_only_hf_safe"])]
    assert len(challenge) == 10

    output = []
    for row in challenge:
        key = (row["pde"], int(row["ic_idx"]))
        boundary = float(row["boundary"])
        fixed = (
            ("K=5", "K5", int(row["continuation_only_hf_safe"]), float(row["continuation_only_z_hf"]), 1),
            ("K=15 HF oracle", "K15", int(row["k15_hf_oracle_safe"]),
             float(row["k15_hf_oracle_z_hf"]), 1),
            ("HF-CEM oracle", "HF-CEM", int(row["hf_oracle_safe"]),
             float(row["hf_oracle_z_hf"]), 1),
        )
        for label, short_label, safe, risk, within_contract in fixed:
            output.append(
                {
                    "pde": row["pde"],
                    "ic_idx": int(row["ic_idx"]),
                    "intervention": label,
                    "short_label": short_label,
                    "within_frozen_contract": within_contract,
                    "authority_multiplier": 1.0,
                    "hf_safe": safe,
                    "hf_risk": risk,
                    "safety_limit": boundary,
                    "normalized_margin_percent": 100.0 * (boundary - risk) / boundary,
                }
            )

        diagnostic = saturation_by_key[key]
        for multiplier in (1.25, 2.0, 4.0):
            item = next(
                value for value in diagnostic["authority_counterfactual"]
                if np.isclose(float(value["multiplier"]), multiplier)
            )
            output.append(
                {
                    "pde": row["pde"],
                    "ic_idx": int(row["ic_idx"]),
                    "intervention": f"{multiplier:g}x actuator authority",
                    "short_label": f"{multiplier:g}x authority",
                    "within_frozen_contract": 0,
                    "authority_multiplier": multiplier,
                    "hf_safe": int(item["safe"]),
                    "hf_risk": float(item["z_hf"]),
                    "safety_limit": boundary,
                    "normalized_margin_percent": 100.0 * (boundary - float(item["z_hf"])) / boundary,
                }
            )

    labels = ("K5", "K15", "HF-CEM", "1.25x authority", "2x authority", "4x authority")
    safe_counts = [sum(row["short_label"] == label and row["hf_safe"] for row in output) for label in labels]
    assert safe_counts == [0, 0, 0, 1, 3, 4]
    assert len(output) == 60
    return output


def ablation_rows() -> list[dict]:
    ablation = read_json(ABLATION_PATH)
    modes = (
        ("full_plan", "Full plan", 1),
        ("continuation_preserving", "Complete candidate", 1),
        ("block_only", "Block only", 0),
    )
    rows = []
    for key, label, retained in modes:
        values = ablation["by_mode"][key]
        rows.append(
            {
                "mode": key,
                "label": label,
                "candidate_releases": int(values["candidate_releases"]),
                "n_proposed_blocks": 96,
                "execution_rate": float(values["candidate_release_fraction"]),
                "unsafe_episodes": int(values["unsafe_episodes"]),
                "n_episodes": int(values["episodes"]),
                "continuation_available_next_update": retained,
            }
        )
    assert [row["candidate_releases"] for row in rows] == [77, 82, 82]
    assert [row["unsafe_episodes"] for row in rows] == [0, 0, 0]
    return rows


def panel_a_failure_trajectories(ax) -> None:
    rows = failure_rows()
    x = np.arange(3, dtype=float)
    unsafe = np.asarray([row["unsafe_execution"] for row in rows])
    unavailable = np.asarray([row["continuation_unavailable"] for row in rows])
    total = np.asarray([row["overall_failure"] for row in rows])

    series = (
        (unsafe, -0.045, COLORS["unsafe"], "s", "Unsafe"),
        (unavailable, 0.0, COLORS["unavailable"], "o", "Unresolved"),
        (total, 0.045, COLORS["charcoal"], "D", "Total"),
    )
    for values, offset, color, marker, label in series:
        ax.plot(x + offset, values, color=color, linewidth=1.25, marker=marker,
                markersize=4.5, label=label, zorder=3)
        for xpos, value in zip(x + offset, values):
            if label == "Unresolved" and value >= 40:
                text_y, va = value - 2.6, "top"
            else:
                text_y, va = value + 2.1, "bottom"
            ax.text(xpos, text_y, str(int(value)), ha="center", va=va,
                    fontsize=5.4, color=color)

    ax.annotate(
        "Unsafe acceptance removed",
        xy=(1 - 0.045, 0), xytext=(0.38, 15),
        arrowprops={"arrowstyle": "->", "color": COLORS["unsafe"], "linewidth": 0.65},
        color=COLORS["unsafe"], fontsize=5.4, ha="center",
    )
    ax.annotate(
        "Only 6 fewer failures",
        xy=(2 + 0.045, 44), xytext=(1.50, 57),
        arrowprops={"arrowstyle": "->", "color": COLORS["charcoal"], "linewidth": 0.65},
        color=COLORS["charcoal"], fontsize=5.4, ha="center",
    )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.set_xlim(-0.18, 2.22)
    ax.set_ylim(-4, 64)
    ax.set_ylabel("Intervention failures / 720")
    ax.set_title("Detection cannot create a safe continuation",
                 loc="left", fontsize=6.4, fontweight="bold", pad=3)
    ax.legend(loc="upper left", ncols=3, fontsize=5.1, columnspacing=0.75,
              handlelength=1.2, handletextpad=0.3)
    clean_axes(ax, grid=False)
    panel_label(ax, "a", x=-0.13, y=1.03)


def panel_b_continuation_margins(ax) -> None:
    rows = continuation_margin_rows()
    y_lookup = {"burgers": 2.0, "grayscott": 1.0, "kolmogorov": 0.0}
    rng = np.random.default_rng(20260716)
    for pde in PDE_ORDER:
        group = [row for row in rows if row["pde"] == pde]
        jitter = rng.uniform(-0.18, 0.18, size=len(group))
        x = np.asarray([row["normalized_hf_margin_percent"] for row in group])
        y = y_lookup[pde] + jitter
        ax.scatter(x, y, s=10, color=PDE_COLORS[pde], alpha=0.27,
                   edgecolors="none", rasterized=True, zorder=1)

    unsafe = [row for row in rows if row["exact_label_floor_role"] == "unsafe_continuation"]
    abstain = [row for row in rows if row["exact_label_floor_role"] == "safe_certificate_abstention"]
    ax.scatter(
        [row["normalized_hf_margin_percent"] for row in unsafe],
        [y_lookup[row["pde"]] for row in unsafe],
        s=26, marker="D", color=COLORS["unsafe"], edgecolor="white", linewidth=0.35,
        label="Unsafe under HF replay", zorder=4,
    )
    ax.scatter(
        [row["normalized_hf_margin_percent"] for row in abstain],
        [y_lookup[row["pde"]] for row in abstain],
        s=36, marker="o", facecolor="white", edgecolor=COLORS["certificate"], linewidth=1.2,
        label="Safe but not accepted", zorder=5,
    )
    ax.axvspan(-7, 0, color=COLORS["light_red"], alpha=0.60, linewidth=0, zorder=0)
    ax.axvline(0, color=COLORS["charcoal"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(-0.15, 2.38, "unsafe", ha="right", fontsize=5.2, color=COLORS["unsafe"])
    ax.text(0.15, 2.38, "safe", ha="left", fontsize=5.2, color=COLORS["controller"])
    ax.text(-5.8, 0.42, "10 states\n40/44 unsafe continuations", color=COLORS["unsafe"],
            fontsize=5.4, ha="left", va="bottom")
    ax.annotate(
        "1 state\n4/44 not accepted",
        xy=(abstain[0]["normalized_hf_margin_percent"], y_lookup["grayscott"]),
        xytext=(3.2, 0.42), fontsize=5.2, color=COLORS["certificate"],
        arrowprops={"arrowstyle": "-", "color": COLORS["certificate"], "linewidth": 0.55},
    )
    ax.set_yticks([2, 1, 0], [PDE_LABELS[pde] for pde in PDE_ORDER])
    ax.set_xlim(-7, 18)
    ax.set_ylim(-0.45, 2.55)
    ax.set_xlabel(r"Selected-continuation HF margin $(z_{\rm lim}-Z_{\rm HF})/z_{\rm lim}$ (%)")
    ax.set_title("HF labels expose the remaining intervention floor",
                 loc="left", fontsize=6.4, fontweight="bold", pad=3)
    clean_axes(ax, grid=False)
    panel_label(ax, "b", x=-0.045, y=1.03)


def panel_c_action_stress(ax) -> None:
    rows = action_stress_rows()
    state_keys = []
    for pde in ("burgers", "grayscott"):
        state_keys.extend(sorted({(row["pde"], row["ic_idx"]) for row in rows if row["pde"] == pde}))
    labels = ("K5", "K15", "HF-CEM", "1.25x authority", "2x authority", "4x authority")
    matrix = np.zeros((len(state_keys), len(labels)), dtype=int)
    by_key = {(row["pde"], row["ic_idx"], row["short_label"]): row for row in rows}
    for i, (pde, ic_idx) in enumerate(state_keys):
        for j, label in enumerate(labels):
            matrix[i, j] = int(by_key[(pde, ic_idx, label)]["hf_safe"])

    cmap = ListedColormap([COLORS["light_red"], COLORS["light_green"]])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "o" if matrix[i, j] else "x", ha="center", va="center",
                    fontsize=5.5, fontweight="bold",
                    color=COLORS["controller"] if matrix[i, j] else COLORS["unsafe"])
    safe_counts = np.sum(matrix, axis=0)
    for j, count in enumerate(safe_counts):
        ax.text(j, -1.02, f"{int(count)}/10", ha="center", va="bottom",
                fontsize=5.5, fontweight="bold")
    ax.axvline(2.5, color=COLORS["charcoal"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(1.0, -1.78, "Original horizon and actuator bounds", ha="center",
            fontsize=5.4, color=COLORS["charcoal"])
    ax.text(4.0, -1.78, "Wider actuator range", ha="center",
            fontsize=5.4, color=COLORS["unavailable"])
    ax.set_xticks(np.arange(len(labels)), ["K=5", "K=15\nHF-selected", "CEM\nHF-selected",
                                                   "1.25x", "2x", "4x"])
    ax.set_yticks(
        np.arange(len(state_keys)),
        [f"{'B' if pde == 'burgers' else 'GS'}-{ic_idx}" for pde, ic_idx in state_keys],
    )
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(9.5, -2.05)
    ax.set_title("Broader search found no safe control within the original limits",
                 loc="left", fontsize=6.4, fontweight="bold", pad=3)
    clean_axes(ax, grid=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "c", x=-0.055, y=1.07)


def arrow(ax, start, end, color, *, connectionstyle="arc3", linewidth=0.8) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=6.5, linewidth=linewidth,
            color=color, connectionstyle=connectionstyle, transform=ax.transAxes,
            clip_on=False,
        )
    )


def segment(ax, x, y, width, height, face, edge, text, *, fontsize=5.4) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor=face, edgecolor=edge,
                           linewidth=0.8, transform=ax.transAxes))
    ax.text(x + width / 2, y + height / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color=COLORS["charcoal"])


def panel_d_recursive_certificate(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "d", x=-0.035, y=1.02)
    ax.text(0.0, 1.01, "Both branches preserve a verified sequence for the next update",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, fontweight="bold")

    ax.text(0.02, 0.88, "A complete candidate passes", transform=ax.transAxes,
            fontsize=5.7, fontweight="bold", color=COLORS["controller"])
    segment(ax, 0.03, 0.67, 0.13, 0.10, COLORS["light_blue"], COLORS["candidate"], r"block $V_k$")
    segment(ax, 0.16, 0.67, 0.20, 0.10, COLORS["light_green"], COLORS["controller"],
            r"candidate $T_k^{(j)}$")
    ax.text(0.195, 0.61, r"$S_k^{(j)}=V_k\Vert T_k^{(j)}$", transform=ax.transAxes,
            ha="center", fontsize=5.3)
    arrow(ax, (0.37, 0.72), (0.45, 0.72), COLORS["certificate"])
    segment(ax, 0.45, 0.65, 0.13, 0.14, COLORS["light_gold"], COLORS["certificate"],
            "multilevel\nnumerical check", fontsize=5.0)
    arrow(ax, (0.58, 0.72), (0.66, 0.72), COLORS["controller"])
    segment(ax, 0.66, 0.67, 0.12, 0.10, COLORS["light_blue"], COLORS["candidate"],
            r"execute $V_k$")
    segment(ax, 0.80, 0.67, 0.17, 0.10, COLORS["light_green"], COLORS["controller"],
            "$R_{k+1}$" + "\n" + r"$=T_k^{(j^\star)}$", fontsize=5.0)

    ax.text(0.02, 0.42, "No complete candidate passes", transform=ax.transAxes,
            fontsize=5.7, fontweight="bold", color=COLORS["unsafe"])
    segment(ax, 0.03, 0.21, 0.13, 0.10, COLORS["light_green"], COLORS["controller"],
            r"$R_k[0{:}b_k]$", fontsize=5.0)
    segment(ax, 0.16, 0.21, 0.20, 0.10, COLORS["light_green"], COLORS["controller"],
            r"$R_k[b_k{:}h_k]$", fontsize=5.0)
    ax.text(0.195, 0.15, r"$R_k\in\mathcal{C}_k$", transform=ax.transAxes,
            ha="center", fontsize=5.3)
    arrow(ax, (0.58, 0.72), (0.58, 0.35), COLORS["unsafe"],
          connectionstyle="arc3,rad=0.18")
    ax.text(0.60, 0.47, "reject", transform=ax.transAxes, fontsize=5.2,
            color=COLORS["unsafe"])
    arrow(ax, (0.37, 0.26), (0.66, 0.26), COLORS["controller"])
    segment(ax, 0.66, 0.21, 0.12, 0.10, COLORS["light_green"], COLORS["controller"],
            r"execute $A_k$", fontsize=5.0)
    segment(ax, 0.80, 0.21, 0.17, 0.10, COLORS["light_green"], COLORS["controller"],
            "$R_{k+1}$" + "\n" + r"$=R_k[b_k{:}h_k]$", fontsize=4.7)

    ax.plot([0.79, 0.79], [0.13, 0.84], transform=ax.transAxes,
            color=COLORS["mid_gray"], linewidth=0.6, linestyle=(0, (2, 2)))
    ax.text(0.785, 0.88, "next update", transform=ax.transAxes,
            ha="center", fontsize=5.2, color=COLORS["unavailable"])
    ax.text(0.885, 0.49, "verified\ncontinuation\nremains", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.6, color=COLORS["controller"],
            fontweight="bold")


def panel_e_decision_units(ax) -> None:
    rows = ablation_rows()
    ax.set_xlim(-5.8, 31.0)
    ax.set_ylim(-0.45, 3.25)
    ax.axis("off")
    panel_label(ax, "e", x=-0.07, y=1.02)
    ax.text(0.0, 1.01, "Complete-sequence check retains future control",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, fontweight="bold")
    ax.text(0.0, 0.93, "0/16 unsafe in all three checks", transform=ax.transAxes,
            ha="left", fontsize=5.3, color=COLORS["unavailable"])
    ax.text(27.0, 2.70, "verified\ncontinuation", ha="center", va="bottom",
            fontsize=4.9, linespacing=0.95)

    y_centres = (2.25, 1.25, 0.25)
    for row, y0 in zip(rows, y_centres):
        ax.text(-0.45, y0 + 0.18, row["label"], ha="right", va="center",
                fontsize=5.5, fontweight="bold")
        for index in range(96):
            column = index % 24
            subrow = index // 24
            x = column
            y = y0 + 0.37 - 0.18 * subrow
            released = index < row["candidate_releases"]
            face = COLORS["candidate"] if released else COLORS["controller"]
            ax.scatter(x, y, marker="s", s=5.7, color=face, edgecolors="none")
        ax.text(24.0, y0 + 0.10,
                f"{row['candidate_releases']}/96\n{100 * row['execution_rate']:.1f}%",
                ha="left", va="center", fontsize=5.2)
        retained = bool(row["continuation_available_next_update"])
        if retained:
            ax.scatter(27.0, y0 + 0.10, s=30, marker="o", color=COLORS["controller"],
                       edgecolor=COLORS["controller"], linewidth=1.0)
        else:
            ax.scatter(27.0, y0 + 0.10, s=30, marker="x", color=COLORS["unsafe"],
                       linewidth=1.0)
        ax.text(27.0, y0 - 0.23, "yes" if retained else "no", ha="center",
                fontsize=5.0, color=COLORS["controller"] if retained else COLORS["unsafe"])

    handles = [
        Line2D([], [], marker="s", linestyle="none", color=COLORS["candidate"],
               markersize=4, label="Optimized block"),
        Line2D([], [], marker="s", linestyle="none", color=COLORS["controller"],
               markersize=4, label="Continuation block"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.01),
              ncols=2, fontsize=5.0, handletextpad=0.25, columnspacing=0.7)


def main() -> None:
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, min(5.45, MAX_HEIGHT_IN)))
    grid = fig.add_gridspec(
        3, 12, height_ratios=[1.0, 0.88, 1.0], hspace=0.43, wspace=0.62,
        left=0.075, right=0.992, bottom=0.050, top=0.975,
    )
    panel_a_failure_trajectories(fig.add_subplot(grid[0, 0:5]))
    panel_b_continuation_margins(fig.add_subplot(grid[0, 5:12]))
    panel_c_action_stress(fig.add_subplot(grid[1, 0:12]))
    panel_d_recursive_certificate(fig.add_subplot(grid[2, 0:8]))
    panel_e_decision_units(fig.add_subplot(grid[2, 8:12]))
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".svg"))
    fig.savefig(OUT.with_suffix(".png"), dpi=600)
    plt.close(fig)
    raster = iio.imread(OUT.with_suffix(".png"))
    tifffile.imwrite(
        OUT.with_suffix(".tiff"),
        raster,
        photometric="rgb",
        resolution=(600, 600),
        resolutionunit="INCH",
    )
    print(OUT)
    print("failure totals:", [row["overall_failure"] for row in failure_rows()])
    print("continuation-margin states:", len(continuation_margin_rows()))
    stress = action_stress_rows()
    for label in ("K5", "K15", "HF-CEM", "1.25x authority", "2x authority", "4x authority"):
        print(label, sum(row["short_label"] == label and row["hf_safe"] for row in stress), "/ 10")


if __name__ == "__main__":
    main()
