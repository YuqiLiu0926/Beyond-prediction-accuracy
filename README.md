# Decision reliability and physical recoverability of neural PDE surrogates

This repository contains the code used in the manuscript **Beyond prediction
accuracy: decision reliability and physical recoverability of neural PDE
surrogates**. It evaluates learned PDE models after optimization, analyzes
decision-defining extrema and operator defects, measures finite-horizon
recoverability, and tests continuation-preserving control in high-fidelity
closed loops.

The numerical data and trained weights are distributed separately because they
exceed normal GitHub file limits. The code expects that package at
`external_data/`.

## Repository contents

- `scripts/`: training, decision generation, statistical analyses, closed-loop
  evaluation and manuscript figure generation.
- `scripts/figures/`: entry points for Figs. 2-6 and Extended Data Figs. 1-6.
- `tools/`: data linking, release validation, archive completion and file
  manifest generation.
- `docs/`: data dictionary, code map and staged reproduction instructions.
- `tests/`: lightweight release-structure tests.

The companion data package contains the four trajectory datasets, thirteen
frozen checkpoints, decision archives, grouped detector records, recoverability
audits, closed-loop episodes and per-figure Source Data workbooks.

## Installation

The recorded environment used Python 3.10 and PyTorch 2.5.1 with CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate neural-pde-decision-reliability
```

CPU-only users may replace `pytorch-cuda` with the appropriate PyTorch CPU
package. Figure reproduction and most statistical audits do not require a GPU.

## Attach the data package

Place the code and data directories next to one another, then run:

```bash
python tools/link_data.py ../neural-pde-decision-reliability-data
python tools/validate_release.py
```

The linker creates `external_data` without copying the 2.4 GB package. Pass
`--copy` only on systems where directory links are unavailable.

## Reproduce manuscript figures

```bash
python scripts/reproduce_all_figures.py
```

Outputs are written to `outputs/figures/` and `outputs/extended_data/`. Each
figure script checks the manuscript totals before exporting PDF, SVG and PNG
files. See `docs/REPRODUCIBILITY.md` for individual commands and the distinction
between figure reproduction, statistical reanalysis and full model reruns.

## Main analysis entry points

```bash
python scripts/audit_data_integrity.py
python scripts/analyze_grouped_statistics.py
python scripts/analyze_critical_events.py
python scripts/analyze_operator_defect_features.py
python scripts/evaluate_continuation_availability.py
python scripts/analyze_recoverability_surface.py
python scripts/analyze_static_design.py
python scripts/audit_gray_scott_closed_loop.py
python scripts/audit_thermal_closed_loop.py
python scripts/compare_predictive_filter.py
```

The frozen thermal numerical allowances can be recomputed independently from
the 1,565 archived calibration queries:

```bash
python scripts/calibrate_thermal_numerical_levels.py
```

The final hierarchy evaluates a `32 x 32` realization first and escalates to a
`64 x 64` realization when the lower-cost evidence is insufficient. The
escalated estimate includes the Richardson defect relative to the `32 x 32`
result. A temporally refined `64 x 64` realization is used only as the offline
calibration target and as the closed-loop reference dynamics.

The operator-defect feature audit reads the frozen discovery decisions,
finite-difference risk geometry and nineteen PDE operator-defect summaries. It
reproduces the grouped ranking metrics and conformal operating points reported
in Extended Data Fig. 4.

The Gray--Scott closed-loop audit defaults to the three independent evaluation
sets. It replays all 48 episodes from the stored controls and verifies 36 unsafe
raw episodes, zero unsafe episodes under continuation preservation, 249 released
optimizer blocks out of 288 decisions and a 9.47-fold mean task value relative
to continuation-only control. Use `--scope development` to audit the separate
sixteen-episode development set.

The analysis scripts preserve physical-state groups whenever multiple surrogate
models share the same state. High-fidelity trajectories are used for offline
evaluation and closed-loop state advancement, not as candidate-specific future
information in the online decision rule.

## Training and decision generation

Model training and candidate generation are substantially more expensive than
the archived-data analyses. The principal entry points are:

```bash
python scripts/train_discovery_surrogates.py --help
python scripts/train_thermal_surrogate.py --help
python scripts/generate_burgers_decisions.py --help
python scripts/generate_gray_scott_decisions.py --help
python scripts/generate_kolmogorov_decisions.py --help
```

The published checkpoints and decision arrays are supplied so that readers can
reproduce the reported analyses without retraining twelve discovery models.

## Reproducibility and provenance

- `FILE_MANIFEST.csv` records SHA-256 hashes for the code release.
- The data package has its own `FILE_MANIFEST.csv` with hashes for every file.
- Each decision archive contains arrays, a flat decision table, a run summary
  and a manifest linking the dataset, checkpoint and generation script.
- Protocol files record fixed state assignments, numerical levels and one-sided
  allowances used in independent evaluation.

See `docs/CODE_MAP.md`, `docs/DATA_DICTIONARY.md`,
`docs/RELEASE_CHECKLIST.md` and `docs/GITHUB_AND_ARCHIVAL_RELEASE.md` for
details.

## Citation



## License


