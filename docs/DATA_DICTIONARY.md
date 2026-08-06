# Data dictionary

## Directory-level organization

| Directory | Contents |
|---|---|
| `datasets/` | Controlled PDE trajectories used for training and evaluation |
| `checkpoints/` | Twelve discovery checkpoints and one thermal FNO checkpoint |
| `decision_archives/` | Optimized controls, surrogate and HF trajectories, flat tables and run manifests |
| `analysis/mismatch_detection/` | Grouped detector development, calibration and evaluation records |
| `analysis/risk_geometry/` | Surrogate-risk traces and finite-difference control-sensitivity features |
| `analysis/operator_defect_features/` | Frozen nineteen-feature archive used in the discovery feature audit |
| `analysis/mismatch_detection/` | Nineteen PDE operator-defect summaries and grouped detector records |
| `analysis/critical_events/` | Temporal-extremum descriptors and grouped model evaluations |
| `analysis/recoverability/` | Continuation outcomes, HF audits, stress tests and sequence ablations |
| `analysis/static_pde_design/` | Candidate designs, selected designs and HF oracle labels |
| `closed_loop/` | Gray--Scott and thermal episode records and frozen protocols |
| `numerical_qualification/` | Numerical-level queries and one-sided allowances |
| `source_data/` | Excel workbooks supplied with the manuscript and machine-readable CSV exports |

## Decision archives

Every directory below `decision_archives/<split>/<pde>/<model>/challenge_conditions/`
contains four files:

- `decision_data.npz`: multidimensional arrays;
- `decision_summary.csv`: one row per optimized decision;
- `summary.json`: run-level statistics and fixed configuration; and
- `run_manifest.json`: links to the checkpoint, dataset and generation script.

The three splits have distinct roles. `discovery` contains 200 conditions for
each of twelve surrogate--PDE pairs. `detector_development` and `evaluation`
contain 100 conditions per pair and use disjoint physical-state groups.

Important NPZ arrays are:

| Field | Meaning |
|---|---|
| `initial_states` | Initial PDE states where embedded in development/evaluation archives; discovery states are reconstructed from the dataset, index and fixed generation rule |
| `U_all` | Selected full-horizon control sequences |
| `alpha_all` | Blockwise control parameterization used by the optimizer |
| `z_sur`, `z_hf` | Surrogate and high-fidelity maximum risks |
| `margin_sur`, `margin_hf` | Safety limit minus the corresponding risk |
| `J_sur`, `J_hf` | Surrogate and high-fidelity task values |
| `z_sur_ts_all`, `z_hf_ts_all` | Time-resolved risk traces |
| `time_state`, `time_control` | Stored state and control times |
| `tags` | `high_gradient` or `amplified_out_of_distribution` condition type |

Positive margins indicate a safe classification. False safety denotes a
nonnegative surrogate margin and a negative HF margin.

## Numerical levels

Public records use `N0` and `N1` for the two prevalidated numerical levels.
`N0` is evaluated first. `N1` is evaluated when `N0` does not provide a
nonnegative certified margin. A level entry of `unavailable` means that neither
validated level accepted the sequence; it is not an HF unsafe label.

For the thermal system, `N0` uses a `32 x 32` spatial grid and ten internal
substeps per control interval. Its upper-risk estimate is
`z_N0 + 0.04231428630284917`. `N1` uses a `64 x 64` grid and ten substeps. Its
upper-risk estimate is
`z_N1 + (1/3) * abs(z_N1 - z_N0)`, because the calibrated additional
one-sided allowance at this level is zero. The offline calibration target and
closed-loop reference dynamics use a `64 x 64` grid with twenty substeps. The
calibration file reports the full precision used by the code.

Thermal certificate records use the fields `z_N0`, `z_N1`,
`resolution_defect`, `N0_allowance`, `N1_allowance`, `richardson_gamma` and
`z_upper`. `z_N1` and `resolution_defect` are `null` when a sequence is
accepted at `N0`, because the higher-cost level was not evaluated.

## Missing values

Blank CSV cells and JSON `null` values denote quantities that were not stored
or were not defined for that record. The release tools do not impute optimizer
diagnostics or physical outcomes.
