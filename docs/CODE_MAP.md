# Code map

This map links the manuscript analyses to the public entry points. Helper
modules are listed only where they define a scientific calculation rather than
plot styling.

| Manuscript component | Primary entry point | Principal archived input |
|---|---|---|
| Discovery surrogate training | `scripts/train_discovery_surrogates.py` | `datasets/discovery/` |
| Thermal surrogate training | `scripts/train_thermal_surrogate.py` | `datasets/thermal/` |
| Surrogate-guided decisions | `scripts/generate_*_decisions.py` | checkpoints and controlled trajectories |
| Decision-regime integrity | `scripts/audit_data_integrity.py` | `decision_archives/discovery/` |
| Grouped regime statistics | `scripts/analyze_grouped_statistics.py` | discovery and evaluation archives |
| Optimization-path evidence | `scripts/figures/reproduce_extended_data_figures.py` | Extended Data Fig. 3 Source Data |
| Critical-event analysis | `scripts/extract_critical_event_features.py`, `scripts/analyze_critical_events.py` | `analysis/critical_events/` |
| Operator-defect features | `scripts/operator_defect_features.py` | decision archives and PDE operators |
| Operator-defect feature audit | `scripts/analyze_operator_defect_features.py` | risk-geometry and mismatch-detection records |
| Residual-feature mismatch detection | `scripts/mismatch_detection.py`, `scripts/evaluate_continuation_availability.py` | detector development, calibration and evaluation records |
| Continuation availability | `scripts/continuation_library.py`, `scripts/evaluate_continuation_availability.py` | `analysis/recoverability/` |
| Recoverability surface | `scripts/evaluate_recoverability_surface.py`, `scripts/analyze_recoverability_surface.py` | recoverability state sets |
| HF reference search | `scripts/reference_recoverability_search.py` | ten challenging states |
| Gray--Scott closed loop | `scripts/gray_scott_continuation_controller.py`, `scripts/audit_gray_scott_closed_loop.py` | `closed_loop/gray_scott/` |
| Sequence-definition ablation | `scripts/compare_sequence_definitions.py`, `scripts/audit_sequence_definitions.py` | sequence-evaluation records |
| Predictive safety-filter comparison | `scripts/compare_predictive_filter.py` | full-plan and continuation-augmented episodes |
| Thermal numerical calibration | `scripts/calibrate_thermal_numerical_levels.py` | `numerical_qualification/thermal/allowance_queries.jsonl` |
| Thermal numerical certificate | `scripts/thermal_numerical_certificate.py` | `numerical_qualification/thermal/one_sided_allowance.json` |
| Thermal closed loop | `scripts/run_thermal_closed_loop.py`, `scripts/audit_thermal_closed_loop.py` | `closed_loop/thermal/` |
| Static PDE design | `scripts/evaluate_static_design.py`, `scripts/analyze_static_design.py` | `analysis/static_pde_design/` |
| Main Figs. 2-6 | `scripts/figures/figure_*.py` | archived analyses and Source Data |
| Extended Data Figs. 1-6 | `scripts/figures/reproduce_extended_data_figures.py` | one Source Data workbook per figure |

`scripts/surrogate_pair_evaluator.py` and the PDE-specific dynamics modules
provide the shared rollout, risk and state-transfer operations used by several
entry points.

Release utilities are `tools/link_data.py`, `tools/validate_release.py`,
`tools/build_file_manifest.py` and `tools/create_release_archives.py`.
