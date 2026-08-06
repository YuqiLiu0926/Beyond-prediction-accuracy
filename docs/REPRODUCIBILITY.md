# Reproducibility workflow

The release supports three levels of reproduction. Begin by linking the data
package and running `python tools/validate_release.py`.

## Level 1: regenerate all submitted plots

```bash
python scripts/reproduce_all_figures.py
```

This reads the archived decision arrays and submitted Source Data workbooks,
checks manuscript totals, and exports Figs. 2-6 and Extended Data Figs. 1-6.
Fig. 1 is a conceptual overview and is not generated from numerical data.

Individual entry points are:

```bash
python scripts/figures/figure_2_margin_plane.py
python scripts/figures/figure_3_tail_risk.py
python scripts/figures/figure_4_recoverability.py
python scripts/figures/figure_5_gray_scott_closed_loop.py
python scripts/figures/figure_6_thermal_transfer.py
python scripts/figures/reproduce_extended_data_figures.py --suffix _reproduced
```

## Level 2: recompute reported analyses from archived records

```bash
python scripts/audit_data_integrity.py
python scripts/analyze_grouped_statistics.py
python scripts/analyze_critical_events.py
python scripts/analyze_operator_defect_features.py
python scripts/analyze_recoverability_surface.py
python scripts/analyze_static_design.py
python scripts/audit_gray_scott_closed_loop.py
python scripts/audit_thermal_closed_loop.py
python scripts/compare_predictive_filter.py
```

These commands recompute regime totals, grouped uncertainty, critical-event
metrics, recoverability summaries, static-design outcomes and closed-loop
statistics. They do not retrain the neural surrogates.

The operator-defect audit should report 1,791 surrogate-accepted decisions,
including 116 false-safe decisions. Adding the nineteen operator-defect
summaries to the four risk summaries changes accepted-only AUROC from 0.7140
to 0.8116 and average precision from 0.1633 to 0.3195.

The thermal allowances can also be recomputed directly:

```bash
python scripts/calibrate_thermal_numerical_levels.py \
  --output outputs/numerical_qualification/thermal/one_sided_allowance.json
```

This calculation groups 1,565 sequence queries into sixteen commissioning
episodes. It recovers the frozen `N0` allowance `0.04231428630284917`, the
`N1` allowance `0`, and the Richardson coefficient `1/3`.

## Level 3: regenerate trajectories and decisions

Training and decision generation require substantially more computation. Use
the `--help` interface of the training and PDE-specific decision scripts. Keep
the provided state splits, seeds, actuator bounds and safety limits unchanged
when attempting an exact replication.

The Gray--Scott and thermal closed-loop protocols are stored in
`external_data/closed_loop/<system>/protocol.json`. Numerical-level calibration
records are stored under `external_data/numerical_qualification/`.

## Expected principal checks

- Discovery decisions: 2,400 total, comprising 1,675 safe agreement, 116 false
  safety, 117 conservative rejection and 492 unsafe agreement.
- Gray--Scott independent closed loops: 48 episodes, 36 unsafe raw episodes,
  zero observed unsafe episodes under continuation preservation, 249 released
  optimizer blocks out of 288 decisions and a 9.47-fold mean task value relative
  to continuation-only control.
- Thermal independent closed loops: 128 episodes and zero observed unsafe
  episodes under continuation preservation.
- Frozen checkpoints: twelve discovery models and one thermal FNO.

All stochastic analyses group the four surrogate evaluations that share a
physical state. A result is not considered reproduced if this grouping is
discarded.
