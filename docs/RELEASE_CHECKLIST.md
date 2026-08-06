# Public release checklist

## Required before creating the GitHub release

- [ ] Select and add the approved code license.
- [ ] Select the companion data license.
- [ ] Add the final data-record URL to the manuscript and `CITATION.cff`.
- [ ] Relabel the README worksheet inside each Extended Data workbook using the
      checklist in the Source Data directory.
- [x] Run `python tools/validate_release.py --verify-hashes` (6 August 2026).
- [x] Run `python -m pytest -q` (6 August 2026; 5 tests passed).
- [x] Run `python scripts/reproduce_all_figures.py` in the recorded environment
      (6 August 2026; 11 quantitative figures reproduced).
- [x] Inspect the regenerated PDFs against the submitted figures (6 August
      2026).
- [x] Replay the 48 independent Gray--Scott episodes from stored controls
      (6 August 2026; all trajectory checks and registered gates passed).
- [x] Create and test the local `v1.0.0` code and data upload archives and
      generate `SHA256SUMS.txt` (6 August 2026).
- [x] Confirm that no checkpoint, dataset, Source Data workbook or generated
      output is staged in the Git repository (6 August 2026).
- [ ] Create a versioned GitHub release, for example `v1.0.0`.
- [ ] Archive the code release and data package in DOI-minting repositories.
- [ ] Add the final code and data DOIs to `CITATION.cff` and the manuscript.

## Source Data workbook check

The six Extended Data workbooks were inherited from an earlier supplementary
layout. Their numerical worksheets are current, but the README labels must be
changed from `Fig. S1-S6` to `Extended Data Fig. 1-6` before upload. Do not edit
the numerical sheets during this relabelling step.
