from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thermal_numerical_certificate import (  # noqa: E402
    ThermalCertificateCalibration,
    certify_control_family,
)


def test_frozen_thermal_calibration_matches_manuscript_values():
    calibration = ThermalCertificateCalibration.from_json(
        ROOT
        / "external_data"
        / "numerical_qualification"
        / "thermal"
        / "one_sided_allowance.json"
    )
    assert calibration.n0_grid_size == 32
    assert calibration.n0_substeps == 10
    assert calibration.n1_grid_size == 64
    assert calibration.n1_substeps == 10
    assert calibration.target_substeps == 20
    assert np.isclose(calibration.n0_allowance, 0.04231428630284917)
    assert calibration.n1_allowance == 0.0
    assert np.isclose(calibration.richardson_gamma, 1.0 / 3.0)


def test_certificate_escalates_and_applies_richardson_defect():
    calibration = ThermalCertificateCalibration(
        n0_allowance=0.04,
        n1_allowance=0.01,
        richardson_gamma=1.0 / 3.0,
    )
    grids = {"N0": {"name": "N0"}, "N1": {"name": "N1"}}
    controls = [np.asarray([0.0]), np.asarray([1.0])]

    def rollout(state, control, grid, store_states):
        del state, store_states
        risks = {
            ("N0", 0): 1.98,
            ("N0", 1): 2.02,
            ("N1", 0): 1.95,
            ("N1", 1): 1.99,
        }
        return None, risks[(grid["name"], int(control[0]))]

    selected, certificate = certify_control_family(
        np.zeros((2, 64, 64)), controls, grids, calibration, 2.0, rollout
    )
    expected_upper = 1.95 + (1.0 / 3.0) * abs(1.95 - 1.98) + 0.01
    assert selected[0] == 0.0
    assert certificate["level"] == "N1"
    assert certificate["N1_evaluated"] == 1
    assert np.isclose(certificate["z_upper"], expected_upper)
    assert certificate["certified"] == 1
