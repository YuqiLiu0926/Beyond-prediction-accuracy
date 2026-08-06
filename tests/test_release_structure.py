from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_entry_points_exist():
    required = [
        "scripts/reproduce_all_figures.py",
        "scripts/train_discovery_surrogates.py",
        "scripts/generate_burgers_decisions.py",
        "scripts/generate_gray_scott_decisions.py",
        "scripts/generate_kolmogorov_decisions.py",
        "scripts/audit_data_integrity.py",
        "scripts/analyze_critical_events.py",
        "scripts/analyze_operator_defect_features.py",
        "scripts/analyze_recoverability_surface.py",
        "scripts/audit_gray_scott_closed_loop.py",
        "scripts/audit_thermal_closed_loop.py",
        "scripts/thermal_numerical_certificate.py",
        "scripts/calibrate_thermal_numerical_levels.py",
    ]
    assert all((ROOT / relative).is_file() for relative in required)


def test_release_documentation_exists():
    required = [
        "README.md",
        "environment.yml",
        "CITATION.cff",
        "docs/CODE_MAP.md",
        "docs/DATA_DICTIONARY.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RELEASE_CHECKLIST.md",
    ]
    assert all((ROOT / relative).is_file() for relative in required)


def test_data_is_not_committed_inside_code_tree():
    assert not (ROOT / "checkpoints").exists()
    assert not any(ROOT.glob("*.pt"))
    assert not any(ROOT.glob("*.npz"))
