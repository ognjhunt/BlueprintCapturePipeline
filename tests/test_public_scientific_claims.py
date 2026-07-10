from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINTER = ROOT / "scripts" / "lint_public_scientific_claims.py"


def _run(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LINTER), *(str(path) for path in paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_committed_public_claim_surfaces_are_bounded() -> None:
    completed = _run()

    assert completed.returncode == 0, completed.stderr
    assert "[public-scientific-claims] ok" in completed.stdout


def test_linter_rejects_blueprint_0929_and_percent_accuracy(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "Blueprint's 0.929 rank fidelity means the product has 93% accuracy.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 1
    assert "unsupported_percent_accuracy" in completed.stderr
    assert "blueprint_0929_attribution" in completed.stderr
    assert "blueprint_disclaimer_missing" in completed.stderr


def test_linter_rejects_percentage_form_of_blueprint_metric(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "Blueprint reports 92.9% rank fidelity for customer evaluations.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 1
    assert "unsupported_percent_accuracy" in completed.stderr
    assert "blueprint_0929_attribution" in completed.stderr
    assert "blueprint_disclaimer_missing" in completed.stderr


def test_linter_accepts_fully_attributed_sc3_result(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "SC3-Eval reports overall closed-loop Pearson correlation 0.929 across "
        "seven policies. This is not a Blueprint measurement; Blueprint has not "
        "measured equivalent rank fidelity.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 0, completed.stderr


def test_linter_rejects_oscar_metrics_transferred_to_sc3(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "SC3-Eval reports Pearson 0.852 and success_rate_difference_pp 1.73.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 1
    assert "oscar_metric_transferred_to_sc3" in completed.stderr


def test_linter_rejects_sc3_metrics_transferred_to_oscar(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "OSCAR reports Pearson 0.929 and MMRV 0.119 across seven policies.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 1
    assert "sc3_metric_transferred_to_oscar" in completed.stderr


def test_linter_rejects_reverse_order_metric_transfer(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "Pearson 0.852 is SC3-Eval's reported result. MMRV 0.119 belongs to OSCAR.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 1
    assert "oscar_metric_transferred_to_sc3" in completed.stderr
    assert "sc3_metric_transferred_to_oscar" in completed.stderr


def test_linter_accepts_family_scoped_side_by_side_metrics(tmp_path: Path) -> None:
    claim = tmp_path / "claim.md"
    claim.write_text(
        "OSCAR reports Pearson 0.852; SC3-Eval reports overall closed-loop Pearson "
        "0.929 across seven policies. This is not a Blueprint measurement.\n",
        encoding="utf-8",
    )

    completed = _run(claim)

    assert completed.returncode == 0, completed.stderr


def test_linter_rejects_legacy_sisr_delta_variants(tmp_path: Path) -> None:
    for variant in ("SISR delta", "SISR-delta", "sisr_delta"):
        claim = tmp_path / f"claim-{variant.replace(' ', '-')}.md"
        claim.write_text(f"OSCAR reports {variant} 1.73.\n", encoding="utf-8")

        completed = _run(claim)

        assert completed.returncode == 1
        assert "legacy_sisr_delta_metric" in completed.stderr
