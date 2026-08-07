from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.public_scene_scannetpp_access import (
    ScanNetPPAccessOutcomeError,
    materialize_scannetpp_access_outcome,
)


PAGE = """
<html><body>
Please create an account, login and create an application. Once your application
is approved, you will receive a personalized token. The data is released under
terms which you can agree to after signing up.
</body></html>
"""

TERMS = """
Researcher shall use the Database only for non-commercial research and educational
purposes. Commercial use is strictly prohibited. The receiving entity has also
agreed to and signed these terms. If Researcher is employed by a for-profit,
commercial entity, Researcher's employer shall also be bound, and Researcher is
fully authorized to enter into this agreement on behalf of such employer.
"""


def _pdftotext(path: Path) -> Path:
    script = path / "pdftotext"
    script.write_text(
        "#!/usr/bin/env python3\n"
        f"print({TERMS!r}, end='')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _sources(root: Path) -> tuple[Path, Path]:
    page = root / "page.html"
    terms = root / "terms.pdf"
    page.write_text(PAGE, encoding="utf-8")
    terms.write_bytes(b"%PDF-test")
    return page, terms


def test_materializes_blocked_access_outcome_from_source_bytes(tmp_path: Path) -> None:
    page, terms = _sources(tmp_path)
    output = tmp_path / "receipt.json"
    receipt = materialize_scannetpp_access_outcome(
        dataset_page=page,
        terms_pdf=terms,
        output_path=output,
        approved_roots=[tmp_path],
        pdftotext=str(_pdftotext(tmp_path)),
    )

    assert receipt["status"] == "blocked"
    assert receipt["execution"]["terms_accepted_by_blueprint"] is False
    assert receipt["claim_boundary"]["blocked_outcome_is_not_component_admission"]
    assert receipt["publisher_sources"]["dataset_page"]["size_bytes"] == len(
        PAGE.encode()
    )
    assert json.loads(output.read_text())["receipt_digest"] == receipt["receipt_digest"]


def test_rejects_source_outside_approved_roots(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    page, terms = _sources(tmp_path)

    with pytest.raises(ScanNetPPAccessOutcomeError) as exc:
        materialize_scannetpp_access_outcome(
            dataset_page=page,
            terms_pdf=terms,
            output_path=approved / "receipt.json",
            approved_roots=[approved],
            pdftotext=str(_pdftotext(tmp_path)),
        )

    assert "scannetpp_dataset_page_outside_approved_roots" in exc.value.codes


def test_rejects_missing_publisher_requirement(tmp_path: Path) -> None:
    page, terms = _sources(tmp_path)
    page.write_text("<html>public dataset</html>", encoding="utf-8")

    with pytest.raises(ScanNetPPAccessOutcomeError) as exc:
        materialize_scannetpp_access_outcome(
            dataset_page=page,
            terms_pdf=terms,
            output_path=tmp_path / "receipt.json",
            approved_roots=[tmp_path],
            pdftotext=str(_pdftotext(tmp_path)),
        )

    assert any(
        code.startswith("publisher_page_requirement_missing:")
        for code in exc.value.codes
    )
