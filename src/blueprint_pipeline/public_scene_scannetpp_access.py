"""Derive a fail-closed ScanNet++ access outcome from publisher source bytes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


SCHEMA_VERSION = "adp009_scannetpp_access_outcome.v1"
DATASET_URL = "https://scannetpp.mlsg.cit.tum.de/scannetpp/"
TERMS_URL = (
    "https://scannetpp.mlsg.cit.tum.de/scannetpp/static/"
    "scannetpp-terms-of-use.pdf"
)


class ScanNetPPAccessOutcomeError(ValueError):
    """A deterministic ScanNet++ access-evidence validation failure."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: str | Path, roots: Sequence[Path], code: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_symlink() or not candidate.is_file():
        raise ScanNetPPAccessOutcomeError([code])
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise ScanNetPPAccessOutcomeError([code])
    return candidate


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _html_text(path: Path) -> str:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8"))
    return _normalized(" ".join(parser.parts))


def _pdf_text(path: Path, *, pdftotext: str) -> tuple[str, str]:
    binary = shutil.which(pdftotext)
    if binary is None:
        raise ScanNetPPAccessOutcomeError(["scannetpp_pdftotext_unavailable"])
    process = subprocess.run(
        [binary, "-layout", str(path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0 or not process.stdout.strip():
        raise ScanNetPPAccessOutcomeError(["scannetpp_terms_pdf_parse_failed"])
    return _normalized(process.stdout), binary


def materialize_scannetpp_access_outcome(
    *,
    dataset_page: str | Path,
    terms_pdf: str | Path,
    output_path: str | Path,
    approved_roots: Sequence[str | Path],
    pdftotext: str = "pdftotext",
) -> dict[str, Any]:
    """Bind current publisher access/terms bytes without accepting their agreement."""

    roots = tuple(Path(root).expanduser().resolve() for root in approved_roots)
    if not roots:
        raise ScanNetPPAccessOutcomeError(["scannetpp_approved_roots_missing"])
    page = _under(dataset_page, roots, "scannetpp_dataset_page_outside_approved_roots")
    terms = _under(terms_pdf, roots, "scannetpp_terms_pdf_outside_approved_roots")
    output = Path(output_path).expanduser().resolve()
    if not any(output == root or root in output.parents for root in roots):
        raise ScanNetPPAccessOutcomeError(["scannetpp_receipt_output_outside_approved_roots"])

    page_text = _html_text(page)
    terms_text, pdftotext_binary = _pdf_text(terms, pdftotext=pdftotext)
    page_requirements = {
        "account_required": "create an account" in page_text,
        "application_required": "create an application" in page_text,
        "publisher_approval_required": "once your application is approved" in page_text,
        "personalized_token_required": "personalized token" in page_text,
        "terms_acceptance_required": "agree to after signing up" in page_text,
    }
    terms_requirements = {
        "noncommercial_research_or_education_only": (
            "only for non-commercial research and educational purposes" in terms_text
        ),
        "commercial_use_prohibited": "commercial use is strictly prohibited" in terms_text,
        "downstream_recipients_must_sign": (
            "receiving entity has also agreed to and signed these terms" in terms_text
        ),
        "for_profit_employer_bound": (
            "employed by a for-profit, commercial entity" in terms_text
            and "employer shall also be bound" in terms_text
        ),
        "requester_must_have_employer_authority": (
            "fully authorized to enter into this agreement on behalf of such employer"
            in terms_text
        ),
    }
    missing = [
        f"publisher_page_requirement_missing:{key}"
        for key, observed in page_requirements.items()
        if not observed
    ] + [
        f"publisher_terms_requirement_missing:{key}"
        for key, observed in terms_requirements.items()
        if not observed
    ]
    if missing:
        raise ScanNetPPAccessOutcomeError(missing)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009C",
        "status": "blocked",
        "blockers": [
            "scannetpp_account_application_approval_and_terms_authority_required"
        ],
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "publisher_sources": {
            "dataset_page": {
                "url": DATASET_URL,
                "path": str(page),
                "size_bytes": page.stat().st_size,
                "sha256": _sha256(page),
            },
            "terms_pdf": {
                "url": TERMS_URL,
                "path": str(terms),
                "size_bytes": terms.stat().st_size,
                "sha256": _sha256(terms),
                "text_extractor": pdftotext_binary,
            },
        },
        "observed_access_requirements": page_requirements,
        "observed_terms_requirements": terms_requirements,
        "execution": {
            "account_created_by_blueprint": False,
            "application_submitted_by_blueprint": False,
            "terms_accepted_by_blueprint": False,
            "dataset_bytes_downloaded": False,
            "scene_selected": False,
        },
        "smallest_next_authority": (
            "authorized requester must create or use a ScanNet++ account, submit an "
            "application, obtain approval, and accept the current terms with authority "
            "to bind any applicable for-profit employer"
        ),
        "claim_boundary": {
            "publisher_page_and_terms_bytes_inspected": True,
            "access_granted": False,
            "rights_admitted": False,
            "scene_admitted": False,
            "blocked_outcome_is_not_component_admission": True,
        },
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-page", required=True)
    parser.add_argument("--terms-pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--pdftotext", default="pdftotext")
    args = parser.parse_args(argv)
    materialize_scannetpp_access_outcome(
        dataset_page=args.dataset_page,
        terms_pdf=args.terms_pdf,
        output_path=args.output,
        approved_roots=args.approved_root,
        pdftotext=args.pdftotext,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "ScanNetPPAccessOutcomeError",
    "materialize_scannetpp_access_outcome",
]
