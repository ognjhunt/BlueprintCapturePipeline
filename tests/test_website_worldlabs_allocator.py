"""The website World Labs allocator admits the two website entry sources only.

Both website lanes -- the browser recorder (``browser_self_capture``) and the
Blueprint app or App Clip through the same capture link (``site_self_capture``)
-- pass the entry check and go on to the prepared-view and admission checks.
Any other source is refused before anything else is read, and nothing is ever
submitted to the provider from a refused descriptor.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import website_worldlabs_allocator
from blueprint_pipeline.website_capture_entry import BROWSER_SELF_CAPTURE, SITE_SELF_CAPTURE


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        descriptor=str(tmp_path / "descriptor.json"),
        capture_root=str(tmp_path / "capture"),
        experimental_branch_diagnostic=False,
        execute=False,
    )


def _never(*_args, **_kwargs):
    raise AssertionError("reached a step the entry check should have decided")


def _run(tmp_path: Path, source: object) -> dict:
    return website_worldlabs_allocator.run_website_worldlabs(
        _args(tmp_path),
        load_json=lambda _path: {"metadata": {"capture_entry_source": source}},
        source_checkout_blockers=_never,
        admission_issuer=_never,
    )


@pytest.mark.parametrize("source", [BROWSER_SELF_CAPTURE, SITE_SELF_CAPTURE])
def test_website_entry_sources_pass_the_entry_check(source: str, tmp_path: Path, monkeypatch) -> None:
    checked = []

    def prepared_views(*, descriptor, capture_root):
        checked.append(descriptor["metadata"]["capture_entry_source"])
        raise ValueError("prepared_views_checked")

    monkeypatch.setattr(website_worldlabs_allocator, "validate_website_prepared_views", prepared_views)

    result = _run(tmp_path, source)

    # The entry check passed, so the prepared-view check ran (and, here, blocked).
    assert checked == [source]
    assert result["blockers"] == ["prepared_views_checked"]
    assert result["status"] == "blocked"
    assert result["provider_mutation_attempted"] is False


@pytest.mark.parametrize("source", [None, "", "iphone", "browser_self_capture ", "SITE_SELF_CAPTURE", 7])
def test_any_other_entry_source_is_refused_first(source: object, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(website_worldlabs_allocator, "validate_website_prepared_views", _never)

    result = _run(tmp_path, source)

    assert result["blockers"] == ["website_reconstruction_entry_source_invalid"]
    assert result["status"] == "blocked"
    assert result["provider_mutation_attempted"] is False
    written = json.loads((tmp_path / "out" / "website_worldlabs_preflight.json").read_text(encoding="utf-8"))
    assert written == result
