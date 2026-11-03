"""A website job's grant: checked from what this host retains, or also re-read from the WebApp.

The launch catalog's projection runs in the ExecStartPre of every guarded unit and checks
only the retained sources. Every other caller, each paid path among them, also re-reads the
current context and grant and refuses when either changed.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_authority.py

from __future__ import annotations

import pytest

from blueprint_pipeline import website_preparation_authority as authority
from blueprint_pipeline import website_task_context

CONTEXT = {"request_id": "request-1", "scene_id": "scene-1", "capture_id": "capture-1",
           "context_digest": "sha256:" + "1" * 64}
GRANT = {"authority_digest": "sha256:" + "2" * 64}


def _retained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority, "_retained_sources", lambda **_: (dict(CONTEXT), dict(GRANT), None))


def _webapp_unreachable(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reads: list[str] = []

    def refused(name):
        def read(**_):
            reads.append(name)
            raise OSError("webapp unreachable")
        return read

    monkeypatch.setattr(website_task_context, "load_current_website_task_context", refused("task_context"))
    monkeypatch.setattr(website_task_context, "load_website_scene_sponsorship", refused("sponsorship"))
    return reads


def test_inside_retained_sources_only_the_webapp_is_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _retained(monkeypatch)
    reads = _webapp_unreachable(monkeypatch)
    with authority.retained_sources_only():
        assert authority.require_retained_preparation_authority(request={}, queue_root="/x", now=5.0) == 5.0
    assert reads == []
    # Outside the block the same check needs the WebApp, and refuses without it.
    with pytest.raises(OSError, match="webapp unreachable"):
        authority.require_retained_preparation_authority(request={}, queue_root="/x", now=5.0)
    assert reads == ["task_context"]


def test_the_retained_sources_still_refuse_inside_the_block(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", raising=False)
    reads = _webapp_unreachable(monkeypatch)
    with authority.retained_sources_only(), pytest.raises(ValueError, match="website_preparation_source_missing"):
        authority.require_retained_preparation_authority(
            request={"submission_id": "submission-1"}, queue_root=tmp_path / "intents", now=5.0)
    assert reads == []


def test_every_other_caller_refuses_a_context_the_site_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    _retained(monkeypatch)
    monkeypatch.setattr(website_task_context, "load_current_website_task_context",
                        lambda **_: {**CONTEXT, "context_digest": "sha256:" + "3" * 64})
    with pytest.raises(ValueError, match="website_preparation_source_context_changed"):
        authority.require_retained_preparation_authority(request={}, queue_root="/x", now=5.0)


def test_the_reads_come_back_when_the_block_ends_by_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _retained(monkeypatch)
    reads = _webapp_unreachable(monkeypatch)
    with pytest.raises(RuntimeError), authority.retained_sources_only():
        raise RuntimeError("the projection failed")
    with pytest.raises(OSError):
        authority.require_retained_preparation_authority(request={}, queue_root="/x", now=5.0)
    assert reads == ["task_context"]
