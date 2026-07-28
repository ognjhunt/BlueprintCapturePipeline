"""Regression: the delivery-time consent-takedown gate must be WIRED into every
production webapp-sync call site.

The gate lives inside ``sync_webapp_pipeline_attachment`` but only runs when the
caller passes ``capture_root=`` (it is ``None`` by default and the gate is
guarded by ``if capture_root is not None``). A caller that omits it silently
bypasses the entire consent-revocation subsystem — green gate tests, dead gate
in prod. These tests fail if that bypass ever exists.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import blueprint_pipeline.alpha_readiness as alpha
import blueprint_pipeline.consent_takedown as ct
import blueprint_pipeline.site_package_orchestrator as qualification


def _min_capture(tmp_path: Path, *, revoked: bool = False) -> Path:
    root = tmp_path / "scenes" / "s1" / "captures" / "c1"
    (root / "raw").mkdir(parents=True)
    (root / "pipeline").mkdir(parents=True)
    (root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "s1",
                "capture_id": "c1",
                "raw_prefix_uri": "gs://bucket/scenes/s1/captures/c1/raw",
                "frames_index_uri": "gs://bucket/scenes/s1/captures/c1/raw/frames.json",
                "site_submission_id": "sub1",
                "buyer_request_id": "buy1",
                "capture_job_id": "job1",
            }
        ),
        encoding="utf-8",
    )
    (root / "raw" / "rights_consent.json").write_text(
        json.dumps(
            {
                "consent_status": "revoked" if revoked else "documented",
                **(
                    {"consent_revoked": True, "consent_revoked_at": "2026-07-04T00:00:00Z"}
                    if revoked
                    else {}
                ),
            }
        ),
        encoding="utf-8",
    )
    return root


def _sync_call_capture_root_flags(module) -> list[bool]:
    """For each `sync_webapp_pipeline_attachment(...)` call in a module's source,
    whether it passes a `capture_root=` keyword."""
    tree = ast.parse(inspect.getsource(module))
    flags: list[bool] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "sync_webapp_pipeline_attachment":
                flags.append(any(kw.arg == "capture_root" for kw in node.keywords))
    return flags


def test_every_production_sync_call_wires_the_consent_gate():
    for module in (qualification, alpha):
        flags = _sync_call_capture_root_flags(module)
        assert flags, f"no sync_webapp_pipeline_attachment call found in {module.__name__}"
        assert all(flags), (
            f"a sync_webapp_pipeline_attachment call in {module.__name__} omits "
            "capture_root — the consent-takedown gate is bypassed"
        )


def test_alpha_readiness_sync_invokes_consent_gate_with_capture_root(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)
    seen: list[Path] = []

    def spy(*, capture_root, **_kwargs):
        seen.append(Path(capture_root))
        return {"schema_version": "x", "status": "allowed", "serve_allowed": True, "blockers": []}

    monkeypatch.setattr(ct, "evaluate_delivery_time_takedown_gate", spy)
    root = _min_capture(tmp_path)

    alpha.sync_webapp_evaluation_prep(capture_root=root)

    assert seen, "consent gate was never evaluated on the alpha_readiness sync path (capture_root not wired)"
    assert seen[0] == root
