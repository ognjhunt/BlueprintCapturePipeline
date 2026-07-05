"""Beta-launch blockers PIPE-01, PIPE-03, PIPE-04.

Privacy/rights fail-open gates in the capture pipeline. See
docs/beta-launch-audit-2026-07-03/blueprint-capture-pipeline/SPECS.md.

PIPE-01 — raw un-redacted walkthrough must not reach a buyer-facing "launchable"
          artifact; runtime status must fail closed on privacy/rights not cleared,
          and the render descriptor / conditioning local_paths must never fall back
          to the raw walkthrough.
PIPE-03 — privacy is off by default; the qualification privacy_postprocess_gate must
          not PASS on ``not_run`` for delivery (buyer/reviewer-facing) runs.
PIPE-04 — the WorldLabs preview input video must be gated on derived-generation
          rights, mirroring scene-memory readiness.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import blueprint_pipeline.evaluation_prep_stage as eps
import blueprint_pipeline.qualification as qual


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

_CLEARED_PRIVACY_REVIEW = {"status": "cleared", "pipeline_status": "person_removed"}
_CLEARED_RIGHTS_REVIEW = {
    "status": "cleared",
    "privacy": _CLEARED_PRIVACY_REVIEW,
    "rights": {"status": "cleared"},
}


def _runtime_status_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        qualification_state="ready",
        downstream_evaluation_eligibility=True,
        scene_memory_bundle_manifest={"status": "complete"},
        object_geometry_manifest={},
        protected_regions_manifest={},
        required_runtime_artifact_paths=[],
        runtime_service_url="http://runtime.test",
        rights_review=_CLEARED_RIGHTS_REVIEW,
        privacy_processing={"status": "person_removed"},
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------------------
# PIPE-01: canonical runtime status fails closed on privacy/rights
# --------------------------------------------------------------------------------------


def test_pipe01_runtime_launchable_when_privacy_and_rights_cleared() -> None:
    status = eps._canonical_site_world_runtime_status(**_runtime_status_kwargs())
    assert status["launchable"] is True
    assert status["blockers"] == []


def test_pipe01_runtime_blocked_when_privacy_not_run() -> None:
    status = eps._canonical_site_world_runtime_status(
        **_runtime_status_kwargs(
            rights_review={
                "status": "needs_review",
                "privacy": {"status": "needs_review", "pipeline_status": "not_run"},
                "rights": {"status": "cleared"},
            },
            privacy_processing={"status": "not_run"},
        )
    )
    assert status["launchable"] is False
    assert any(blocker.startswith("privacy_processing") for blocker in status["blockers"])


def test_pipe01_runtime_blocked_when_privacy_failed_closed() -> None:
    status = eps._canonical_site_world_runtime_status(
        **_runtime_status_kwargs(
            rights_review={
                "status": "blocked",
                "privacy": {"status": "blocked", "pipeline_status": "failed_closed"},
                "rights": {"status": "cleared"},
            },
            privacy_processing={"status": "failed_closed"},
        )
    )
    assert status["launchable"] is False
    assert any(blocker.startswith("privacy_processing") for blocker in status["blockers"])


def test_pipe01_runtime_blocked_when_rights_needs_review() -> None:
    status = eps._canonical_site_world_runtime_status(
        **_runtime_status_kwargs(
            rights_review={
                "status": "needs_review",
                "privacy": _CLEARED_PRIVACY_REVIEW,
                "rights": {"status": "needs_review"},
            },
            privacy_processing={"status": "person_removed"},
        )
    )
    assert status["launchable"] is False
    assert any(blocker.startswith("rights_review") for blocker in status["blockers"])


def test_pipe01_runtime_blocked_when_rights_review_missing() -> None:
    status = eps._canonical_site_world_runtime_status(
        **_runtime_status_kwargs(rights_review=None, privacy_processing={"status": "person_removed"})
    )
    assert status["launchable"] is False
    assert any(blocker.startswith("rights_review") for blocker in status["blockers"])


# --------------------------------------------------------------------------------------
# PIPE-01: render descriptor / conditioning local_paths never expose raw walkthrough
# --------------------------------------------------------------------------------------


def _eps_context(tmp_path: Path):
    capture_root = tmp_path / "storage" / "local" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    pipeline_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        capture_root=capture_root,
        raw_root=raw_root,
        pipeline_root=pipeline_root,
        descriptor_path=descriptor_path,
        storage_root=tmp_path / "storage" / "local",
        bucket="local",
        scene_id="scene-1",
        capture_id="capture-1",
        capture_prefix="scenes/scene-1/captures/capture-1",
    )


def test_pipe01_render_descriptor_ignores_raw_video_fallback() -> None:
    # Even with poses+intrinsics present, a bare raw_video_path must not be treated as a
    # runtime render source. Without a privacy-safe video ref, we degrade rather than
    # embed the un-redacted walkthrough.
    descriptor = eps._primary_runtime_render_descriptor(
        conditioning_bundle={},
        local_paths={
            "raw_video_path": "/tmp/walkthrough.mov",
            "arkit_poses_path": "/tmp/poses.jsonl",
            "arkit_intrinsics_path": "/tmp/intrinsics.json",
        },
        canonical_world_model={
            "status": "missing",
            "world_model_backend": "site_world_runtime",
            "scene_representation": "pending_world_model_service",
            "render_source": "pending_world_model_service",
        },
    )
    # The raw walkthrough must not become the "full capture" runtime render source.
    assert descriptor["runtime_render_source"] != "site_world_runtime_full_capture"
    assert descriptor["runtime_render_source"] == "pending_world_model_service"


def test_pipe01_render_descriptor_accepts_privacy_safe_world_model_video() -> None:
    descriptor = eps._primary_runtime_render_descriptor(
        conditioning_bundle={"world_model_video_uri": "gs://bucket/privacy/final_walkthrough.mov"},
        local_paths={
            "arkit_poses_path": "/tmp/poses.jsonl",
            "arkit_intrinsics_path": "/tmp/intrinsics.json",
        },
        canonical_world_model={"status": "missing", "world_model_backend": "site_world_runtime"},
    )
    assert descriptor["runtime_render_source"] == "site_world_runtime_full_capture"


def test_pipe01_conditioning_local_paths_never_include_raw_video(tmp_path: Path) -> None:
    context = _eps_context(tmp_path)
    # Populate the un-redacted raw walkthrough on disk.
    (context.raw_root / "walkthrough.mov").write_bytes(b"raw-capture")
    local_paths = eps._conditioning_local_paths(context=context, conditioning_bundle={})
    assert "raw_video_path" not in local_paths
    assert "/walkthrough.mov" not in str(local_paths.values())


# --------------------------------------------------------------------------------------
# PIPE-03: privacy_postprocess_gate must not PASS on not_run for delivery runs
# --------------------------------------------------------------------------------------


def test_pipe03_privacy_gate_passes_not_run_for_non_delivery_runs() -> None:
    gate = qual._privacy_postprocess_gate(
        privacy_status="not_run",
        delivery_run=False,
    )
    assert gate.passed is True


def test_pipe03_privacy_gate_blocks_not_run_for_delivery_runs() -> None:
    gate = qual._privacy_postprocess_gate(
        privacy_status="not_run",
        delivery_run=True,
    )
    assert gate.passed is False


def test_pipe03_privacy_gate_passes_cleared_status_for_delivery_runs() -> None:
    gate = qual._privacy_postprocess_gate(
        privacy_status="person_removed",
        delivery_run=True,
    )
    assert gate.passed is True


def test_pipe03_privacy_gate_blocks_fallback_redaction_for_delivery_runs() -> None:
    for privacy_status in (
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    ):
        gate = qual._privacy_postprocess_gate(
            privacy_status=privacy_status,
            delivery_run=True,
        )
        assert gate.passed is False
        assert "verified privacy removal" in gate.detail


def test_pipe03_privacy_gate_blocks_failed_closed_always() -> None:
    for delivery_run in (False, True):
        gate = qual._privacy_postprocess_gate(
            privacy_status="failed_closed",
            delivery_run=delivery_run,
        )
        assert gate.passed is False


# --------------------------------------------------------------------------------------
# PIPE-04: WorldLabs preview input gated on derived-generation rights
# --------------------------------------------------------------------------------------


def test_pipe04_worldlabs_rights_gate_absent_rights_blocks() -> None:
    assert qual._worldlabs_derived_rights_allowed(metadata={}) is False
    assert (
        qual._worldlabs_derived_rights_allowed(
            metadata={"capture_rights": {"derived_scene_generation_allowed": False}}
        )
        is False
    )


def test_pipe04_worldlabs_rights_gate_present_rights_allows() -> None:
    assert (
        qual._worldlabs_derived_rights_allowed(
            metadata={"capture_rights": {"derived_scene_generation_allowed": True}}
        )
        is True
    )
