"""The control-plane render pointer must survive the provider's own render.

The provider packet carries two different digests for the same render inputs:
``result_digest`` identifies the portable record the provider was handed, and
``control_plane_result_digest`` points back at the authoritative control-plane
record that the packet was derived from (see
``task_evaluation_scene_configuration_bundle`` where the portable record is
sealed).  Stage one's adapter validates the provider render handoff against the
*control-plane* pointer carried by the envelope it executes under, so the
provider must not rewrite that pointer to its own local digest while finishing
the render it owed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    _target_camera_ring,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pending_provider_packet(
    tmp_path: Path, *, control_plane_result_digest: str | None
) -> tuple[dict[str, object], Path]:
    """Build the render-inputs packet a provider is handed before it renders."""

    from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
        resolve_scene_configuration_disclosure,
    )

    splat = tmp_path / "scene.ply"
    splat.write_bytes(b"raw-interiorgs-source")
    calibration = tmp_path / "cameras.json"
    ring = _target_camera_ring(
        minimum_xyz=[2.91, -6.83, 0.754], maximum_xyz=[3.04, -6.69, 0.884]
    )
    calibration.write_text(
        json.dumps(
            [
                {
                    "id": row["camera_id"],
                    "spec": {
                        "pose": {
                            "T_world_camera_opencv": row[
                                "T_world_camera_provider_frame"
                            ]
                        },
                        "intrinsics": row["intrinsics"],
                    },
                }
                for row in ring
            ]
        ),
        encoding="utf-8",
    )
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration={
            "provider_disclosure": {
                "raw_interiorgs_bytes": True,
                "derived_rendered_views": True,
            },
            "human_authority": {
                "authority_reference": "operator-2026-08-30",
                "provider_retention_terms_accepted": True,
                "provider_training_authorized": False,
            },
        },
        rights_admission={
            "provider_disclosure": {
                "raw_interiorgs_downloaded_bytes_may_be_uploaded": True,
                "provider_training_allowed": False,
                "public_redistribution_allowed": False,
                "provider_retention_rule": "bounded_then_teardown",
            }
        },
    )
    pending: dict[str, object] = {
        "schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": "derived_method_inputs_pending_provider_render",
        "source_splat_digest": _sha256(splat),
        "raw_interiorgs_bytes_in_provider_packet": True,
        "disclosure_decision": decision,
        "render_execution_site": "provider_gpu",
        "source_appearance": {"path": str(splat), "digest": _sha256(splat)},
        "camera_calibration": {"path": str(calibration)},
        "source_object_masks": {
            "source": "registered_source_object_bounds_projection"
        },
        "derived_frames": [],
        "derived_frame_count": 0,
        "render_manifest": None,
        "result_digest": "sha256:" + "c" * 64,
    }
    if control_plane_result_digest is not None:
        pending["control_plane_result_digest"] = control_plane_result_digest
    return pending, splat


def _install_provider_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make the module believe it is executing from a provider bundle."""

    from blueprint_pipeline import (
        task_evaluation_scene_configuration_render_inputs as render_module,
    )
    from blueprint_pipeline.task_evaluation_splat_render_runtime import (
        SCENE_CONFIGURATION_BUNDLE_SCHEMA_VERSION,
    )

    provider_runtime = tmp_path / "provider_runtime"
    provider_package = provider_runtime / "blueprint_pipeline"
    provider_package.mkdir(parents=True)
    fake_module = (
        provider_package
        / "task_evaluation_scene_configuration_render_inputs.py"
    )
    fake_module.write_text("# provider copy\n", encoding="utf-8")
    (
        provider_runtime / f"{SCENE_CONFIGURATION_BUNDLE_SCHEMA_VERSION}.json"
    ).write_text("{}\n", encoding="utf-8")
    provider_renderer = provider_runtime / "renderer"
    provider_renderer.mkdir()
    monkeypatch.setattr(render_module, "__file__", str(fake_module))
    monkeypatch.setattr(
        render_module,
        "runtime_from_provider_bundle",
        lambda *, provider_runtime_root: {
            "node": "/provider/node",
            "browser_executable": "/provider/chrome",
            "renderer_root": str(provider_renderer),
            "repository_root": str(provider_renderer),
            "identity": {"mode": "provider_bundle"},
        },
    )


def _stub_renderer(**kwargs: object) -> dict[str, object]:
    """Stand in for the GPU rasteriser with deterministic frames."""

    output = Path(str(kwargs["output_dir"]))
    frames = output / "frames"
    frames.mkdir(parents=True)
    rows = []
    for camera in kwargs["cameras"]:  # type: ignore[union-attr]
        frame = frames / f"{camera['camera_id']}.png"
        Image.new("RGB", (1024, 1024), color=(12, 34, 56)).save(frame)
        rows.append(
            {
                "camera_id": camera["camera_id"],
                "relative_path": f"frames/{frame.name}",
                "digest": _sha256(frame),
                "width": 1024,
                "height": 1024,
            }
        )
    result = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "authorization_class": "method_input",
        "splat_digest": kwargs["source_splat_digest"],
        "renders": rows,
        "render_count": len(rows),
        "sealed_camera_render_manifest_digest": "",
    }
    result["sealed_camera_render_manifest_digest"] = canonical_digest(
        result, digest_field="sealed_camera_render_manifest_digest"
    )
    return result


def _complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    control_plane_result_digest: str | None,
) -> dict[str, object]:
    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
        complete_provider_render_inputs,
    )

    pending, splat = _pending_provider_packet(
        tmp_path, control_plane_result_digest=control_plane_result_digest
    )
    _install_provider_runtime(tmp_path, monkeypatch)
    return complete_provider_render_inputs(
        render_inputs=pending,
        appearance_path=splat,
        source_object={
            "aabb_min_xyz_m": [2.91, -6.83, 0.754],
            "aabb_max_xyz_m": [3.04, -6.69, 0.884],
        },
        output_root=tmp_path / "provider-out",
        renderer=_stub_renderer,
    )


def test_provider_completion_preserves_the_control_plane_render_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finishing the owed render must not repoint the packet at itself.

    The portable packet already names the control-plane record it came from.
    Overwriting that name with the portable record's own digest silently
    detaches the completed render from the only record stage one's adapter can
    compare against.
    """

    control_plane_digest = "sha256:" + "a" * 64
    completed = _complete(
        tmp_path, monkeypatch, control_plane_result_digest=control_plane_digest
    )

    assert completed["control_plane_result_digest"] == control_plane_digest
    assert completed["result_digest"] != control_plane_digest
    assert completed["result_digest"] == canonical_digest(
        completed, digest_field="result_digest"
    )


def test_provider_completion_falls_back_to_the_packet_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A packet with no control-plane pointer still gains one from itself."""

    completed = _complete(
        tmp_path, monkeypatch, control_plane_result_digest=None
    )

    assert completed["control_plane_result_digest"] == "sha256:" + "c" * 64


def test_render_handoff_binds_the_pointer_stage_one_validates_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handoff must carry the digest the adapter actually compares to.

    ``execute_artifixer3d_observed_object_removal`` compares the handoff's
    ``control_plane_render_result_digest`` against the
    ``control_plane_result_digest`` on the envelope's render-inputs result.  If
    the provider rewrote that pointer while completing the render, the handoff
    names the portable digest instead and stage one refuses an otherwise
    accepted appearance as ``artifixer3d_object_removal_result_invalid``.
    """

    from blueprint_pipeline.task_evaluation_scene_configuration_render_handoff import (
        materialize_provider_render_handoff,
        validate_provider_render_handoff,
    )

    control_plane_digest = "sha256:" + "a" * 64
    completed = _complete(
        tmp_path, monkeypatch, control_plane_result_digest=control_plane_digest
    )
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    record = materialize_provider_render_handoff(
        render_inputs=completed, output_root=handoff_root
    )
    manifest, _frames = validate_provider_render_handoff(record["path"])

    assert manifest["control_plane_render_result_digest"] == control_plane_digest
    assert manifest["render_completed_on_provider"] is True
    assert manifest["source_render_result_digest"] == completed["result_digest"]
