"""One command provisions the whole configured-controls continuation for a fresh scene.

For 839873 the continuation inputs were nineteen per-commit directories of
hand-written files: a runtime bundle rebound by editing a manifest, a camera
template lifted from an older packet, a trajectory plan lifted from an older
diagnostic, and per-phase authority files typed by hand.  The provisioner
derives every one of them from the owner's task request, the scene's
preparation result, the deployed commit, and the retained runtime payload,
registers the intent, and leaves the first-run-only inputs deferred to the
autostart.  It allocates nothing and never touches a provider.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import (
    task_evaluation_configured_controls_continuation_provisioning as provisioning,
)
from blueprint_pipeline import task_evaluation_configured_controls_autostart as autostart
from blueprint_pipeline import (
    task_evaluation_configured_controls_deferred_inputs as deferred,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_configured_controls_autostart_support import (
    _placement_aware_camera_candidates,
)
from blueprint_pipeline.task_evaluation_shared_mutation_window import (
    validate_shared_mutation_window_template,
)
from blueprint_pipeline.vast_evidence_contracts import VAST_PROVIDER_ZERO_API_CALL


COMMIT = "308a0fd77" + "1" * 31
TEAM = "blueprint-adp"
SCENE_ID = "interiorgs-841757"
TASK_ID = "scene-841757-book-to-tray"
PREPARATION_ID = "adp-new-scene-book-to-tray-841757-308a0fd7-20260905t040000z-preparation"
NOW = datetime(2026, 9, 5, 4, 5, 0, tzinfo=timezone.utc)
WRIST_MATRIX = [
    -0.3128833, 0.0076292, 0.9497609, 0.011,
    -0.9496289, 0.0159892, -0.3129682, -0.031,
    -0.0175737, -0.9998430, 0.0022421, -0.074,
    0.0, 0.0, 0.0, 1.0,
]


def _write(path: Path, value: dict | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_row(path: Path, contract_path: str) -> dict:
    return {
        "contract_path": contract_path,
        "uri": f"s3://blueprint/task-evaluation/production-inputs/ns/{path.name}",
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
        "materialized_path": str(path),
        "full_byte_service_account_readback_passed": True,
    }


def _preparation(
    tmp_path: Path, *, run_mode: str = "scene_configuration", commit: str = COMMIT
) -> Path:
    prepared = tmp_path / "prepared-references" / PREPARATION_ID
    template = _write(
        prepared / "task_template.json",
        {
            "schema_version": "task_evaluation_rigid_relocation_template.v1",
            "task_identity": {"id": TASK_ID, "version": "v1"},
            "object_identity": {"id": "scene-841757-book-replacement", "version": "v1"},
            "strategy": "pick_and_place",
            "start_center_xyz_m": [3.25, -6.56, 0.29],
            "target_center_xyz_m": [3.25, -6.2673, 0.29],
        },
    )
    mount = _write(prepared / "robot_mount_interface_plan.json", {"schema_version": "task_evaluation_robot_mount_interface_plan.v1"})
    calibration = _write(prepared / "camera_calibration_plan.json", {"schema_version": "task_evaluation_scene_camera_calibration_plan.v1"})
    rights = _write(prepared / "rights_admission.json", {"schema_version": "task_evaluation_scene_rights_admission.v1", "status": "admitted_for_internal_development"})
    references = [
        _reference_row(template, "task.definition"),
        _reference_row(mount, "scene.registration.robot_mount_interface"),
        _reference_row(calibration, "scene.registration.camera_calibration"),
        _reference_row(rights, "scene.rights.admission"),
    ]
    request = {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": run_mode,
        "expected_production_commit": commit,
        "preparation_id": PREPARATION_ID,
        "team_namespace": TEAM,
        "run_id": PREPARATION_ID.removesuffix("-preparation") + "-scene-configuration",
        "scene": {"identity": {"id": SCENE_ID, "version": "book-tray-v1"}},
        "task": {"identity": {"id": TASK_ID, "version": "v1"}},
    }
    request_digest = canonical_digest(request)
    queue = tmp_path / "preparations"
    envelope = {
        "schema_version": "task_evaluation_launch_preparation_intake_envelope.v1",
        "request_digest": request_digest,
        "request": request,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    suffix = request_digest.removeprefix("sha256:")
    _write(queue / "materialized" / f"{PREPARATION_ID}-{suffix}.json", envelope)
    result = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "queued_for_production_scene_configuration",
        "preparation_id": PREPARATION_ID,
        "run_mode": run_mode,
        "team_namespace": TEAM,
        "source_commit": commit,
        "references": references,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return _write(queue / "results" / f"{PREPARATION_ID}-{suffix}.json", result)


def _embodiment_camera_template(tmp_path: Path) -> Path:
    intrinsics = {"cx": 159.5, "cy": 89.5, "fx": 172.88839142740494, "fy": 172.88839142740494, "height": 180, "width": 320}
    cameras = [
        {"role": "external", "pose_frame": "world", "parent_prim_path": "{ENV_REGEX_NS}", "policy_input": True, "scoring_input": False, "optical_convention": "opencv", "frame_from_camera_matrix": [1.0] + [0.0] * 14 + [1.0], "intrinsics": intrinsics},
        {"role": "wrist", "pose_frame": "robot_body", "parent_prim_path": "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link", "policy_input": True, "scoring_input": False, "optical_convention": "opencv", "frame_from_camera_matrix": WRIST_MATRIX, "intrinsics": intrinsics},
        {"role": "overview", "pose_frame": "world", "parent_prim_path": "{ENV_REGEX_NS}", "policy_input": False, "scoring_input": False, "optical_convention": "opencv", "frame_from_camera_matrix": [1.0] + [0.0] * 14 + [1.0], "intrinsics": intrinsics},
    ]
    return _write(tmp_path / "embodiment" / "droid_camera_template.json", {"schema_version": "native_task_arena_packet_request.v1", "cameras": cameras})


def _payload(tmp_path: Path) -> Path:
    payload = tmp_path / "runtime-payload"
    _write(payload / "native_task_runtime_source_packet.v1.json", {"schema_version": "native_task_runtime_source_packet.v1", "status": "sealed"})
    _write(payload / "native_task_runtime_sources.zip", b"PK" + b"\x00" * 4096)
    return payload


def _provider_zero() -> dict:
    zero = {
        "api_command": list(VAST_PROVIDER_ZERO_API_CALL),
        "api_confirmed": True,
        "global_live_resource_count": 0,
        "inventory": [],
        "observed_at_utc": NOW.isoformat(),
        "provider": "vast",
        "provider_zero": True,
        "raw_secret_values_recorded": False,
        "schema_version": "adp_paid_provider_zero.v1",
        "stderr_present": False,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(zero, digest_field="provider_zero_digest")
    return zero


class _Publisher:
    def __init__(self) -> None:
        self.published: dict[str, tuple[str, bytes]] = {}
        self.layers: list[dict] = []

    def artifact(self, *, path: Path, artifact_kind: str) -> dict:
        payload = Path(path).read_bytes()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        uri = (
            "s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/"
            f"{artifact_kind}/sha256/{digest.removeprefix('sha256:')}/{Path(path).name}"
        )
        self.published[uri] = (artifact_kind, payload)
        return {
            "schema_version": "task_evaluation_scene_artifact_reference.v1",
            "status": "remote_verified",
            "artifact_kind": artifact_kind,
            "uri": uri,
            "digest": digest,
            "size_bytes": len(payload),
            "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": digest,
            "readback_size_bytes": len(payload),
        }

    def layers_of(self, receipt: dict) -> dict:
        rows = []
        for row in receipt["external_layers"]:
            self.layers.append(dict(row))
            rows.append({"uri": row["uri"], "digest": row["sha256"], "size_bytes": row["size_bytes"], "relative_path": row["relative_path"]})
        return {"schema_version": "task_evaluation_runtime_source_layer_publication.v1", "status": "remote_verified", "layer_count": len(rows), "layers": rows}


def _provision(tmp_path: Path, **overrides):
    publisher = _Publisher()
    kwargs = dict(
        expected_production_commit=COMMIT,
        preparation_result_path=_preparation(tmp_path),
        preparation_queue_root=tmp_path / "preparations",
        robot_asset_usd_path=_write(tmp_path / "robot" / "franka.usd", b"#usda 1.0\n"),
        runtime_source_payload_dir=_payload(tmp_path),
        embodiment_camera_template_path=_embodiment_camera_template(tmp_path),
        project_spend_reconciliation_path=_write(tmp_path / "spend.json", {"schema_version": "adp_project_spend_reconciliation.v1", "total_cost_usd": 61.42}),
        controls_root=tmp_path / "controls" / "scene841757-308a0fd7",
        profile_dir=tmp_path / "profiles",
        authorization_reference="Blueprint owner direction 2026-09-04: scene 841757 book-to-tray end to end",
        authorized_by="Blueprint-owner",
        release_reference="Scene 841757 configured-controls automatic continuation",
        openai_project_id="proj_test",
        openai_api_key_id="key_visual_review",
        policy_camera_resolution=(640, 360),
        overview_camera_resolution=(1280, 720),
        provider_zero_collector=_provider_zero,
        artifact_publisher=publisher.artifact,
        layer_publisher=publisher.layers_of,
        external_layer_bucket="blueprint",
        external_layer_min_bytes=1024,
        now=NOW,
    )
    kwargs.update(overrides)
    return provisioning.provision_configured_controls_continuation(**kwargs), publisher


def test_provisioning_registers_a_deferred_intent_bound_to_every_derived_input(tmp_path: Path) -> None:
    result, publisher = _provision(tmp_path)
    assert result["status"] == "configured_controls_continuation_provisioned"
    intent = json.loads(Path(result["intent_path"]).read_text())
    assert autostart.validate_configured_controls_autostart_intent(intent) == intent
    assert intent["expected_production_commit"] == COMMIT
    assert intent["team_namespace"] == TEAM
    assert intent["scene_id"] == SCENE_ID
    assert intent["task_id"] == TASK_ID
    assert intent["target_position_world_m"] == [3.25, -6.2673, 0.29]
    assert intent["configuration_adoption"] == {"mode": "same_commit_automatic"}
    assert deferred.deferred_declarations(intent["paths"]) == {
        "native_trajectory_plan_path": deferred.TRAJECTORY_MODE,
        "overview_image_paths": deferred.OVERVIEW_MODE,
    }
    paths = intent["paths"]
    assert Path(paths["robot_mount_interface_path"]).name == "robot_mount_interface_plan.json"
    assert Path(paths["scene_camera_calibration_path"]).name == "camera_calibration_plan.json"
    assert Path(paths["robot_asset_usd_path"]).name == "franka.usd"
    assert Path(result["intent_path"]).name == autostart.configured_controls_autostart_registry_name(
        team_namespace=TEAM, scene_id=SCENE_ID, task_id=TASK_ID
    )


def test_provisioning_authors_the_runtime_binding_with_a_deferred_scene_mount_and_published_bundle(
    tmp_path: Path,
) -> None:
    result, publisher = _provision(tmp_path)
    intent = json.loads(Path(result["intent_path"]).read_text())
    binding = json.loads(Path(intent["paths"]["runtime_binding_path"]).read_text())
    assert set(binding) == {"runtime", "execution_adapter", "spend"}
    runtime = binding["runtime"]
    assert runtime["identity"] == {"id": "native-arena", "version": "isaac-2026-1"}
    assert runtime["mounts"][0] == {
        "source": {"deferred": deferred.SCENE_BUNDLE_MODE},
        "container_path": "/inputs",
        "mode": "read_only",
    }
    assert runtime["network"] == {"default": "deny", "allowlist": []}
    assert runtime["secret_refs"] == []
    health = runtime["health_protocol"]
    assert publisher.published[health["uri"]][0] == "native-health-protocol"
    protocol = json.loads(publisher.published[health["uri"]][1])
    assert protocol["schema_version"] == "task_evaluation_native_arena_health_protocol.v1"
    assert protocol["source_commit"] == COMMIT
    assert protocol["health_protocol_digest"] == canonical_digest(protocol, digest_field="health_protocol_digest")
    bundle = binding["execution_adapter"]["runtime_source_bundle"]
    kind, wrapper = publisher.published[bundle["uri"]]
    assert kind == "native-runtime-source"
    assert bundle["digest"] == "sha256:" + hashlib.sha256(wrapper).hexdigest()
    # The 4 GB payload member is stored once by digest and referenced, so the
    # wrapper stays small and the same runtime is never uploaded twice.
    assert len(wrapper) < 64 * 1024
    assert [row["relative_path"] for row in publisher.layers] == ["payload/native_task_runtime_sources.zip"]
    assert binding["spend"] == {
        "maximum_hourly_rate_usd": 0.8,
        "hard_cap_usd": 2.0,
        "hard_ttl_seconds": 9000,
        "retry_cap": 0,
        "selected_provider": "vast",
        "provider_allowlist": ["vast"],
    }


def test_provisioning_authors_phase_authority_from_the_owner_and_the_scene_rights(tmp_path: Path) -> None:
    result, publisher = _provision(tmp_path)
    intent = json.loads(Path(result["intent_path"]).read_text())
    phases = intent["phases"]
    assert set(phases) == {"construction", "controls"}
    for phase, rows in phases.items():
        template = json.loads(Path(rows["release_window_template_path"]).read_text())
        validate_shared_mutation_window_template(template, team_namespace=TEAM, expected_production_commit=COMMIT)
        assert template["maximum_hard_cap_usd"] == 2.0
        authorization = json.loads(Path(rows["authorization_path"]).read_text())
        assert authorization["authorized_by"] == "Blueprint-owner"
        assert authorization["reference"].startswith("Blueprint owner direction")
        authority = json.loads(Path(rows["launch_authority_path"]).read_text())
        assert authority["max_spend_usd"] == 2.0
        assert authority["rights_scope"] == "internal_noncommercial_research_and_development_Task_Evaluation_Run"
        assert authority["rights_evidence"]["uri"].endswith("rights_admission.json")
        assert authority["expires_at"] == authorization["standing_authorization_expires_at"]
    lineage = json.loads(Path(phases["construction"]["lineage_path"]).read_text())
    assert lineage["kind"] == "initial_project"
    assert publisher.published[lineage["initial_provider_zero"]["uri"]][0] == "initial-provider-zero"
    assert publisher.published[lineage["project_spend_reconciliation"]["uri"]][0] == "prior-official-spend"
    assert "lineage_path" not in phases["controls"]


def test_provisioning_authors_policy_and_overview_cameras_at_the_requested_resolutions(tmp_path: Path) -> None:
    result, _publisher = _provision(tmp_path)
    intent = json.loads(Path(result["intent_path"]).read_text())
    template = json.loads(Path(intent["paths"]["cameras_path"]).read_text())
    assert template["schema_version"] == "native_task_arena_packet_request.v1"
    assert template["world_camera_poses_authoritative"] is False
    by_role = {row["role"]: row for row in template["cameras"]}
    assert by_role["external"]["intrinsics"] == by_role["wrist"]["intrinsics"]
    assert by_role["external"]["intrinsics"]["width"] == 640
    assert by_role["external"]["intrinsics"]["height"] == 360
    assert by_role["external"]["intrinsics"]["cx"] == pytest.approx(319.5)
    assert by_role["external"]["intrinsics"]["fx"] == pytest.approx(172.88839142740494 * 2.0)
    assert by_role["overview"]["intrinsics"]["width"] == 1280
    assert by_role["overview"]["intrinsics"]["fx"] == pytest.approx(172.88839142740494 * 4.0)
    assert by_role["overview"]["policy_input"] is False
    assert by_role["wrist"]["frame_from_camera_matrix"] == WRIST_MATRIX
    # The derived world cameras keep the overview's own intrinsics.
    plan = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "pick_and_place",
        "execution_parameters": {"arrival_tolerance_m": 0.02, "arrival_orientation_tolerance_rad": 0.08},
        "phases": [
            {"phase_id": "pregrasp", "position_world_m": [3.25, -6.56, 0.45], "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0], "gripper_state": "open", "gate_ids": ["a"]},
            {"phase_id": "place", "position_world_m": [3.25, -6.27, 0.45], "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0], "gripper_state": "open", "gate_ids": ["b"]},
        ],
        "phase_count": 2,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
        placement_trajectory_from_native_plan,
    )

    derived = _placement_aware_camera_candidates(
        camera_template=template,
        accepted_pose={"position_world_m": [2.6, -6.4, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        selected_candidate_id="candidate-0001",
        trajectory=placement_trajectory_from_native_plan(plan),
        source_commit=COMMIT,
    )
    derived_by_role = {row["role"]: row for row in derived["cameras"]}
    assert derived_by_role["overview"]["intrinsics"]["width"] == 1280
    assert derived_by_role["external"]["intrinsics"]["width"] == 640


def test_provisioning_is_idempotent_and_refuses_a_non_configuration_preparation(tmp_path: Path) -> None:
    first, _p1 = _provision(tmp_path)
    second, _p2 = _provision(
        tmp_path,
        preparation_result_path=next((tmp_path / "preparations" / "results").glob("*.json")),
        robot_asset_usd_path=tmp_path / "robot" / "franka.usd",
        runtime_source_payload_dir=tmp_path / "runtime-payload",
        embodiment_camera_template_path=tmp_path / "embodiment" / "droid_camera_template.json",
        project_spend_reconciliation_path=tmp_path / "spend.json",
    )
    assert second["intent_digest"] == first["intent_digest"]
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(
        provisioning.ConfiguredControlsProvisioningError,
        match="configured_controls_provisioning_preparation_not_scene_configuration",
    ):
        _provision(
            other,
            preparation_result_path=_preparation(other, run_mode="episode_evaluation"),
            preparation_queue_root=other / "preparations",
            controls_root=other / "controls",
        )


COMMIT_NEXT = "1359447d4" + "2" * 31


def test_registry_install_supersedes_a_stale_release_registration(tmp_path: Path, monkeypatch) -> None:
    """Every deploy changes the commit; the same continuation re-installs without a conflict."""

    result, _publisher = _provision(tmp_path)
    registry = tmp_path / "registry"
    registry.mkdir()
    first = provisioning.install_intent_into_registry(
        intent_path=result["intent_path"], intent_root=registry,
        expected_production_commit=COMMIT, service_group=None,
    )
    from blueprint_pipeline import task_evaluation_intent_registry as registry_module
    monkeypatch.setattr(registry_module, "_supersession_authority", lambda commit: None)
    next_result, _next_publisher = _provision(
        tmp_path,
        expected_production_commit=COMMIT_NEXT,
        preparation_result_path=_preparation(tmp_path / "next", commit=COMMIT_NEXT),
        preparation_queue_root=tmp_path / "next" / "preparations",
        controls_root=tmp_path / "controls" / "scene841757-1359447d",
    )
    second = provisioning.install_intent_into_registry(
        intent_path=next_result["intent_path"], intent_root=registry,
        expected_production_commit=COMMIT_NEXT, service_group=None,
    )
    assert second["registry_path"] == first["registry_path"]
    assert Path(second["registry_path"]).read_bytes() == Path(next_result["intent_path"]).read_bytes()
    identity = Path(first["registry_path"]).name.removesuffix(".json")
    retired = registry / f"{identity}.superseded-{COMMIT}.json"
    assert retired.read_bytes() == Path(result["intent_path"]).read_bytes()
    assert retired.stat().st_mode & 0o777 == 0o440
    assert sorted(path.name for path in registry.glob("*.json")) == sorted([f"{identity}.json", retired.name])
    # Re-installing the live release is a no-op; another decision at that release is refused.
    assert provisioning.install_intent_into_registry(
        intent_path=next_result["intent_path"], intent_root=registry,
        expected_production_commit=COMMIT_NEXT, service_group=None,
    )["registry_path"] == first["registry_path"]
    conflicting = json.loads(Path(next_result["intent_path"]).read_text())
    conflicting["authorization_reference"] = "another decision at the same release"
    conflicting["intent_digest"] = canonical_digest(conflicting, digest_field="intent_digest")
    conflicting_path = _write(tmp_path / "controls" / "conflicting" / Path(next_result["intent_path"]).name, conflicting)
    with pytest.raises(provisioning.ConfiguredControlsProvisioningError):
        provisioning.install_intent_into_registry(
            intent_path=conflicting_path, intent_root=registry,
            expected_production_commit=COMMIT_NEXT, service_group=None,
        )


def test_registry_install_writes_the_exact_intent_bytes_read_only(tmp_path: Path) -> None:
    result, _publisher = _provision(tmp_path)
    registry = tmp_path / "registry"
    registry.mkdir()
    installed = provisioning.install_intent_into_registry(
        intent_path=result["intent_path"],
        intent_root=registry,
        expected_production_commit=COMMIT,
        service_group=None,
    )
    target = Path(installed["registry_path"])
    assert target.parent == registry
    assert target.read_bytes() == Path(result["intent_path"]).read_bytes()
    assert target.stat().st_mode & 0o777 == 0o440
    again = provisioning.install_intent_into_registry(
        intent_path=result["intent_path"],
        intent_root=registry,
        expected_production_commit=COMMIT,
        service_group=None,
    )
    assert again["registry_path"] == installed["registry_path"]
    with pytest.raises(
        provisioning.ConfiguredControlsProvisioningError,
        match="configured_controls_provisioning_intent_commit_mismatch",
    ):
        provisioning.install_intent_into_registry(
            intent_path=result["intent_path"],
            intent_root=registry,
            expected_production_commit="2" * 40,
            service_group=None,
        )
