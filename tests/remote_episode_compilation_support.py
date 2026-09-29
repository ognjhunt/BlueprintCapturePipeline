"""Plan 14 PR 4 test support: one control-plane host's episode-compilation paths under a fake ``/``.

``Host`` lays the production paths out under ``tmp_path/fs`` (``/var/lib/blueprint/...``), so the
host's own paths and the worker's descriptor paths differ only by that root, exactly as a Cloud Run
execution's in-memory ``/var/lib/blueprint`` differs from the host's.  ``stage`` writes one sealed
compilation envelope with its materialized references: a configured-scene bundle whose appearance
member is chosen by the test, a runtime-source bundle with a member manifest, and a robot document.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.remote_cpu_environment import DIGESTED_FIELDS
from blueprint_pipeline.task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive
from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import MANIFEST_NAME
from blueprint_pipeline.task_evaluation_scene_construction_queue import ensure_scene_construction_queue_root
from tests.remote_cpu_allocator_fakes import IMAGE, JOB_SHORT, environment
from tests.test_task_evaluation_launch_preparation_contract import request

QUEUE = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations"
INPUTS = "/var/lib/blueprint/task-evaluation-inputs/prepared-references"
OUTPUTS = "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes"
JOBS = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
CACHE = "/var/lib/blueprint/task-evaluation-inputs/particlefield-runtime-assets"
RESERVATIONS = "/var/lib/blueprint/pipeline-control-plane/disk-reservations"
PINS = "/var/lib/blueprint/pipeline-control-plane/storage-pins"
BUNDLE_MANIFEST = "configured_scene_bundle_candidate.v1.json"
HOST_RECORD = environment()


def digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def nurec_usdz(size_bytes: int = 4096) -> bytes:
    """A USDZ-shaped zip holding a ``.nurec`` payload: what a NuRec appearance looks like from outside."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("default.usda", b"#usda 1.0\n")
        archive.writestr("aura_appearance.nurec", (b"nurec" * (size_bytes // 5 + 1))[:size_bytes])
    return buffer.getvalue()


def configured_bundle(appearance: bytes, *, appearance_name: str = "appearance.usda") -> bytes:
    """A configured-scene bundle as the compiler extracts one: three assets and a sealed manifest."""

    payloads = {appearance_name: appearance, "collision.usda": b"#usda 1.0\n# collision\n",
                "replacement.usda": b"#usda 1.0\n# replacement\n"}
    roles = {appearance_name: "appearance", "collision.usda": "collision", "replacement.usda": "replacement"}
    manifest = {
        "schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
        "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
        "robot_specific_base_registration_included": False,
        "assets": [{"role": roles[name], "relative_path": name, "digest": digest_of(data), "size_bytes": len(data)}
                   for name, data in payloads.items()],
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr(BUNDLE_MANIFEST, json.dumps(manifest, sort_keys=True) + "\n")
    return buffer.getvalue()


def runtime_bundle(member_bytes: int = 2048) -> bytes:
    """A runtime-source wrapper whose manifest names its members, as the preparation adapter seals one."""

    members = {"runtime/packet.bin": b"r" * member_bytes, "runtime/config.json": b'{"runtime": 1}\n'}
    entries = [{"relative_path": name, "sha256": digest_of(data), "size_bytes": len(data)}
               for name, data in sorted(members.items())]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in sorted(members.items()):
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data)
        archive.writestr(zipfile.ZipInfo(MANIFEST_NAME, (1980, 1, 1, 0, 0, 0)),
                         json.dumps({"entries": entries}, sort_keys=True))
    return buffer.getvalue()


class Host:
    """The episode-compilation paths of one control-plane host, rooted at ``tmp_path/fs``."""

    def __init__(self, tmp_path: Path) -> None:
        self.fs = tmp_path / "fs"
        self.queue = ensure_scene_construction_queue_root(self.local(QUEUE))
        (self.queue / "results").mkdir(mode=0o750, exist_ok=True)
        self.inputs, self.outputs = self.local(INPUTS), self.local(OUTPUTS)
        self.jobs, self.cache = self.local(JOBS), self.local(CACHE)
        for path in (self.inputs, self.outputs, self.jobs):
            path.mkdir(parents=True, exist_ok=True)

    def local(self, path: str) -> Path:
        return self.fs / path.lstrip("/")

    def stage(self, *, label: str = "prep-1", appearance: bytes | None = None, appearance_name: str = "appearance.usda",
              runtime_member_bytes: int = 2048) -> tuple[dict[str, Any], str]:
        """Write one envelope's references and seal it into ``pending/``; returns it and its row name."""

        value = request()
        value["preparation_id"], value["run_id"] = f"{label}-v1", f"run-{label}-v1"
        value["task"]["configured_scene_revision_digest"] = "sha256:" + "9" * 64
        root = self.inputs / value["preparation_id"]
        root.mkdir(parents=True, exist_ok=True)
        files = {
            "scene.configured_revision.configured_scene_bundle": ("configured-scene.zip", configured_bundle(
                b"#usda 1.0\n# appearance\n" if appearance is None else appearance, appearance_name=appearance_name)),
            "execution_adapter.runtime_source_bundle": ("runtime-source.zip", runtime_bundle(runtime_member_bytes)),
            "robot.configuration": ("robot.json", b'{"robot": "franka"}\n'),
        }
        rows = []
        for contract_path, (name, data) in files.items():
            path = root / name
            path.write_bytes(data)
            path.chmod(0o440)
            rows.append({"contract_path": contract_path, "uri": f"s3://blueprint-production-inputs/{name}",
                         "digest": digest_of(data), "size_bytes": len(data), "materialized_path": str(path),
                         "full_byte_service_account_readback_passed": True})
        envelope: dict[str, Any] = {
            "schema_version": "task_evaluation_episode_compilation_envelope.v1",
            "compilation_id": value["preparation_id"], "preparation_id": value["preparation_id"],
            "run_id": value["run_id"], "team_namespace": value["team_namespace"],
            "expected_production_commit": value["expected_production_commit"],
            "configured_scene_revision_digest": value["task"]["configured_scene_revision_digest"],
            "configured_scene_bundle": {key: rows[0][key] for key in ("uri", "digest", "size_bytes")},
            "materialized_references": rows, "request": value, "preparation_result_digest": "sha256:" + "8" * 64,
            "automatic_progression_required": True, "robot_specific_episode_packet_compiled_in_production": True,
            "customer_supplied_prebuilt_episode_packet": False, "production_compiler_owns_episode_packet": True,
            "provider_mutation_performed": False, "paid_execution_requested": False, "envelope_digest": "",
        }
        envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
        name = f"{value['preparation_id']}-{envelope['envelope_digest'].removeprefix('sha256:')}.json"
        write_launch_preparation_record_exclusive(self.queue / "pending" / name, envelope)
        return envelope, name

    def claim(self, name: str) -> Path:
        claimed = self.queue / "processing" / name
        os.replace(self.queue / "pending" / name, claimed)
        return claimed

    def record_worker_environment(self, worker: dict[str, Any] | None = None, *, image: str = IMAGE,
                                  host: dict[str, Any] | None = None) -> dict[str, Any]:
        """What the allocator's preflight probe records for this image (``remote_cpu_job_allocator``)."""

        worker = HOST_RECORD if worker is None else worker
        host = HOST_RECORD if host is None else host
        record = {"schema_version": "remote_cpu_worker_environment.v1", "stage": "episode_compilation",
                  "job": JOB_SHORT, "image": image, "environment_digest": worker["environment_digest"],
                  "cpu_class": worker["cpu_class"], "host_environment_digest": host["environment_digest"],
                  "parity": {name: worker.get(name) == host.get(name) for name in DIGESTED_FIELDS},
                  "worker_environment": dict(worker), "probe_attempt_id": "rcj-ep-" + "0" * 24 + "-a1-" + "0" * 32,
                  "receipt_digest": "sha256:" + "0" * 64, "recorded_at_epoch": 1.0}
        path = self.jobs / "environment" / "episode_compilation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        path.chmod(0o640)
        return record

    def cache_entry(self, source_digest: str) -> dict[str, Path]:
        """A valid ``particlefield_runtime_asset_cache.v1`` entry for one appearance digest."""

        root = self.cache / source_digest.removeprefix("sha256:")
        root.mkdir(parents=True)
        asset, receipt = root / "scene_appearance.usdc", root / "particlefield_authoring_receipt.v1.json"
        asset.write_bytes(b"PXR-USDC cached particlefield")
        receipt.write_text(json.dumps({"source_sha256": source_digest}) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "particlefield_runtime_asset_cache.v1", "source_configured_appearance_digest": source_digest,
            "particlefield": {"relative_path": asset.name, "digest": digest_of(asset.read_bytes()),
                              "size_bytes": asset.stat().st_size},
            "authoring_receipt": {"relative_path": receipt.name, "digest": digest_of(receipt.read_bytes()),
                                  "size_bytes": receipt.stat().st_size},
            "authoring_implementation": "nvidia_3dgrut_direct_nurec_transcode", "upstream_converter": {},
            "immutable": True, "manifest_digest": "",
        }
        manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
        manifest_path = root / "particlefield_runtime_asset_cache.v1.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        for path in (asset, receipt, manifest_path):
            path.chmod(0o440)
        return {"asset": asset, "receipt": receipt, "manifest": manifest_path}


# ``(policy_observation_override, destination_support, qualification_only)``, as the compiler's own test
# parametrizes its closed compile (``tests/test_task_evaluation_native_arena_episode_compiler.py``).
COMPILE_CASES = [(False, False, False), (True, False, False), (False, True, False), (False, True, True)]


def _documents(value: dict[str, Any], configured: dict[str, Any], *, policy_observation_override: bool,
               destination_support: bool) -> dict[str, dict[str, Any]]:
    """The robot-team and configured-template documents of the compiler's closed-compile test."""

    from tests.test_task_evaluation_native_arena_episode_compiler import _configured_runtime_documents

    robot, controller, task = value["robot"]["identity"], value["controller"]["identity"], value["task"]["identity"]
    success = {"authority": "deterministic_simulator_state", "forbidden_collision_allowed": False,
               "joint_limit_violation_allowed": False, "maximum_final_planar_target_error_m": 0.05,
               "minimum_planar_displacement_m": 0.1, "object_must_remain_on_registered_support": True}
    motion = {"start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
              "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319], "control_frequency_hz": 20,
              "maximum_episode_seconds": 12.0, "maximum_step_count": 240, "resolved_seed": 839873104}
    strategy = "pick_and_place" if destination_support else "planar_push"
    intrinsics = {"fx": 172.0, "fy": 172.0, "cx": 159.5, "cy": 89.5, "width": 320, "height": 180}
    definition = {
        "schema_version": "task_evaluation_rigid_relocation_template.v1",
        "status": "preregistered_candidate_pending_configured_scene_revision", "task_identity": task,
        "object_identity": configured["replacement"]["identity"], "strategy": strategy, **motion,
        "controls_order": ["zero_action", "deterministic_scripted"],
        "failure_metrics": ["insufficient_displacement", "timeout"],
        "preregistration_rule": "Any task change creates a new version.", "success": success,
    }
    if destination_support:
        definition["interaction_affordance"] = {
            "contact_point_scoring_frame_m": [-0.06, 0.0, 0.0], "approach_unit_scoring_frame": [-1.0, 0.0, 0.0],
            "jaw_unit_scoring_frame": [0.0, 1.0, 0.0], "lift_unit_world": [0.0, 0.0, 1.0],
            "pregrasp_clearance_m": 0.12, "minimum_lift_m": 0.08}
    return {
        "robot.configuration": {"schema_version": "task_evaluation_native_robot_configuration.v1", "identity": robot,
                                "joint_reset_positions_rad": {"panda_joint1": 0.0}},
        "robot.kinematics": {"schema_version": "task_evaluation_native_robot_kinematics.v1", "identity": robot},
        "robot.joint_bounds": {"schema_version": "task_evaluation_native_robot_joint_bounds.v1", "identity": robot},
        "robot.base_registration": {
            "schema_version": "task_evaluation_robot_to_scene_registration.v1", "robot_identity": robot,
            "scene_identity": value["scene"]["identity"],
            "robot_mount_interface_digest": configured["registration"]["robot_mount_interface"]["digest"],
            "pose_world": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "controller.configuration": {"schema_version": "task_evaluation_native_controller_configuration.v1",
                                     "identity": controller, "kind": value["controller"]["kind"]},
        "sensors.configuration": {
            "schema_version": "task_evaluation_native_sensor_configuration.v1",
            "scene_camera_calibration_digest": configured["registration"]["camera_calibration"]["digest"],
            "cameras": ([{"role": role, "intrinsics": intrinsics} for role in ("external", "wrist", "overview")]
                        if policy_observation_override else [{"role": "external"}])},
        "scene.configured_revision.task_template.definition": definition,
        "scene.configured_revision.task_template.success_criteria": {
            "schema_version": "task_evaluation_rigid_relocation_success_criteria.v1",
            "status": "preregistered_before_any_episode", **success,
            "target_center_xyz_m": motion["target_center_xyz_m"]},
        "scene.configured_revision.task_template.execution": {
            "schema_version": "task_evaluation_rigid_relocation_execution_spec.v1",
            "status": "preregistered_before_any_episode", "strategy": strategy, **motion,
            "action_bounds_m_per_step": {"minimum": -0.02, "maximum": 0.02},
            "collision_exclusions": ["robot_self_collision_pairs_declared_by_robot_configuration"],
            "termination": ["success", "timeout"]},
        **_configured_runtime_documents(configured),
    }


def _destination(inputs: Path, value: dict[str, Any], configured: dict[str, Any], docs: dict[str, Any],
                 paths: dict[str, Path], references: dict[str, dict[str, Any]], *, qualification_only: bool) -> None:
    """The compiler test's destination case, bound to this revision's subject and support plane."""

    from tests.test_task_evaluation_native_arena_episode_compiler import _destination_case

    collision = inputs / "configured-collision-for-qualification.usda"
    collision.write_bytes(b"#usda 1.0\n# collision\n")
    subject_static = docs["scene.configured_revision.replacement.static_qualification"]
    request_value, destination_references, _ = _destination_case(
        inputs, subject_identity=configured["replacement"]["identity"],
        subject_static_path=paths["scene.configured_revision.replacement.static_qualification"],
        subject_static=subject_static,
        subject_scoring_transform={"position_m": subject_static["observed_structure"]["center_of_mass_m"],
                                   "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        configured_scene_revision_digest=configured["revision_digest"], configured_scene_collision_path=collision,
        configured_scene_support_plane_path=paths["scene.configured_revision.registration.support_plane"])
    destination = request_value["task"]["destination"]
    for field in ("asset", "rights_admission", "static_qualification", "native_import_qualification", "geometry",
                  "placement_qualification"):
        record = destination_references[f"task.destination.{field}"]
        destination[field] = {key: record[key] for key in ("uri", "digest", "size_bytes")}
    value["task"]["destination"] = destination
    for contract_path, record in destination_references.items():
        references.setdefault(contract_path, record)
    if qualification_only:
        value["run_mode"] = "destination_qualification"
        destination.pop("placement_qualification")
        references.pop("task.destination.placement_qualification")
        destination["native_probe"] = {
            "schema_version": "task_evaluation_rigid_destination_native_probe_configuration.v1",
            "placement_support_scene_prim_paths": ["/Root/Support"],
            "qualification_limits": {
                "maximum_penetration_m": 0.001, "minimum_support_contact_force_n": 0.01,
                "maximum_forbidden_contact_force_n": 0.1, "settle_translation_tolerance_m": 0.002,
                "settle_rotation_tolerance_rad": 0.01, "reset_translation_tolerance_m": 0.002,
                "reset_rotation_tolerance_rad": 0.01,
                "minimum_camera_pixels": {"external": 100, "wrist": 100, "overview": 100}},
            "settle_sample_count": 3, "settle_steps_per_sample": 60}


def _policy_observation(inputs: Path, value: dict[str, Any], references: dict[str, dict[str, Any]]) -> None:
    """The compiler test's policy observation override; its mount registry is copied under the input root."""

    from tests.test_task_evaluation_native_arena_episode_compiler import _record, _write_json

    appearance = inputs / "policy-observation.usdc"
    appearance.write_bytes(b"verified policy observation particlefield")
    receipt = {
        "schema_version": "nvidia_3dgrut_particlefield_transcode.v1", "status": "completed",
        "schema": "ParticleField3DGaussianSplat", "output": "/producer/path/not-present-on-control-plane.usdc",
        "output_bytes": appearance.stat().st_size, "output_sha256": digest_of(appearance.read_bytes()),
        "source_sha256": "sha256:" + "a" * 64, "splat_count": 1_000_000, "sh_degree": 3,
        "sh_primvar_element_size": 16, "sh_primvar_interpolation": "constant",
        "display_color_fallback_authored": False, "particlefield_emissive_material_binding_authored": False,
        "particlefield_emissive_material_inputs": None, "particlefield_custom_render_hints_authored": False,
        "particlefield_authoring_implementation": "nvidia_3dgrut_direct_nurec_transcode",
        "upstream_converter": {"repository": "https://github.com/nv-tlabs/3dgrut.git",
                               "source_revision": "a37ef721012dea0f29c0fcfff2d525023b4e854a",
                               "module": "threedgrut.export.scripts.transcode", "module_sha256": "sha256:" + "b" * 64,
                               "source_identity_verified": True},
        "upstream_projection_mode_hint": "perspective", "upstream_sorting_mode_hint": "cameraDistance",
        "upstream_color_space": "srgb_rec709_display",
        "gaussian_field_quality": {"schema_version": "gaussian_field_quality.v1", "status": "qualified",
                                   "blockers": [], "learned_tensors_mutated": False},
        "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    registry = inputs / "franka_robotiq_policy_camera_mount_registry.v1.json"
    registry.write_bytes((Path(__file__).resolve().parents[1] / "docs/arm_decision_proof_v1/manifests"
                          / "franka_robotiq_policy_camera_mount_registry.v1.json").read_bytes())
    prefix = "execution_adapter.policy_observation_setup."
    records = {"appearance_asset": _record(appearance, prefix + "appearance_asset"),
               "appearance_authoring_receipt": _record(_write_json(inputs, "appearance-receipt.json", receipt),
                                                       prefix + "appearance_authoring_receipt"),
               "wrist_camera_mount_registry": _record(registry, prefix + "wrist_camera_mount_registry")}
    value["execution_adapter"]["policy_observation_setup"] = {
        "schema_version": "task_evaluation_policy_observation_setup.v1",
        **{key: {field: record[field] for field in ("uri", "digest", "size_bytes")} for key, record in records.items()},
        "fresh_native_mount_sweep_required": True, "policy_master_resolution_wh": [640, 360],
        "overview_review_resolution_wh": [1280, 720]}
    references.update({record["contract_path"]: record for record in records.values()})


def stage_compile(host: Host, *, policy_observation_override: bool = False, destination_support: bool = False,
                  qualification_only: bool = False, appearance: bytes | None = None,
                  appearance_name: str = "appearance.usda") -> tuple[dict[str, Any], str]:
    """Stage the compiler test's closed compile as one envelope in ``pending/``: every reference is a file
    under the host's input root, as preparation materializes them.  Returns the envelope and its row name."""

    from tests.test_task_evaluation_configured_scene_revision import revision
    from tests.test_task_evaluation_native_arena_episode_compiler import _record, _write_json

    value, configured = request(), revision()
    value["team_namespace"] = configured["team_namespace"]
    value["expected_production_commit"] = configured["source_commit"]
    value["scene"]["identity"] = configured["scene_identity"]
    value["task"]["identity"] = configured["task_template"]["identity"]
    value["task"]["subject"]["identity"] = configured["replacement"]["identity"]
    if destination_support:
        value["task"]["strategy"] = "pick_and_place"
    inputs = host.inputs / value["preparation_id"]
    inputs.mkdir(parents=True)
    bundle = inputs / "configured-scene.zip"
    bundle.write_bytes(configured_bundle(b"#usda 1.0\n# appearance\n" if appearance is None else appearance,
                                         appearance_name=appearance_name))
    docs = _documents(value, configured, policy_observation_override=policy_observation_override,
                      destination_support=destination_support)
    paths = {contract_path: _write_json(inputs, f"input-{index}.json", document)
             for index, (contract_path, document) in enumerate(docs.items())}
    for contract_path, section, field in (
            ("scene.configured_revision.task_template.definition", "task_template", "definition"),
            ("scene.configured_revision.task_template.success_criteria", "task_template", "success_criteria"),
            ("scene.configured_revision.task_template.execution", "task_template", "execution"),
            ("scene.configured_revision.registration.support_plane", "registration", "support_plane"),
            ("scene.configured_revision.replacement.source_object", "replacement", "source_object"),
            ("scene.configured_revision.replacement.static_qualification", "replacement", "static_qualification"),
            ("scene.configured_revision.replacement.native_import_qualification", "replacement",
             "native_import_qualification")):
        record = _record(paths[contract_path], contract_path)
        configured[section][field] = {key: record[key] for key in ("uri", "digest", "size_bytes")}
    bundle_record = _record(bundle, "scene.configured_revision.configured_scene_bundle")
    configured["configured_scene_bundle"] = {key: bundle_record[key] for key in ("uri", "digest", "size_bytes")}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    value["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    runtime = inputs / "runtime-source.zip"
    runtime.write_bytes(runtime_bundle())
    references = {"scene.configured_revision": _record(_write_json(inputs, "revision.json", configured),
                                                       "scene.configured_revision"),
                  "scene.configured_revision.configured_scene_bundle": bundle_record,
                  "execution_adapter.runtime_source_bundle": _record(runtime, "execution_adapter.runtime_source_bundle"),
                  **{contract_path: _record(path, contract_path) for contract_path, path in paths.items()}}
    value["execution_adapter"]["runtime_source_bundle"] = {
        key: references["execution_adapter.runtime_source_bundle"][key] for key in ("uri", "digest", "size_bytes")}
    if destination_support:
        _destination(inputs, value, configured, docs, paths, references, qualification_only=qualification_only)
    if policy_observation_override:
        _policy_observation(inputs, value, references)
    for path in inputs.iterdir():
        path.chmod(0o440)
    rows = [{**record, "full_byte_service_account_readback_passed": True} for record in references.values()]
    envelope: dict[str, Any] = {
        "schema_version": "task_evaluation_episode_compilation_envelope.v1", "compilation_id": value["preparation_id"],
        "preparation_id": value["preparation_id"], "run_id": value["run_id"], "team_namespace": value["team_namespace"],
        "expected_production_commit": value["expected_production_commit"],
        "configured_scene_revision_digest": configured["revision_digest"],
        "configured_scene_bundle": {key: bundle_record[key] for key in ("uri", "digest", "size_bytes")},
        "materialized_references": rows, "request": value, "preparation_result_digest": "sha256:" + "8" * 64,
        "automatic_progression_required": True, "robot_specific_episode_packet_compiled_in_production": True,
        "customer_supplied_prebuilt_episode_packet": False, "production_compiler_owns_episode_packet": True,
        "provider_mutation_performed": False, "paid_execution_requested": False, "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    name = f"{value['preparation_id']}-{envelope['envelope_digest'].removeprefix('sha256:')}.json"
    write_launch_preparation_record_exclusive(host.queue / "pending" / name, envelope)
    return envelope, name
