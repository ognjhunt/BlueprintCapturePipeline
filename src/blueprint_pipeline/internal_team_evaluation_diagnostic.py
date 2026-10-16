"""Owner-operated development fixture for the beta L5 200-episode runtime.

This is an isolated diagnostic, never a website job or a customer recommendation.
It authors real USD geometry and a hold-position HTTPS policy, then uses the
production matrix, bundle, first-shard gate, allocator, scorer and closeout.
Its locally issued single-use credential is explicitly NOT a WebApp KMS lease.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest

SCENE_ID = "internal-team-runtime-diagnostic"
TASK_ID = "authored-cube-planar-push"
CANDIDATE_ID = "blueprint-internal.hold-position-v1"
POLICY_ID = "blueprint-internal-hold-position-v1"
POLICY_DEFINITION = {"schema_version": "blueprint.internal_diagnostic_policy.v1",
                     "algorithm": "repeat_observed_joint_and_gripper_position",
                     "action_chunk_rows": 10, "trained_model": False}
MODEL_DIGEST = canonical_digest(POLICY_DEFINITION)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("internal_diagnostic_record_invalid")
    return value


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as stream:
        path.chmod(0o600)
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def metadata() -> dict[str, Any]:
    from .droid_https_policy_client import INTERFACE_ID, METADATA_SCHEMA
    return {"schema_version": METADATA_SCHEMA, "policy_id": POLICY_ID,
            "policy_interface_id": INTERFACE_ID, "action_space": "joint_position",
            "action_chunk_rows": 10, "model_identity_digest": MODEL_DIGEST}


def hold_position(request: dict[str, Any]) -> dict[str, Any]:
    """Return actions only. No outcome or privileged simulator state is read."""
    from .droid_https_policy_client import INTERFACE_ID, REQUEST_SCHEMA
    if request.get("schema_version") != REQUEST_SCHEMA or request.get("policy_interface_id") != INTERFACE_ID:
        raise ValueError("internal_diagnostic_policy_interface_invalid")
    observation = request.get("observation") or {}
    joints, gripper = observation.get("observation/joint_position"), observation.get("observation/gripper_position")
    if not isinstance(joints, list) or len(joints) != 7 or not isinstance(gripper, list) or len(gripper) != 1:
        raise ValueError("internal_diagnostic_policy_state_invalid")
    state = [float(v) for v in joints + gripper]
    if any(not math.isfinite(v) for v in state):
        raise ValueError("internal_diagnostic_policy_state_invalid")
    return {"actions": [list(state) for _ in range(10)]}


def serve_policy(root: Path, *, port: int) -> None:
    """Ephemeral loopback service; the existing host TLS proxy supplies HTTPS."""
    token_path = root / "policy-bearer"
    if token_path.is_symlink() or token_path.stat().st_mode & 0o077:
        raise ValueError("internal_diagnostic_policy_token_not_private")
    token = token_path.read_text().strip()
    path = urlsplit(_read(root / "diagnostic.json")["policy_url"]).path

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass  # Never log authorization, policy observations or response bodies.

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            return self.path == path and hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self) -> None:
            self._reply(200, metadata()) if self._authorized() else self._reply(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if not self._authorized():
                self._reply(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("request_size")
                result = hold_position(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, KeyError):
                self._reply(400, {"error": "invalid_observation"})
                return
            self._reply(200, result)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def _box(stage: Any, path: str, low: list[float], high: list[float], *, collision: bool) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x, y, z in
                          [(low[0], low[1], low[2]), (high[0], low[1], low[2]),
                           (high[0], high[1], low[2]), (low[0], high[1], low[2]),
                           (low[0], low[1], high[2]), (high[0], low[1], high[2]),
                           (high[0], high[1], high[2]), (low[0], high[1], high[2])]])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
                                    1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.45, 0.55)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


def materialize_fixture(root: Path, *, droid_profile: Path, runtime_source: Path) -> dict[str, Any]:
    """Author fresh geometry. Reuse only robot calibration and the real SDK packet."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from .adp_task_scoring import seal_rigid_task_success_contract
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .team_scene_packet import EXECUTION_AUTHORITY, PENDING_CONTROLS, SCHEMA
    from .task_evaluation_diagnostic_native_arena_compiler import _world_camera

    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    asset_root = root / "authored"
    asset_root.mkdir()
    profile = _read(droid_profile)
    pose = {"position_world_m": [0.5, 0.0, 0.78], "orientation_xyzw": [0., 0., 0., 1.]}
    assets = []
    for role in ("scene_collision", "scene_appearance", "task_object"):
        filename = role + ".usda"
        stage = Usd.Stage.CreateNew(str(asset_root / filename))
        prim = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
        stage.SetDefaultPrim(prim)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        if role == "task_object":
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.1)
            _box(stage, "/Root/Cube", [-0.03] * 3, [0.03] * 3, collision=True)
        else:
            _box(stage, "/Root/Floor", [-2., -2., -0.05], [2., 2., 0.], collision=role == "scene_collision")
            _box(stage, "/Root/Table", [-0.2, -0.6, 0.70], [1.2, 0.6, 0.75], collision=role == "scene_collision")
            if role == "scene_appearance":
                _box(stage, "/Root/Backdrop", [-2., 1.8, 0.], [2., 2., 2.5], collision=False)
        stage.GetRootLayer().Save()
        path = asset_root / filename
        row = {"semantic_role": role, "filename": filename,
               "source": {"root": "evidence", "relative_path": "authored/" + filename,
                          **{k: v for k, v in _record(path).items() if k != "path"}},
               "pose_world": pose if role == "task_object" else
                             {"position_world_m": [0., 0., 0.], "orientation_xyzw": [0., 0., 0., 1.]}}
        if role == "task_object":
            row.update(asset_id="diagnostic_cube", object_type="RIGID",
                       reset_state={"root_pose_world": pose, "joint_positions": {}})
        assets.append(row)
    spec = {"schema_version": "adp_task_spec.v2", "task_kind": "rigid_pick_place",
            "manipulation_strategy": "planar_push", "subject_asset_id": "diagnostic_cube",
            "source_subject_identity": "authored-diagnostic-cube", "instruction_subject_label": "blue cube",
            "visible_target_label": "green marker", "prompt": "Push the blue cube onto the green marker.",
            "start_pose_world": [0.5, 0., 0.78, 0., 0., 0., 1.], "require_sealed_start_pose": True,
            "target_position_world_m": [0.5, 0.2, 0.78], "destination_position_world_m": [0.5, 0.2, 0.78],
            "destination_position_bounds_world_m": {"minimum": [0.46, 0.16, 0.765], "maximum": [0.54, 0.24, 0.795]},
            "destination_orientation_xyzw": [0., 0., 0., 1.], "destination_orientation_tolerance_rad": 0.2,
            "subject_collision_bounds_scoring_frame_m": {"minimum": [-0.03] * 3, "maximum": [0.03] * 3},
            "support_plane_z_m": 0.75, "support_height_interval_m": [0.765, 0.795],
            "movement_epsilon_m": 0.005, "settle_position_tolerance_m": 0.005,
            "settle_orientation_tolerance_rad": 0.03,
            "reset_translation_tolerance_m": 0.002, "reset_orientation_tolerance_rad": 0.01,
            "release_gripper_width_min_m": 0.06, "task_contact_minimum_force_n": 0.5, "release_required": True,
            "collision_failure_minimum_force_n": 1.,
            "workspace_position_bounds_world_m": {"minimum": [-0.2, -0.6, 0.70],
                                                   "maximum": [1.2, 0.6, 1.4]},
            "minimum_translation_m": 0.1, "minimum_lift_m": 0., "control_frequency_hz": 15,
            "maximum_action_steps": 30, "settle_window_samples": 4}
    from .native_task_episode_environment import NativeRigidScoringEnvironment

    NativeRigidScoringEnvironment.validate_task_spec(spec)
    affordance = {"schema_version": "native_rigid_interaction_affordance.v1",
                  "subject_asset_id": "diagnostic_cube", "scoring_frame_id": "task_scoring_frame",
                  "asset_root_from_scoring_frame": {"position_m": [0., 0., 0.], "orientation_xyzw": [0., 0., 0., 1.]},
                  "contact_point_scoring_frame_m": [0., -0.03, 0.], "approach_unit_scoring_frame": [0., -1., 0.],
                  "lift_unit_world": [0., 0., 1.], "gripper_orientation_scoring_frame_xyzw": [0., 0., 0., 1.],
                  "pregrasp_clearance_m": 0.08, "arrival_orientation_tolerance_rad": 0.1,
                  "allowed_contact_prim_paths": ["/Root"], "intended_support_prim_paths": ["/Root/Table"],
                  "affordance_digest": ""}
    affordance["affordance_digest"] = canonical_digest(affordance, digest_field="affordance_digest")
    spec["interaction_affordance"] = affordance
    contract = seal_rigid_task_success_contract(task_spec=spec, site_id=SCENE_ID, task_id=TASK_ID,
        author_source="compatibility_default", author_id="blueprint:authored_internal_diagnostic",
        confirmation_status="confirmed")
    spec.update(task_success_contract=contract, task_success_contract_digest=contract["contract_digest"])
    intrinsics = {"fx": 300., "fy": 300., "cx": 159.5, "cy": 89.5, "width": 320, "height": 180}
    wrist = next(row for row in profile["cameras"] if row["role"] == "wrist")
    cameras = [_world_camera("external", position=[1.2, -0.8, 1.4], target=[0.5, 0., 0.78], intrinsics=intrinsics),
               wrist, _world_camera("overview", position=[0.95, 0.75, 1.45], target=[0.5, 0.1, 0.78], intrinsics=intrinsics)]
    cameras[-1]["policy_input"] = False
    context = {"schema_version": "adp009d_scenario_instance.v1", "program_id": "internal-beta-diagnostic",
               "cell_id": "diagnostic.nominal", "seed": 0, "policy_neutral": True,
               "caller_asserted_success": False, "learned_policy_outcomes_consulted": False, "instance_digest": ""}
    context["instance_digest"] = canonical_digest(context, digest_field="instance_digest")
    request = {"schema_version": "native_task_arena_packet_request.v1", "scene_id": SCENE_ID, "task_id": TASK_ID,
               "task_spec": spec, "task_joint_bindings": [], "task_state_binding": None, "assets": assets,
               "robot_base_pose_world": {"position_world_m": [0., 0., 0.75], "orientation_xyzw": [0., 0., 0., 1.]},
               "robot_joint_reset_positions_rad": profile["robot_joint_reset_positions_rad"],
               "cameras": cameras, "physics_frequency_hz": 120, "request_digest": "",
               "scenario": {"context_kind": "evaluation_cell", "cell_id": "diagnostic.nominal", "seed": 0,
                            "instance_digest": context["instance_digest"], "context_document": context}}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    packet = root / "native-task-packet"
    materialize_native_task_arena_packet(request=request, evidence_root=root, output_dir=packet)
    plan = _read(packet / "native_task_arena_scene_plan.v1.json")
    receipt = _record(packet / "native_task_arena_packet_receipt.v1.json")
    value = {"schema_version": SCHEMA, "claim_ceiling": "diagnostic_policy_execution",
             "scene_revision_digest": canonical_digest({"authored_fixture": request}),
             "configuration_digest": request["request_digest"], "plan_digest": plan["plan_digest"],
             "task_success_contract": contract, "task_success_contract_digest": contract["contract_digest"],
             "base_native_packet": receipt, "runtime_source": _record(runtime_source), "construction_result": receipt,
             "execution_authority": EXECUTION_AUTHORITY, "control_diagnostic": PENDING_CONTROLS,
             "robot_calibration_source": _record(droid_profile),
             "evidence_scope": "development_only", "customer_capture_used": False, "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    _write(root / "team_scene_inputs.v1.json", value)
    return value


def internal_admission(*, facts: dict[str, Any], approval: dict[str, Any], policy_url: str,
                       credential_digest: str) -> dict[str, Any]:
    """An owner-issued diagnostic admission, with no customer or KMS assertion.

    The customer admission entrypoint remains unchanged. This operator-only
    fixture is restricted to the authored scene and the source-defined policy.
    """
    from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
    from .native_task_arena_policy_canary_session import TEAM_ADMISSION_SCHEMA_VERSION, TEAM_RUN_KIND
    from .team_candidate_quick10_admission import validate_admission
    from .team_evaluation_plan import campaign_ttl_seconds

    expires = datetime.fromisoformat(str(approval.get("expires_at_iso")).replace("Z", "+00:00"))
    if (approval.get("schema_version") != "blueprint.internal_team_diagnostic_owner_approval.v1"
            or approval.get("approved_by") != "@ognjhunt" or approval.get("episodes") != 200
            or approval.get("cap_usd") != 20 or approval.get("max_attempts") != 1
            or approval.get("scene_id") != SCENE_ID or approval.get("candidate_id") != CANDIDATE_ID
            or approval.get("customer_recommendation_permitted") is not False
            or not approval.get("approval_reference") or expires.tzinfo is None
            or expires <= datetime.now(timezone.utc)):
        raise ValueError("internal_diagnostic_owner_approval_invalid")
    parsed = urlsplit(policy_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("internal_diagnostic_https_endpoint_invalid")
    if _read(Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json")["scene_id"] != SCENE_ID:
        raise ValueError("internal_diagnostic_authored_scene_required")
    contract = facts["scene"]["task_success_contract"]
    if contract["scope"] != {"site_id": SCENE_ID, "task_id": TASK_ID}:
        raise ValueError("internal_diagnostic_authored_contract_required")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    value = {"schema_version": TEAM_ADMISSION_SCHEMA_VERSION, "status": "admitted", "run_kind": TEAM_RUN_KIND,
             "runtime": "team_evaluation", "evaluation_purpose": "scored", "episodes": 200,
             "team_id": "blueprint-internal", "checkpoint_id": "hold-position-v1", "candidate_id": CANDIDATE_ID,
             "policy_interface_id": metadata()["policy_interface_id"], "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID,
             "embodiment": None,
             "delivery": {"mode": "https_policy", "endpoint_url": policy_url, "metadata_url": policy_url,
                          "action_chunk_rows": 10, "open_loop_horizon": 10,
                          "model_identity_digest": MODEL_DIGEST, "team_policy_id": POLICY_ID},
             "credential": {"lease_id": "internal-diagnostic-" + credential_digest[7:39],
                            "credential_ref_digest": credential_digest, "credential_env": "BLUEPRINT_TEAM_POLICY_BEARER_FILE",
                            "origin": origin, "authority": "owner_issued_local_single_run", "kms_lease_claimed": False},
             "scene_revision_digest": facts["scene"]["scene_revision_digest"],
             "task_spec_digest": canonical_digest(facts["scene"]["task_spec"]),
             "task_success_contract_digest": contract["contract_digest"],
             "robot_free_probe_receipt_digest": facts["probe"]["receipt_digest"],
             "rights_digest": canonical_digest({"author": "Blueprint", "policy": POLICY_DEFINITION,
                                                "permission": "owner_authorized_internal_execution_only"}),
             "founder_approval_digest": canonical_digest(approval),
             "execution_deadline_at_iso": approval["expires_at_iso"],
             "budget": {"cap_usd": 20., "hard_ttl_seconds": campaign_ttl_seconds(), "max_attempts": 1, "episodes": 200},
             "claim_ceiling": "diagnostic_policy_execution", "evidence_scope": "development_only",
             "adp_qualified_evidence_eligible": False, "physical_performance_claimed": False,
             "customer_recommendation_permitted": False, "internal_diagnostic": True, "trained_model": False,
             "admission_digest": ""}
    value["admission_digest"] = canonical_digest(value, digest_field="admission_digest")
    return validate_admission(value)


def stage(root: Path, *, policy_url: str, owner_approval: Path, source_commit: str,
          robot_free_probe_root: Path | None = None) -> dict[str, Any]:
    """Seal one 200-cell session before its sole GPU allocation. Does not launch."""
    from .controlled_policy_configuration import canonical_request_digest
    from . import team_candidate_quick10_launch as launch
    from .team_candidate_quick10_bundle import assert_secret_absent
    from .team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA, offer_eligibility

    if launch._source_commit() != source_commit:
        raise ValueError("internal_diagnostic_source_commit_mismatch")
    job = root / "evaluation"
    capture = root / "capture"
    scene_inputs = _read(root / "scene" / "team_scene_inputs.v1.json")
    profile = {"capture_root": str(capture.resolve()), "scene_id": SCENE_ID, "task_id": TASK_ID, "scenario_id": "nominal",
               "scene_revision_digest": scene_inputs["scene_revision_digest"],
               "team_scene_inputs_path": str((root / "scene" / "team_scene_inputs.v1.json").resolve()),
               "robot_free_probe_root": str((robot_free_probe_root or root / "probe").resolve()), "open_loop_horizon": 10,
               "max_hourly_rate_usd": 0.8, "nominal_dynamic_friction": 0.5, "scored_matrix_salt": secrets.token_hex(32)}
    facts, blockers = offer_eligibility(profile)
    if facts is None:
        raise ValueError(";".join(blockers))
    approval = _read(owner_approval)
    token = secrets.token_urlsafe(48)
    token_digest = canonical_digest({"credential_sha256": hashlib.sha256(token.encode()).hexdigest()})
    admission = internal_admission(facts=facts, approval=approval, policy_url=policy_url, credential_digest=token_digest)
    job.mkdir(mode=0o700)
    capture.mkdir()
    _write(capture / "capture_descriptor.json", {"development_only": True, "source": "authored_internal_diagnostic"})
    registry = {"schema_version": REGISTRY_SCHEMA, "profiles": [profile], "registry_digest": ""}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path = _write(root / "registry.json", registry)
    os.environ[REGISTRY_ENV] = str(registry_path.resolve())
    # Real HTTPS metadata is checked separately before any paid launch, once the
    # policy service is running. The private credential is issued exactly once.
    with (root / "policy-bearer").open("x") as stream:
        (root / "policy-bearer").chmod(0o600)
        stream.write(token + "\n")
    request = {"job_id": "internal-team-diagnostic-" + source_commit[:12], "capture_root": str(capture.resolve()),
               "requested_tasks": [{"task_id": TASK_ID, "scenario_ids": ["nominal"]}],
               "internal_diagnostic": True, "customer_recommendation_permitted": False}
    _write(job / "team_candidate_admission.json", admission)
    _write(job / launch.REQUEST_FILENAME, request)
    cells = launch._session_cells(purpose="scored", facts=facts, profile=profile)
    groups = [cells]
    base_run_id = "internal-team-diag-" + source_commit[:12]
    for index, group in enumerate(groups):
        shard = launch._stage_shard(job_dir=job, index=index, cells=group, admission=admission, facts=facts,
            profile=profile, cap_usd=20., base_run_id=base_run_id, commit=source_commit)
        assert_secret_absent(shard["bundle_path"], token)
    zero = launch._default_provider_zero()
    _write(job / "initial_provider_zero.json", zero)
    if zero.get("provider_zero_verified") is not True:
        raise ValueError("internal_diagnostic_initial_provider_zero_unproven")
    private = job / ".team-credential"
    private.mkdir(mode=0o700)
    with (private / "bearer").open("x") as stream:
        (private / "bearer").chmod(0o600)
        stream.write(token + "\n")
    _write(job / launch.INTENT_FILENAME, {"job_id": request["job_id"], "canonical_request_digest": canonical_request_digest(request),
        "pipeline_run_id": base_run_id, "admission_digest": admission["admission_digest"], "run_id": base_run_id,
        "evaluation_purpose": "scored", "episodes": len(cells), "shard_count": len(groups), "source_commit": source_commit})
    expires = datetime.fromisoformat(approval["expires_at_iso"].replace("Z", "+00:00")).timestamp() * 1000
    _write(job / launch.CONTROL_FILENAME, {"run_id": base_run_id, "internal_diagnostic": True,
        "execution_deadline_at_ms": min(expires, datetime.now(timezone.utc).timestamp() * 1000 + 16 * 60 * 60 * 1000),
        "owner_approval": _record(owner_approval)})
    result = {"schema_version": "blueprint.internal_team_diagnostic.v1", "policy_url": policy_url,
              "source_commit": source_commit, "job_dir": str(job.resolve()), "registry_path": str(registry_path.resolve()),
              "owner_approval": _record(owner_approval), "episodes": len(cells), "cap_usd": 20.,
              "evidence_scope": "development_only", "customer_recommendation_permitted": False}
    _write(root / "diagnostic.json", result)
    return result


def run(root: Path) -> dict[str, Any] | None:
    from . import team_candidate_quick10_launch as launch
    from .team_candidate_quick10_registry import REGISTRY_ENV

    diagnostic = _read(root / "diagnostic.json")
    if launch._source_commit() != diagnostic["source_commit"]:
        raise ValueError("internal_diagnostic_source_commit_mismatch")
    terminal_path = Path(diagnostic["job_dir"]) / launch.TERMINAL_FILENAME
    if terminal_path.is_file():
        return _read(terminal_path)
    os.environ[REGISTRY_ENV] = diagnostic["registry_path"]
    approval_record = diagnostic["owner_approval"]
    approval_path = Path(approval_record["path"])
    if _record(approval_path) != approval_record:
        raise ValueError("internal_diagnostic_owner_approval_changed")
    from .team_candidate_quick10_registry import offer_eligibility, profile_for_request
    request = _read(Path(diagnostic["job_dir"]) / launch.REQUEST_FILENAME)
    facts, blockers = offer_eligibility(profile_for_request(request))
    if facts is None:
        raise ValueError(";".join(blockers))
    admitted = _read(Path(diagnostic["job_dir"]) / "team_candidate_admission.json")
    renewed = internal_admission(facts=facts, approval=_read(approval_path), policy_url=diagnostic["policy_url"],
                                 credential_digest=admitted["credential"]["credential_ref_digest"])
    if renewed["admission_digest"] != admitted["admission_digest"]:
        raise ValueError("internal_diagnostic_admission_changed")
    token = (root / "policy-bearer").read_text().strip()
    parsed = urlsplit(diagnostic["policy_url"])
    observed = launch._team_metadata(diagnostic["policy_url"], f"{parsed.scheme}://{parsed.netloc}", token, None)
    if observed != metadata():
        raise ValueError("internal_diagnostic_policy_identity_mismatch")
    # The production queue runner normally owns this lock. This isolated
    # operator entrypoint must enforce the same single-writer rule itself.
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal = launch.run_staged_evaluation(job_dir=Path(diagnostic["job_dir"]),
            hooks={"execution_monitor": _diagnostic_execution_monitor})
    if terminal is not None:
        path = root / "internal_diagnostic_result.json"
        if not path.exists():
            _write(path, {"evidence_scope": "development_only", "trained_policy_evaluated": False,
                          "customer_recommendation_permitted": False, "terminal": terminal})
    return terminal


def _diagnostic_execution_monitor(control: dict[str, Any]) -> dict[str, Any]:
    """Internal authored diagnostics have owner authority, never a fabricated WebApp claim."""
    record = control.get("owner_approval") or {}
    path = Path(str(record.get("path") or ""))
    if control.get("internal_diagnostic") is not True or _record(path) != record:
        return {"cancellation_requested": True}
    approval = _read(path)
    expires = datetime.fromisoformat(str(approval.get("expires_at_iso")).replace("Z", "+00:00"))
    if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
        return {"cancellation_requested": True}
    return {"ok": True, "run_id": control["run_id"], "execution_deadline_at_ms": control["execution_deadline_at_ms"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--root", type=Path, required=True)
    fixture.add_argument("--droid-profile", type=Path, required=True)
    fixture.add_argument("--runtime-source-receipt", type=Path, required=True)
    serve = sub.add_parser("serve-policy")
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--port", type=int, default=18943)
    prepare = sub.add_parser("stage")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--policy-url", required=True)
    prepare.add_argument("--owner-approval", type=Path, required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--robot-free-probe-root", type=Path,
                         help="Directory retaining a real passed probe at this exact scene revision.")
    execute = sub.add_parser("run")
    execute.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fixture":
        value = materialize_fixture(args.root, droid_profile=args.droid_profile, runtime_source=args.runtime_source_receipt)
        print(json.dumps({"scene_revision_digest": value["scene_revision_digest"], "evidence_scope": "development_only"}))
    elif args.command == "serve-policy":
        serve_policy(args.root, port=args.port)
    elif args.command == "stage":
        print(json.dumps(stage(args.root, policy_url=args.policy_url, owner_approval=args.owner_approval,
                               source_commit=args.source_commit, robot_free_probe_root=args.robot_free_probe_root)))
    else:
        result = run(args.root)
        print(json.dumps(result))
        return 0 if result is not None and result.get("status") == "completed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
