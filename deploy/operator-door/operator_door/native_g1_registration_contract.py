"""Networkless, stdlib-only reconstruction of site-bound G1 receipt registrations.

Registration grants no execution authority. The planner reopens live task/consent
and composes the reviewed model terms immediately before planning and launch.
Caller inputs are identities and digests, never filesystem paths or policy code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

if __package__:
    from . import public_policy_registration_contract as common
else:
    _spec = importlib.util.spec_from_file_location("_g1_common", Path(__file__).with_name("public_policy_registration_contract.py"))
    assert _spec and _spec.loader
    common = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(common)

SCHEMA = "native_g1_policy_install_request.v1"
OPERATION = "install-native-g1-policies"
PROPOSAL_SCHEMA = "native_g1_policy_install_proposal.v1"
RECEIPT_SCHEMA = "native_g1_policy_install_receipt.v1"
CANDIDATES = ("humanoidarena_dp_g1_dex3_sonic", "humanoidarena_pi05_g1_dex3_sonic",
              "humanoidarena_dp_g1_dex3_sonic_vision_navi", "humanoidarena_pi05_g1_dex3_sonic_vision_navi")
SHARED_REGISTRY = Path("/var/lib/blueprint/task-evaluation-inputs/g1-team-campaign-registry.json")
SOURCE_QUEUE = Path("/var/lib/blueprint/pipeline-control-plane/team-scene-packets")
SOURCE_FIELDS = ("scene_intent_id", "scene_intent_digest", "capture_id", "task_id", "task_context_digest", "scene_revision_digest")
ROBOT_TASK_FIELDS = {"robot_workspace_position_bounds_world_m", "release_gripper_width_min_m",
    "action_bounds_m_per_step", "control_frequency_hz", "maximum_action_steps", "maximum_episode_seconds", "retreat_clearance_m"}
FIELDS = {"schema_version", "installation_id", "release_commit", "site_id", "source", "candidates",
          "publisher_source", "runtime_source", "shared_registry_digest", "expected_registry_digest",
          "authorization_reference", "request_digest"}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError("native_g1_install_" + reason)


def identity(value):
    require(isinstance(value, dict) and set(value) == {"sha256", "size_bytes"}
        and isinstance(value["sha256"], str) and DIGEST.fullmatch(value["sha256"])
        and type(value["size_bytes"]) is int and 0 < value["size_bytes"] <= 8 * 1024**2, "reference_invalid")
    return value


def validate_request(value):
    require(isinstance(value, dict) and set(value) == FIELDS and value["schema_version"] == SCHEMA, "request_invalid")
    require(isinstance(value["installation_id"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value["installation_id"]), "installation_id_invalid")
    require(isinstance(value["release_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", value["release_commit"]), "release_invalid")
    require(isinstance(value["site_id"], str) and re.fullmatch(r"site-[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", value["site_id"]), "site_invalid")
    source = value["source"]
    require(isinstance(source, dict) and set(source) == set(SOURCE_FIELDS), "source_invalid")
    for key, item in source.items():
        require(isinstance(item, str) and (DIGEST.fullmatch(item) if key.endswith("digest") else
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}", item)), "source_invalid")
    for field in ("shared_registry_digest", "expected_registry_digest"):
        require(isinstance(value[field], str) and DIGEST.fullmatch(value[field]), "registry_digest_invalid")
    for field in ("publisher_source", "runtime_source"):
        if value[field] is not None:
            identity(value[field])
    rows = value["candidates"]
    require(isinstance(rows, list) and 1 <= len(rows) <= 4, "candidates_invalid")
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"candidate_id", "scene_packet", "navigation_authority"}
            and isinstance(row["candidate_id"], str) and row["candidate_id"] in CANDIDATES
            and row["candidate_id"] not in seen, "candidate_invalid")
        seen.add(row["candidate_id"])
        for field in ("scene_packet", "navigation_authority"):
            if row[field] is not None:
                identity(row[field])
        require(row["candidate_id"].endswith("_vision_navi") or row["navigation_authority"] is None, "navigation_candidate_invalid")
    reference = value["authorization_reference"]
    require(isinstance(reference, str) and 1 <= len(reference) <= 300 and reference.isprintable(), "authorization_reference_invalid")
    require(value["request_digest"] == common.digest(value, "request_digest"), "request_digest_invalid")
    return json.loads(common.canonical(value))


def sealed(path, schema, field):
    value = common.read(path)
    require(value.get("schema_version") == schema and value.get(field) == common.digest(value, field), "retained_record_changed")
    return value


def file_record(path):
    path = Path(path)
    require(".." not in path.parts, "bound_path_invalid")
    path = common.safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= 32 * 1024**3
            and not before.st_mode & 0o022, "bound_file_invalid")
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(block)
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "bound_file_changed_during_read")
    return {"path": str(path), "sha256": "sha256:" + hasher.hexdigest(), "size_bytes": before.st_size}


def bound(path, reference):
    require(set(reference) == {"sha256", "size_bytes"} and type(reference["size_bytes"]) is int
        and 0 <= reference["size_bytes"] <= 32 * 1024**3 and DIGEST.fullmatch(reference["sha256"]), "reference_invalid")
    record = file_record(path)
    require(all(record[key] == reference[key] for key in reference), "retained_file_changed")
    return record


def protected(path, *, service_owned=False):
    path = common.safe_path(Path(path))
    info = path.stat()
    owners = {0, os.geteuid()}
    if service_owned:
        try:
            owners.add(pwd.getpwnam("blueprint").pw_uid)
        except KeyError:
            pass
    require(stat.S_ISREG(info.st_mode) and info.st_uid in owners and not info.st_mode & 0o022, "protected_record_invalid")
    return path


def _child(root, relative):
    pure = PurePosixPath(str(relative))
    require(bool(pure.parts) and not pure.is_absolute() and ".." not in pure.parts and str(pure) == relative, "retained_member_path_invalid")
    return common.safe_path(root / relative)


def discover(root, reference, *, names):
    """Only one site's existing scene directory; never global filesystem search."""
    identity(reference)
    root = common.safe_path(root)
    require(root.is_dir(), "current_scene_packet_missing")
    matches, visited = [], 0
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in {".git", "__pycache__"})
        for name in directories:
            require(not (Path(directory) / name).is_symlink(), "retained_tree_symlink")
        visited += len(files) + len(directories)
        require(visited <= 20_000, "retained_lookup_limit")
        for name in sorted(set(files) & set(names)):
            path = common.safe_path(Path(directory) / name)
            if path.stat().st_size == reference["size_bytes"]:
                record = file_record(path)
                if record["sha256"] == reference["sha256"]:
                    matches.append(record)
    require(len(matches) == 1, "current_scene_packet_missing" if not matches else "retained_reference_ambiguous")
    return matches[0]


def site_inputs(request, registry, *, source_queue=SOURCE_QUEUE):
    source, site = request["source"], request["site_id"]
    profiles = [row for row in registry["profiles"] if row.get("scene_id") == site]
    settings = registry.get("derived_profiles") or {}
    if not profiles:
        root = common.safe_path(Path(settings["root"]))
        paths = sorted(root.glob("*.json"))
        require(len(paths) <= 10_000, "profile_lookup_limit")
        for path in paths:
            row = common.read(protected(path, service_owned=True))
            if row.get("scene_id") == site:
                require(row.get("schema_version") == "blueprint.team_candidate_derived_profile.v1"
                    and row.get("profile_digest") == common.digest(row, "profile_digest"), "derived_profile_changed")
                profiles.append(row)
    require(len(profiles) == 1, "current_site_profile_missing_or_ambiguous")
    profile = profiles[0]
    require(profile.get("task_id") == source["task_id"] and profile.get("scene_revision_digest") == source["scene_revision_digest"], "current_site_changed")
    context = sealed(common.safe_path(Path(profile["capture_root"])) / "pipeline/website_task_context.json", "website_site_task_context.v1", "context_digest")
    consent = context.get("capture_rights") or {}
    require(context.get("scene_id") == site and context.get("capture_id") == source["capture_id"]
        and context.get("context_digest") == source["task_context_digest"]
        and consent.get("consent_status") == "granted" and consent.get("consent_revoked") is False
        and "robot_evaluation" in consent.get("consent_scope", [])
        and consent.get("derived_scene_generation_allowed") is True, "retained_task_consent_invalid")
    # A digest alone is not a new intent. Reopen the existing completed source
    # queue record used by the beta worker, never a caller-written authority.
    paths = sorted((common.safe_path(source_queue) / "processed").glob("*.json"))
    require(len(paths) <= 10_000, "source_lookup_limit")
    matches = []
    for path in paths:
        row = common.read(path)
        if row.get("intent_id") == source["scene_intent_id"] and row.get("capture_id") == source["capture_id"]:
            require(row.get("schema_version") == "blueprint.team_scene_packet_request.v1"
                and row.get("request_digest") == common.digest(row, "request_digest")
                and row.get("intent_digest") == source["scene_intent_digest"]
                and row.get("task_id") == source["task_id"]
                and row.get("scene_revision_digest") == source["scene_revision_digest"], "current_intent_changed")
            matches.append(row)
    require(len(matches) == 1, "current_intent_missing_or_ambiguous")
    inputs_path = common.safe_path(Path(profile["team_scene_inputs_path"]))
    output_root = common.safe_path(Path(settings["packet_inputs"]["output_root"]))
    require(inputs_path.is_relative_to(output_root) and inputs_path.parent != output_root, "current_site_packet_root_invalid")
    inputs = sealed(inputs_path, "blueprint.team_scene_inputs.v1", "inputs_digest")
    require(inputs.get("scene_revision_digest") == source["scene_revision_digest"], "current_scene_revision_changed")
    base = inputs["base_native_packet"]
    bound(base["path"], {key: base[key] for key in ("sha256", "size_bytes")})
    require(Path(base["path"]).is_relative_to(inputs_path.parent), "current_base_packet_root_invalid")
    base_receipt, base_request, base_scene = packet(Path(base["path"]))
    require(base_scene.get("scene_id") == site and base_scene.get("task_id") == source["task_id"], "current_base_scene_changed")
    return inputs_path.parent, inputs, base_receipt, base_request, base_scene


def packet(path):
    root = path.parent
    receipt = sealed(path, "native_task_arena_packet_receipt.v1", "receipt_digest")
    artifacts = receipt.get("artifacts", [])
    require(isinstance(artifacts, list) and len(artifacts) == 3
        and {row.get("relative_path") for row in artifacts} == {
            "native_task_arena_packet_request.v1.json", "native_task_runtime_contract.v1.json",
            "native_task_arena_scene_plan.v1.json"}, "packet_artifact_inventory_changed")
    for row in artifacts:
        bound(_child(root, row["relative_path"]), {key: row[key] for key in ("sha256", "size_bytes")})
    for row in receipt.get("source_bindings", []):
        bound(_child(root, row["staged_relative_path"]), {"sha256": row["staged_sha256"], "size_bytes": row["staged_size_bytes"]})
    request = sealed(root / "native_task_arena_packet_request.v1.json", "native_task_arena_packet_request.v1", "request_digest")
    require(receipt.get("request_digest") == request["request_digest"], "packet_request_changed")
    scene = sealed(root / "native_task_arena_scene_plan.v1.json", "native_task_arena_scene_plan.v1", "plan_digest")
    require(receipt.get("arena_scene_plan_digest") == scene["plan_digest"], "scene_receipt_changed")
    return receipt, request, scene


def publisher(record, profile):
    receipt = sealed(Path(record["path"]), "native_g1_publisher_source_stage.v1", "receipt_digest")
    root = common.safe_path(Path(record["path"]).parent / "source")
    expected = {"status": "ready", "source_repository": "https://github.com/William-wAng618/HumanoidArena",
        "source_revision": profile["source_revision"], "source_root": str(root),
        "inventory_file_sha256": profile["inventory_sha256"], "model_weights_included": False, "gpu_allocated": False}
    require(all(receipt.get(k) == v for k, v in expected.items()), "publisher_identity_changed")
    expected_paths = {row["path"] for row in profile["publisher_files"]}
    observed_paths = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        if Path(directory) == root:
            directories[:] = [name for name in directories if name != ".git"]
        for name in directories:
            require(not (Path(directory) / name).is_symlink(), "publisher_source_symlink")
        for name in files:
            relative = (Path(directory) / name).relative_to(root)
            # Historical bytecode is not source and the beta bundle copies only
            # pinned tracked files. Never ignore arbitrary Python or libraries.
            if "__pycache__" in relative.parts and relative.suffix == ".pyc":
                common.safe_path(root / relative)
                continue
            observed_paths.add(str(relative))
    require(observed_paths == expected_paths, "publisher_source_inventory_changed")
    for row in profile["publisher_files"]:
        path = _child(root, row["path"])
        actual = file_record(path)
        require(actual["size_bytes"] == row["size_bytes"], "publisher_source_changed")
        hasher = hashlib.sha1(b"blob " + str(row["size_bytes"]).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                hasher.update(block)
        require(hasher.hexdigest() == row["git_blob_sha1"], "publisher_source_changed")


def runtime(record, profile):
    receipt = sealed(Path(record["path"]), "native_task_runtime_source_packet.v1", "receipt_digest")
    require(receipt.get("runtime_profile") == "unitree_g1" and receipt.get("redistribution_permitted") is True
        and receipt.get("policy_bytes_included") is False, "runtime_profile_changed")
    path = common.safe_path(Path(receipt["packet_path"]))
    # The protected source registry chose this receipt. Its archive must be a
    # sibling, rather than a receipt-selected arbitrary filesystem object.
    require(path.parent == Path(record["path"]).parent and path.name == "native_task_runtime_sources.zip", "runtime_archive_path_invalid")
    file_identity = file_record(path)
    require(file_identity["sha256"] == receipt["packet_sha256"] and file_identity["size_bytes"] == receipt["packet_size_bytes"], "runtime_archive_changed")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and len(names) <= 50_000, "runtime_archive_members_invalid")
        manifest = common.decode(archive.read("native_task_runtime_source_manifest.v1.json"))
        require(manifest.get("manifest_digest") == common.digest(manifest, "manifest_digest") == receipt["manifest_digest"]
            and manifest.get("runtime_profile") == "unitree_g1" and manifest.get("redistribution_permitted") is True
            and manifest.get("policy_bytes_included") is False, "runtime_manifest_changed")
        wheels = manifest["runtime_dependency_wheels"]
        require(wheels == receipt["runtime_dependency_wheels"], "runtime_wheels_changed")
        pinned = {row["filename"]: row for row in profile["runtime_wheels"]}
        require({row["filename"] for row in wheels} == set(pinned), "runtime_wheel_inventory_changed")
        for row in wheels:
            require(all(row.get(k) == v for k, v in pinned[row["filename"]].items()), "runtime_wheel_identity_changed")
        repositories = manifest["repositories"]
        require({(row["repository"], row["commit"], row["tree"]) for row in repositories} == {
            ("https://github.com/isaac-sim/IsaacLab.git", "ffff603eafc6b74264a5261cc0183d6a65390d78", "2f82f1afb2cfaf6816b328e03c7b3ddc12069658"),
            ("https://github.com/isaac-sim/IsaacLab-Arena.git", "8b82dca224f2b5af08f339f987613c59ce9cdbaa", "a52514015a8573ac03b6448688bfa61f9cea18a9")}, "runtime_source_revision_changed")
        rows = [*wheels, *(file for repo in repositories for file in repo["files"])]
        require(set(names) == {"native_task_runtime_source_manifest.v1.json", *(row["archive_path"] for row in rows)}, "runtime_archive_members_invalid")
        for row in rows:
            name = row["archive_path"]
            _child(path.parent, name)
            require(archive.getinfo(name).file_size == row["size_bytes"], "runtime_member_changed")
            hasher = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    hasher.update(block)
            require("sha256:" + hasher.hexdigest() == row["sha256"], "runtime_member_changed")


def verify_scene(record, *, candidate_id, profile, source, site_id, inputs, base_receipt, base_request, base_scene):
    receipt, request, scene = packet(Path(record["path"]))
    derivation = request.get("g1_scene_derivation") or {}
    require(derivation.get("source_packet_receipt_digest") == base_receipt["receipt_digest"]
        and derivation.get("source_scene_plan_digest") == base_receipt["arena_scene_plan_digest"]
        and all(request.get(key) == base_request.get(key) for key in
            ("scene_id", "task_id", "task_joint_bindings", "task_state_binding", "appearance_variant"))
        and [{k: v for k, v in row.items() if k != "source"} for row in request["assets"]]
            == [{k: v for k, v in row.items() if k != "source"} for row in base_request["assets"]], "current_scene_derivation_changed")
    def assets(row):
        return sorted((f["semantic_role"], f["staged_sha256"], f["staged_size_bytes"]) for f in row["source_bindings"])
    require(assets(receipt) == assets(base_receipt), "current_scene_assets_changed")
    task = scene["task_spec"]
    original = {**base_scene["task_spec"], "task_success_contract": inputs["task_success_contract"],
        "task_success_contract_digest": inputs["task_success_contract_digest"]}
    require(scene.get("scene_id") == site_id and scene.get("task_id") == source["task_id"]
        and scene.get("task_kind") == "rigid_pick_place" and set(task) == set(original)
        and all(task[k] == original[k] for k in task if k not in ROBOT_TASK_FIELDS)
        and task.get("task_success_contract") == inputs["task_success_contract"]
        and task.get("task_success_contract_digest") == inputs["task_success_contract_digest"]
        and task.get("control_frequency_hz") == 50, "current_task_changed")
    robot = scene["robot"]
    asset = receipt.get("robot_asset_binding") or {}
    require(asset.get("staged_sha256") == profile["robot_usd_sha256"]
        and asset.get("staged_size_bytes") == profile["robot_usd_size_bytes"]
        and robot.get("usd_path") == asset.get("staged_relative_path"), "g1_robot_asset_binding_changed")
    bound(_child(Path(record["path"]).parent, asset["staged_relative_path"]),
        {"sha256": profile["robot_usd_sha256"], "size_bytes": profile["robot_usd_size_bytes"]})
    require(set(robot) == {*profile["robot_template"], "usd_path", "base_pose_world", "usd_size_bytes",
            "usd_dependency_bindings", "usd_runtime_asset_dependencies"}
        and robot["usd_size_bytes"] == profile["robot_usd_size_bytes"] and robot["usd_dependency_bindings"] == []
        and robot["usd_runtime_asset_dependencies"] == profile["robot_runtime_asset_dependencies"]
        and all(robot.get(key) == value for key, value in profile["robot_template"].items()), "g1_robot_numeric_template_changed")
    require(robot.get("robot_id") == "unitree_g1" and robot.get("hand_id") == "unitree_dex3_1"
        and robot.get("action_interface") == "g1_dex3_named_joint_position.v1"
        and robot.get("usd_sha256") == profile["robot_usd_sha256"]
        and all(set(robot.get(key, {})) == set(profile["joint_names"]) for key in
            ("joint_reset_positions_rad", "actuator_parameters", "joint_position_limits_rad")), "g1_robot_identity_changed")
    cameras = [row for row in scene.get("cameras", []) if row.get("role") == "head"]
    require(len(cameras) == 1 and cameras[0].get("policy_input") is True
        and cameras[0].get("intrinsics", {}).get("width") == 640
        and cameras[0].get("intrinsics", {}).get("height") == 480, "g1_head_camera_changed")
    require(bool(task.get("g1_navigation_goal")) == candidate_id.endswith("_vision_navi"), "candidate_task_mismatch")
    verify_assumptions(request=request, scene=scene, base_request=base_request, base_scene=base_scene,
        base_receipt=base_receipt, profile=profile, source=source, site_id=site_id, inputs=inputs)
    return scene


def verify_assumptions(*, request, scene, base_request, base_scene, base_receipt, profile, source, site_id, inputs):
    derivation = request["g1_scene_derivation"]
    assumptions = derivation.get("assumptions")
    require(derivation.get("schema_version") == "native_g1_beta_scene_packet_derivation.v1"
        and derivation.get("claim_ceiling") == "development_only" and isinstance(assumptions, dict)
        and assumptions.get("assumptions_digest") == common.digest(assumptions, "assumptions_digest"), "simulation_assumptions_missing")
    original = base_request["task_spec"]
    duration = min(float(original["maximum_episode_seconds"]), int(original["maximum_action_steps"]) / float(original["control_frequency_hz"]))
    expected = {"recipe_id": "native_g1_beta_retained_scene.v1", "current_task_source": source, "site_id": site_id,
        "source_packet_receipt_digest": base_receipt["receipt_digest"], "source_scene_plan_digest": base_receipt["arena_scene_plan_digest"],
        "robot_usd_sha256": profile["robot_usd_sha256"], "task_success_contract_digest": inputs["task_success_contract_digest"],
        "stance_method": "existing_scene_collision_floor_and_clear_footprint_search", "stance_is_geometric_estimate": True, "large_collision_bounds_retained": True,
        "measured_robot_calibration_claimed": False, "footprint_half_extent_m": [0.12, 0.23],
        "root_height_above_inferred_floor_m": 0.79, "head_camera": profile["head_camera"],
        "head_camera_calibration_measured": False, "head_camera_basis": "existing_g1_development_mount_simulation_assumption",
        "overview_basis": "unchanged_current_task_review_camera", "task_hand": "right", "task_hand_is_simulation_assumption": True,
        "control_frequency_hz": 50.0, "physics_frequency_hz": 200.0,
        "source_control_frequency_hz": original["control_frequency_hz"], "source_maximum_action_steps": original["maximum_action_steps"],
        "source_maximum_episode_seconds": original["maximum_episode_seconds"],
        "maximum_action_steps": math.floor(duration * 50), "maximum_episode_seconds": duration,
        "navigation_goal_added": False, "source_geometry_changed": False, "policy_queried": False,
        "simulator_executed": False, "physical_outcome_claimed": False, "new_human_review_claimed": False,
        "current_consent_revalidated": False}
    require(set(assumptions) == {*expected, "stance", "subject_bounds_world_m", "subject_bounds_basis", "assumptions_digest"}
        and all(assumptions.get(key) == value for key, value in expected.items())
        and all(type(assumptions[key]) is bool for key, value in expected.items() if type(value) is bool), "simulation_assumptions_changed")
    require(math.isfinite(duration) and duration >= 0.02 and request.get("physics_frequency_hz") == 200
        and request["task_spec"]["maximum_action_steps"] == expected["maximum_action_steps"]
        and request["task_spec"]["maximum_episode_seconds"] == duration, "simulation_budget_changed")
    expected_task = {**original, "task_success_contract": inputs["task_success_contract"],
        "task_success_contract_digest": inputs["task_success_contract_digest"], "control_frequency_hz": 50.0,
        "maximum_action_steps": expected["maximum_action_steps"], "maximum_episode_seconds": duration}
    require(request["task_spec"] == expected_task, "simulation_task_semantics_changed")
    overview = [row for row in base_request.get("cameras", []) if row.get("role") == "overview"]
    require(len(overview) == 1 and request.get("cameras") == [profile["head_camera"], overview[0]], "simulation_camera_recipe_changed")
    stance = assumptions["stance"]
    require(stance.get("schema_version") == "blueprint.team_free_base_stance.v1"
        and stance.get("stance_digest") == common.digest(stance, "stance_digest")
        and stance.get("footprint_half_extent_m") == [0.12, 0.23]
        and stance.get("search_radius_m") == 1.6 and stance.get("grid_step_m") == 0.05
        and type(stance.get("clear_stances")) is int and stance["clear_stances"] > 0
        and stance.get("validator") == "scene_placement.validation.validate_stand_pose", "simulation_stance_invalid")
    collision = [row for row in base_receipt["source_bindings"] if row["semantic_role"] == "scene_collision"]
    require(len(collision) == 1 and stance.get("scene_collision_sha256") == collision[0]["staged_sha256"], "simulation_stance_source_changed")
    position, orientation = stance.get("position_world_m"), stance.get("orientation_xyzw")
    require(isinstance(position, list) and len(position) == 3 and isinstance(orientation, list) and len(orientation) == 4
        and all(type(v) in (int, float) and math.isfinite(v) for v in [*position, *orientation, stance["floor_z_m"], stance["yaw_rad"]])
        and math.isclose(position[2], stance["floor_z_m"] + 0.79, abs_tol=1e-9)
        and all(math.isclose(v, expected, abs_tol=1e-9) for v, expected in zip(orientation,
            [0, 0, math.sin(stance["yaw_rad"] / 2), math.cos(stance["yaw_rad"] / 2)], strict=True)), "simulation_stance_pose_invalid")
    selected = ((base_request.get("configured_task_template_adapter") or {}).get("source_documents") or {}).get("documents", {}).get("source_object")
    if selected:
        bounds = {"minimum": selected["aabb_min_xyz_m"], "maximum": selected["aabb_max_xyz_m"]}
        basis = "retained_source_object_selection_world_aabb"
    else:
        start = original["start_pose_world"]
        require(start[3:] == [0.0, 0.0, 0.0, 1.0], "simulation_subject_bounds_invalid")
        local = original["subject_collision_bounds_scoring_frame_m"]
        bounds = {key: [float(start[i]) + float(local[key][i]) for i in range(3)] for key in ("minimum", "maximum")}
        basis = "retained_unrotated_task_scoring_bounds"
    require(assumptions["subject_bounds_world_m"] == bounds and assumptions["subject_bounds_basis"] == basis
        and all(abs(position[i] - original["start_pose_world"][i]) <= 1.6 + 1e-9 for i in range(2)), "simulation_subject_binding_changed")
    pose = {"position_world_m": position, "orientation_xyzw": orientation}
    require(scene["robot"].get("base_pose_world") == pose and request.get("robot_base_pose_world") == pose,
        "simulation_stance_binding_changed")


def reconstruct(request, registry, *, profile_root, shared_registry_path=SHARED_REGISTRY, source_queue=SOURCE_QUEUE):
    request = validate_request(request)
    profile = common.read(profile_root / "native_g1_registration_profile.json")
    require(profile.get("schema_version") == "native_g1_registration_profile.v1" and set(profile["candidate_digests"]) == set(CANDIDATES), "installed_profile_changed")
    site_root, inputs, base_receipt, base_request, base_scene = site_inputs(request, registry, source_queue=source_queue)
    shared = sealed(protected(shared_registry_path), "native_g1_team_campaign_registry.v1", "registry_digest")
    require(shared["registry_digest"] == request["shared_registry_digest"], "shared_registry_changed")
    common_records, shared_reason = {}, None
    try:
        require(request["publisher_source"] is not None and request["runtime_source"] is not None, "shared_sources_missing")
        matches = []
        for row in shared["bindings"]:
            pub = Path(row["publisher_source_dir"]) / "native_g1_publisher_source_stage.v1.json"
            run = Path(row["runtime_source_receipt_path"])
            try:
                records = {"publisher_source": bound(pub, request["publisher_source"]), "runtime_source": bound(run, request["runtime_source"])}
            except (OSError, ValueError):
                continue
            if records not in matches:
                matches.append(records)
        require(len(matches) == 1, "shared_sources_missing_or_ambiguous")
        common_records = matches[0]
        publisher(common_records["publisher_source"], profile)
        runtime(common_records["runtime_source"], profile)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        shared_reason = "g1_shared_source_verification_failed"
    registrations, results = {}, []
    for row in request["candidates"]:
        candidate = row["candidate_id"]
        reason = shared_reason or "g1_current_scene_packet_missing"
        try:
            require(shared_reason is None, "candidate_inputs_missing")
            if row["scene_packet"] is None:
                scene_record = file_record(site_root / "native-g1" / base_receipt["receipt_digest"][7:] / common.digest(request["source"])[7:] / "native_task_arena_packet_receipt.v1.json")
            else:
                scene_record = discover(site_root / "native-g1", row["scene_packet"], names={"native_task_arena_packet_receipt.v1.json"})
            records = {**common_records, "scene_packet": scene_record}
            reason = "g1_current_scene_verification_failed"
            if candidate.endswith("_vision_navi") and base_scene["task_spec"].get("g1_navigation_goal") is None:
                reason = "g1_current_navigation_goal_authority_missing"
                require(False, "navigation_goal_missing")
            scene = verify_scene(records["scene_packet"], candidate_id=candidate, profile=profile,
                source=request["source"], site_id=request["site_id"], inputs=inputs, base_receipt=base_receipt,
                base_request=base_request, base_scene=base_scene)
            if candidate.endswith("_vision_navi"):
                reason = "g1_current_navigation_goal_authority_missing"
                require(row["navigation_authority"] is not None, "navigation_authority_missing")
                records["navigation_authority"] = discover(site_root, row["navigation_authority"], names={"navigation_authority.json", "native_g1_navigation_goal_authority.v1.json"})
                navigation = common.read(Path(records["navigation_authority"]["path"]))
                require(navigation.get("scope", {}).get("scene_plan_digest") == scene["plan_digest"], "navigation_scene_changed")
            registrations[candidate] = {"native_g1": records}
            results.append({"candidate_id": candidate, "status": "registered"})
        except (OSError, ValueError, KeyError, TypeError):
            results.append({"candidate_id": candidate, "status": "unavailable", "reason": reason})
    return {"schema_version": PROPOSAL_SCHEMA, "request_digest": request["request_digest"],
        "site_id": request["site_id"], "source": request["source"], "registrations": registrations, "results": results,
        "execution_authority_granted": False, "candidate_policy_queried": False, "provider_allocations_performed": 0}


def verify_proposal(request, registry, *, state_root, profile_root, **kwargs):
    destination = common.safe_path(state_root / request["installation_id"])
    require(common.read(destination / "request.json") == request, "retained_request_changed")
    expected = reconstruct(request, registry, profile_root=profile_root, **kwargs)
    require(common.read(destination / "proposal.json") == expected, "fixed_proposal_changed")
    return expected
