"""Installed stdlib reconstruction of fixed scripted registrations; never imports a checkout."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

if __package__:
    from . import public_policy_registration_contract as common
else:
    _spec = importlib.util.spec_from_file_location("_scripted_common", Path(__file__).with_name("public_policy_registration_contract.py"))
    assert _spec and _spec.loader
    common = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(common)

require, read, digest = common.require, common.read, common.digest
CHECKPOINT_ID = "blueprint-scripted-pick-place-v1"
CODE_FILES = ("scripted_robot_baseline.py", "scripted_robot_baseline_runtime.py", "scripted_robot_grasp_frames.py")
VARIANT_CODE_FILES = ("scripted_robot_baseline.py", "scripted_robot_grasp_frames.py", "scripted_robot_robotiq_runtime.py")
VARIANT_BODIES = frozenset({"universal_robots_ur5e_robotiq_2f85_sim", "universal_robots_ur10e_robotiq_2f85_sim",
                          "franka_research_3_robotiq_2f85_sim"})
INDUSTRIAL_CODE_FILES = ("scripted_robot_baseline.py", "scripted_robot_grasp_frames.py", "scripted_robot_industrial_robotiq_runtime.py")
INDUSTRIAL_BODIES = frozenset({'fanuc_crx_10ia_robotiq_2f85_sim', 'abb_gofa_crb15000_5_95_robotiq_2f85_sim', 'fanuc_m710ic_45m_robotiq_2f85_sim', 'abb_gofa_crb15000_12_127_robotiq_2f85_sim', 'fanuc_crx_10ia_l_robotiq_2f85_sim'})
YASKAWA_CODE_FILES = ("scripted_robot_baseline.py", "scripted_robot_grasp_frames.py", "scripted_robot_yaskawa_robotiq_runtime.py")
YASKAWA_BODIES = frozenset({"yaskawa_gp12_robotiq_2f85_sim", "yaskawa_gp20_robotiq_2f85_sim"})
EVIDENCE_KIND = "scripted_simulator_state_baseline"
ACTION_SOURCE = "scripted_simulator_state"
UNAVAILABLE_REASONS = {"prepared_body_unavailable", "prepared_body_verification_failed", "source_or_import_verification_failed",
                       "tool_profile_unavailable"}


def load_profile(profile_root: Path, *, source_root: Path | None = None, license_path: Path | None = None) -> dict:
    profile = read(profile_root / "scripted_robot_baseline_registration_profile.json")
    require(profile["policy_profile"] == common.SCRIPTED_PROFILE_ID and profile["team_id"] == common.SCRIPTED_TEAM_ID
            and profile["checkpoint_id"] == CHECKPOINT_ID and list(profile["bodies"]) == sorted(common.SCRIPTED_BODIES),
            "fixed_script_profile_changed")
    yaskawa = profile["yaskawa_source_identity"]
    require(profile["yaskawa_policy_profile"] == "blueprint_scripted_pick_place_yaskawa_robotiq.v1"
            and set(profile["yaskawa_bodies"]) == YASKAWA_BODIES
            and yaskawa["schema_version"] == "blueprint_scripted_controller_source.v1"
            and [row["path"] for row in yaskawa["files"]] == list(YASKAWA_CODE_FILES), "script_yaskawa_inventory_changed")
    industrial = profile["industrial_source_identity"]
    require(profile["industrial_policy_profile"] == "blueprint_scripted_pick_place_industrial_robotiq.v1"
            and set(profile["industrial_bodies"]) == INDUSTRIAL_BODIES
            and industrial["schema_version"] == "blueprint_scripted_controller_source.v1"
            and [row["path"] for row in industrial["files"]] == list(INDUSTRIAL_CODE_FILES), "script_industrial_inventory_changed")
    identity = profile["source_identity"]
    variant = profile["variant_source_identity"]
    require(profile["variant_policy_profile"] == "blueprint_scripted_pick_place_robotiq.v1"
            and set(profile["variant_bodies"]) == VARIANT_BODIES
            and variant["schema_version"] == "blueprint_scripted_controller_source.v1"
            and [row["path"] for row in variant["files"]] == list(VARIANT_CODE_FILES), "script_variant_inventory_changed")
    require(identity["schema_version"] == "blueprint_scripted_controller_source.v1"
            and [row["path"] for row in identity["files"]] == list(CODE_FILES), "script_source_inventory_changed")
    if source_root is None:
        source_root = profile_root / "scripted-baseline-source"
        require({p.name for p in source_root.iterdir()} == set(CODE_FILES) | set(VARIANT_CODE_FILES) | set(INDUSTRIAL_CODE_FILES) | set(YASKAWA_CODE_FILES), "script_source_inventory_changed")
    for row in [*identity["files"], *variant["files"], *industrial["files"], *yaskawa["files"]]:
        require(set(row) == {"path", "sha256"} and common.file_record(source_root / row["path"])["sha256"] == row["sha256"],
                "script_source_changed")
    actual = common.file_record(license_path or profile_root / "scripted_robot_baseline_license.txt")
    require(all(actual[key] == profile["license_evidence"][key] for key in ("sha256", "size_bytes")),
            "script_license_changed")
    return profile


def source_identity(profile: dict, body: str) -> dict:
    if body in YASKAWA_BODIES:
        return profile["yaskawa_source_identity"]
    if body in INDUSTRIAL_BODIES:
        return profile["industrial_source_identity"]
    return profile["variant_source_identity"] if body in VARIANT_BODIES else profile["source_identity"]


def publication(request: dict, model_root: Path) -> dict:
    value = read(model_root / "batches" / (request["preparation_authorization_id"] + ".json"))
    require(value.get("publication_digest") == request["preparation_publication_digest"] == digest(value, "publication_digest")
            and value.get("status") == "completed" and value.get("claim_ceiling") == "robot_asset_import_only"
            and value.get("candidate_policy_queried") is False and value.get("task_executed") is False,
            "preparation_publication_changed")
    rows = value.get("results")
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
            and len({row.get("embodiment_id") for row in rows}) == len(rows), "preparation_publication_changed")
    return value


def result_row(body: str, status: str, reason: str | None = None) -> dict:
    value = {"embodiment_id": body, "candidate_id": "blueprint_scripted_" + body, "status": status,
             "execution_evidence_kind": EVIDENCE_KIND}
    if reason:
        value["reason"] = reason
    return value


def rotation(q: list) -> list:
    require(isinstance(q, list) and len(q) == 4 and all(type(v) in (float, int) and math.isfinite(v) for v in q)
            and math.isclose(sum(v*v for v in q), 1, abs_tol=1e-4), "script_grasp_transform_changed")
    x, y, z, w = q
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def vector(value: list) -> list:
    require(isinstance(value, list) and len(value) == 3
            and all(type(v) in (float, int) and math.isfinite(v) for v in value), "script_grasp_transform_changed")
    return value


def validated_policy(manifest: dict, policy: dict, profile: dict) -> dict:
    """Reconstruct all fields; accept only equivalent numeric TCP composition within 1e-9."""
    body = manifest["embodiment_id"]
    require(body in common.SCRIPTED_BODIES and manifest.get("embodiment_digest") == digest(manifest, "embodiment_digest"),
            "fixed_manifest_changed")
    frame = profile["frames"][profile["bodies"][body]]
    arms = manifest.get("arms", [])
    names = ["arm"] if body in {"kinova_gen3", "ufactory_xarm7"} | VARIANT_BODIES | INDUSTRIAL_BODIES | YASKAWA_BODIES else ["left", "right"]
    require([arm["name"] for arm in arms] == names and len(manifest["grippers"]) == len(names), "script_source_tool_changed")
    channels = {row["name"]: row for row in manifest["grippers"]}
    bindings = policy.get("grasp_bindings")
    require(isinstance(bindings, dict) and set(bindings) == set(names), "script_grasp_binding_changed")
    for arm in arms:
        prefix = "" if arm["name"] == "arm" else arm["name"] + "_"
        if body == "ufactory_xarm7_bimanual":
            prefix = {"left": "L_", "right": "R_"}[arm["name"]]
        channel = channels[arm["gripper_name"]]
        jaws = {prefix+name: values for name, values in frame["jaws"].items()}
        require(arm["tool_link"] == prefix+frame["tool"] and len(arm["joint_names"]) == frame["dof"]
                and set(channel["contact_links"]) == {prefix+name for name in frame["contacts"]}
                and {joint["name"] for joint in channel["joints"]} == set(jaws), "script_source_tool_changed")
        require(all(math.isclose(joint[field], jaws[joint["name"]][index], abs_tol=1e-8)
                    for joint in channel["joints"] for index, field in enumerate(("open", "closed"))), "script_source_closure_changed")
        binding = bindings[arm["name"]]
        transform = binding.get("tcp_from_grasp", {})
        require(set(transform) == {"position_m", "orientation_xyzw"}, "script_grasp_transform_changed")
        expected = {"schema_version": "blueprint.scripted_robot_grasp_binding.v1", "tcp_from_grasp": transform,
            "basis_reference": frame["url"], "source_geometry_basis": frame["basis"],
            "task_grasp_convention": "+Z approach into feature; +Y jaw separation; +X right handed",
            "additional_source_references": [frame["additional_source"]] if "additional_source" in frame else [],
            "simulation_assumptions": frame["assumptions"], "publisher_calibration_claimed": False,
            "physical_contact_point_measured": False, "native_task_success_claimed": False}
        require(binding == expected, "script_grasp_binding_changed")
        tcp = arm["tool_from_tcp"]
        a, b = rotation(tcp["orientation_xyzw"]), rotation(transform["orientation_xyzw"])
        p, t = vector(tcp["position_m"]), vector(transform["position_m"])
        y, z = frame["jaw"], frame["approach"]
        x = [y[1]*z[2]-y[2]*z[1], y[2]*z[0]-y[0]*z[2], y[0]*z[1]-y[1]*z[0]]
        target = [[x[i], y[i], z[i]] for i in range(3)]
        require(all(math.isclose(sum(a[i][k]*b[k][j] for k in range(3)), target[i][j], rel_tol=0, abs_tol=1e-9)
                    for i in range(3) for j in range(3))
                and all(math.isclose(p[i]+sum(a[i][k]*t[k] for k in range(3)), frame["position"][i], rel_tol=0, abs_tol=1e-9)
                        for i in range(3)), "script_grasp_transform_changed")
    expected = {"policy_interface_id": common.SCRIPTED_PROFILE_ID,
        "policy_id": (profile["yaskawa_policy_profile"] if body in YASKAWA_BODIES else
                      profile["industrial_policy_profile"] if body in INDUSTRIAL_BODIES else
                      profile["variant_policy_profile"] if body in VARIANT_BODIES else common.SCRIPTED_PROFILE_ID),
        "model_identity_digest": digest(source_identity(profile, body)), "source_revision": "1", "embodiment_id": body,
        "configuration_digest": manifest["embodiment_digest"], "action_chunk_rows": 1, "open_loop_horizon": 1,
        "action_hz": manifest["control_hz"], "camera_names": [row["name"] for row in manifest["cameras"]],
        "grasp_bindings": bindings, "execution_evidence_kind": EVIDENCE_KIND, "action_source": ACTION_SOURCE}
    require(policy == expected, "fixed_script_policy_changed")
    return expected


def rights_record(request: dict, profile: dict, body: str) -> dict:
    require(body in common.SCRIPTED_BODIES, "body_mismatch")
    value = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
        "candidate_id": "blueprint_scripted_" + body, "team_id": common.SCRIPTED_TEAM_ID, "checkpoint_id": CHECKPOINT_ID,
        "license_to_run_granted": True, "policy_self_grading_permitted": False, "official_ranking_permitted": False,
        "secret_material_recorded": False, "adp_qualified_evidence_permitted": False, "license_id": "MIT",
        "model_identity_digest": digest(source_identity(profile, body)), "license_basis": "blueprint_owned_source_repository_license",
        "license_evidence": profile["license_evidence"], "controller_source_identity": source_identity(profile, body),
        "authorization_basis": "user_instruction_to_run_simulation_baselines",
        "authorization_reference": request["authorization_reference"], "execution_evidence_kind": EVIDENCE_KIND,
        "private_vendor_authorization_claimed": False, "physical_validation_claimed": False}
    return {**value, "rights_digest": digest(value)}


def deployment(body: str) -> dict:
    return {"configuration_id": body, "policy_route_id": "blueprint_scripted_baseline", "companies": [],
        "relationship": "public_controller_integration", "evidence_url": "BlueprintCapturePipeline:docs/scripted_robot_baseline.md"}


def verify_batch(request: dict, *, state_root: Path, model_root: Path, profile_root: Path) -> tuple[dict, list]:
    profile = load_profile(profile_root)
    published = {row["embodiment_id"]: row for row in publication(request, model_root)["results"]}
    destination = common.safe_path(state_root / request["installation_id"])
    require(read(destination / "request.json") == request, "retained_request_changed")
    proposal = read(destination / "proposal.json")
    registrations, results = {}, []
    supplied = proposal.get("registrations", {})
    rows = proposal.get("results", [])
    require(isinstance(supplied, dict) and isinstance(rows, list) and len(rows) == len(common.SCRIPTED_BODIES),
            "fixed_registration_changed")
    for body, row in zip(common.SCRIPTED_BODIES, rows, strict=True):
        if published.get(body, {}).get("status") != "retained":
            require(row == result_row(body, "unavailable", "prepared_body_unavailable"), "fixed_registration_changed")
        elif row.get("status") == "unavailable":
            # A failed source/import/tool check grants nothing. Preserve its
            # bounded reason; no arbitrary exception text or caller paths.
            require(row.get("reason") in UNAVAILABLE_REASONS
                    and row == result_row(body, "unavailable", row["reason"]), "fixed_registration_changed")
        else:
            require(row == result_row(body, "registered"), "fixed_registration_changed")
            _, original, model = common.retained_body(request, model_root=model_root, embodiment_id=body)
            manifest = original["robot_embodiment"]
            path = destination / body / (manifest["embodiment_digest"][7:] + ".robot_manifest.json")
            require(read(path) == manifest, "fixed_manifest_changed")
            candidate = "blueprint_scripted_" + body
            require(candidate in supplied, "fixed_registration_changed")
            policy = validated_policy(manifest, supplied[candidate].get("policy_spec", {}), profile)
            expected = {"team_id": common.SCRIPTED_TEAM_ID, "checkpoint_id": CHECKPOINT_ID,
                "display_name": "Blueprint scripted pick/place — " + body, "model": model, "manifest": common.file_record(path),
                "policy_spec": policy, "rights_binding": rights_record(request, profile, body),
                "dependency_closure": original["model_record"]["dependency_closure"],
                "source_import_provenance": original["model_record"]["source_import_provenance"],
                "minimum_gpu_memory_gb": 46, "deployment": deployment(body)}
            require(supplied[candidate] == expected, "fixed_registration_changed")
            registrations[candidate] = expected
        results.append(row)
    require(proposal == {"schema_version": "scripted_robot_baseline_install_proposal.v1",
        "request_digest": request["request_digest"], "registrations": registrations, "results": results,
        "candidate_policy_queried": False, "scripted_controller_invoked": False, "provider_allocations_performed": 0},
        "fixed_registration_changed")
    return registrations, results
