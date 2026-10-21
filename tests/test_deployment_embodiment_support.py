"""Shortlisted options reach real robot admission without becoming run authority."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from blueprint_pipeline import deployment_embodiment_support as support
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.job_capability_routing import route_job_capabilities
from blueprint_pipeline.task_evaluation_scene_intake_http import register_scene_intake_routes


def _shortlist(identity="universal_robots_ur5e", route="customer_controller"):
    return support.make_shortlist([{"embodiment_id": identity, "policy_route_ids": [route]}],
                                  source_launch_id="source-scene-v1")


def test_every_catalog_option_can_be_shortlisted_and_replayed_without_execution():
    catalog = support.support_catalog()
    assert catalog["catalog_digest"] == cross_runtime_canonical_digest(catalog, digest_field="catalog_digest")
    for body in catalog["embodiments"]:
        request = support.make_shortlist([{
            "embodiment_id": body["embodiment_id"],
            "policy_route_ids": [route["route_id"] for route in body["policy_routes"]],
        }])
        assert support.validate_shortlist(request) == request
        plan = support.prepare_shortlist(request)
        assert plan["execution_authorized"] is False
        assert plan["provider_mutation_performed"] is False
        assert plan["integrations"][0]["status"] == "requires_setup"
        assert plan["integrations"][0]["blockers"]
        assert all(route["blockers"] for route in plan["integrations"][0]["policy_routes"])
        assert plan["plan_digest"] == cross_runtime_canonical_digest(plan, digest_field="plan_digest")
        if body["arm_count"] == 2:
            assert "independent_bimanual_tool_channels_required" in plan["integrations"][0]["blockers"]


def test_catalog_returns_fresh_data_and_canonical_shortlists_are_selection_order_independent():
    catalog = support.support_catalog()
    bodies = catalog["embodiments"][:2]
    selections = [{"embodiment_id": body["embodiment_id"],
                   "policy_route_ids": [route["route_id"] for route in reversed(body["policy_routes"])]}
                  for body in reversed(bodies)]
    first = support.make_shortlist(selections)
    assert first == support.make_shortlist(list(reversed(selections)))
    catalog["embodiments"][0]["display_name"] = "changed"
    assert support.support_catalog()["embodiments"][0]["display_name"] != "changed"


@pytest.mark.parametrize("selection", [
    [],
    [{"embodiment_id": "unknown", "policy_route_ids": ["customer_controller"]}],
    [{"embodiment_id": "universal_robots_ur5e", "policy_route_ids": ["dreamlabs_dream1"]}],
    [{"embodiment_id": "fanuc_crx_10ia_l", "policy_route_ids": ["dreamlabs_dream1"]}],
    [{"embodiment_id": "yaskawa_gp12", "policy_route_ids": []}],
    [{"embodiment_id": "yaskawa_gp12", "policy_route_ids": ["customer_controller", "customer_controller"]}],
    [{"embodiment_id": "yaskawa_gp12", "policy_route_ids": ["customer_controller"], "runnable": True}],
])
def test_unknown_duplicate_and_incompatible_options_fail_closed(selection):
    with pytest.raises(support.DeploymentSupportError):
        support.make_shortlist(selection)


def test_repeated_robot_and_invalid_source_are_not_coerced():
    row = {"embodiment_id": "universal_robots_ur5e", "policy_route_ids": ["customer_controller"]}
    with pytest.raises(support.DeploymentSupportError, match="embodiment_invalid"):
        support.make_shortlist([row, row])
    with pytest.raises(support.DeploymentSupportError, match="source_invalid"):
        support.make_shortlist([row], source_launch_id="../../elsewhere")


@pytest.mark.parametrize("field,value,reason", [
    ("claim_ceiling", "execution_authorized", "support_shortlist_invalid"),
    ("catalog_digest", "sha256:" + "0" * 64, "catalog_changed"),
    ("shortlist_digest", "sha256:" + "0" * 64, "digest_or_order_invalid"),
    ("source_launch_id", "another-scene", "digest_or_order_invalid"),
])
def test_shortlist_is_bound_to_catalog_scene_and_planning_claim(field, value, reason):
    request = _shortlist()
    request[field] = value
    with pytest.raises(support.DeploymentSupportError, match=reason):
        support.validate_shortlist(request)


def test_no_digest_rewrite_can_substitute_crx_variant_or_add_run_authority():
    request = _shortlist("fanuc_crx_10ia_l")
    request["selections"][0]["embodiment_id"] = "fanuc_crx_10ia"
    request["shortlist_digest"] = cross_runtime_canonical_digest(request, digest_field="shortlist_digest")
    with pytest.raises(support.DeploymentSupportError, match="embodiment_invalid"):
        support.validate_shortlist(request)
    request = _shortlist()
    request["execution_authorized"] = True
    with pytest.raises(support.DeploymentSupportError, match="shortlist_invalid"):
        support.validate_shortlist(request)


@pytest.mark.parametrize("change", ["duplicate_robot", "duplicate_route", "wrong_dof", "unsafe_url", "absolute_path"])
def test_catalog_validation_rejects_ambiguous_identity_and_unsafe_asset_metadata(change):
    catalog = support.support_catalog()
    catalog.pop("catalog_digest")
    if change == "duplicate_robot":
        catalog["embodiments"].append(copy.deepcopy(catalog["embodiments"][0]))
    elif change == "duplicate_route":
        catalog["policy_routes"].append(copy.deepcopy(catalog["policy_routes"][0]))
    elif change == "wrong_dof":
        catalog["embodiments"][0]["arm_dof"] = [True]
    elif change == "unsafe_url":
        catalog["embodiments"][0]["asset_sources"][0]["url"] = "https://user:secret@example.com/robot.usd"
    else:
        catalog["embodiments"][0]["asset_sources"][0]["relative_path"] = "/other/robot.usd"
    with pytest.raises(support.DeploymentSupportError):
        support.validate_support_catalog(catalog)


def test_existing_job_router_lists_support_options_without_creating_runnable_configurations():
    routed = route_job_capabilities("rigid_pick_place", provided_inputs=("metric_object_geometry", "target_region"))
    assert routed["support_options"] == support.support_catalog()
    assert {row["embodiment_id"] for row in routed["configurations"]} == {"droid_franka_robotiq_2f85"}
    gap = route_job_capabilities("rigid_pick_place", embodiment_ids=["fanuc_crx_10ia_l"])
    assert gap["status"] == "capability_gap" and gap["configurations"] == []


def test_cli_replays_browser_shaped_handoff_and_refuses_tampering(tmp_path):
    source, destination = tmp_path / "choice.json", tmp_path / "plan.json"
    source.write_text(json.dumps(_shortlist()))
    assert support.main(["--shortlist", str(source), "--output", str(destination)]) == 0
    plan = json.loads(destination.read_text())
    assert plan["shortlist_digest"] == _shortlist()["shortlist_digest"]
    assert plan["execution_authorized"] is False
    source.write_text(json.dumps({**_shortlist(), "claim_ceiling": "runnable"}))
    with pytest.raises(SystemExit) as error:
        support.main(["--shortlist", str(source)])
    assert error.value.code == 2


def _client(*, issuer="blueprint-webapp"):
    app = FastAPI()

    @app.middleware("http")
    async def verified_issuer(request: Request, call_next):
        request.state.intake_client_id = issuer
        return await call_next(request)

    register_scene_intake_routes(app, lambda: None, lambda: {})
    return TestClient(app)


def test_signed_website_can_fetch_and_preflight_without_any_intake_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(tmp_path))
    client = _client()
    headers = {"x-blueprint-pipeline-signature": "verified-by-admission-dependency"}
    catalog = client.get("/api/live-pipeline/deployment-embodiment-support", headers=headers)
    assert catalog.status_code == 200 and catalog.json() == support.support_catalog()
    assert catalog.headers["cache-control"] == "private, no-store"
    result = client.post("/api/live-pipeline/deployment-embodiment-shortlist-preflight", headers=headers,
                         json=_shortlist())
    assert result.status_code == 200 and result.json() == support.prepare_shortlist(_shortlist())
    assert list(tmp_path.iterdir()) == []
    assert client.post("/api/live-pipeline/deployment-embodiment-shortlist-preflight", headers=headers,
                       json={**_shortlist(), "shortlist_digest": "bad"}).status_code == 422
    assert client.get("/api/live-pipeline/deployment-embodiment-support").status_code == 403
    assert _client(issuer="other").get("/api/live-pipeline/deployment-embodiment-support", headers=headers).status_code == 403


def _native_inputs(tmp_path):
    """Synthetic joint/link admission fixture, never an OEM robot or performance proof."""
    from pxr import Usd, UsdPhysics

    from tests.test_team_embodiment import manifest, station_plan

    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
    for name in ("base_link", "wrist", "left_finger", "right_finger"):
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/" + name, "Xform"))
    joints = [f"test_joint_{index}" for index in range(6)]
    for name in joints:
        UsdPhysics.RevoluteJoint.Define(stage, "/robot/" + name)
    UsdPhysics.PrismaticJoint.Define(stage, "/robot/finger_joint")
    UsdPhysics.FixedJoint.Define(stage, "/robot/anchor").CreateBody1Rel().SetTargets(["/robot/base_link"])
    stage.GetRootLayer().Save()
    declared = manifest(path, embodiment_id="universal_robots_ur5e",
                        joints=[{"name": name, "control": "position", "lower": -3.0, "upper": 3.0,
                                 "max_velocity": 2.0} for name in joints],
                        reset_joint_positions={**dict.fromkeys(joints, 0.0), "finger_joint": 0.04})
    scene = station_plan()
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    return path, declared, scene


def test_shortlisted_single_arm_reaches_existing_usd_binding_but_not_execution(tmp_path):
    path, manifest, scene = _native_inputs(tmp_path)
    result = support.prepare_native_binding(_shortlist(), embodiment_id="universal_robots_ur5e",
                                           manifest=manifest, model_path=path, scene_plan=scene)
    assert result["execution_authorized"] is False
    assert result["robot_binding"]["model"]["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["policy_handoff"]["robot_embodiment"]["embodiment_id"] == "universal_robots_ur5e"
    assert result["policy_handoff"]["policy_interface_id"] == "blueprint_policy_https.v1"
    assert "credential_ref" not in json.dumps(result)
    assert "rights_and_provider_terms" in result["remaining_admission"]
    assert "exact_catalog_asset_provenance_and_license" in result["remaining_admission"]
    assert result["catalog_asset_identity_verified"] is False
    assert result["model_file_digest_verified"] is True
    assert result["scene_plan_digest"] == canonical_digest(scene)
    assert result["task_spec_digest"] == canonical_digest(scene["task_spec"])


def test_native_preflight_identifies_the_checked_scene_and_rejects_a_changed_seal(tmp_path):
    path, manifest, scene = _native_inputs(tmp_path)
    original = support.prepare_native_binding(_shortlist(), embodiment_id="universal_robots_ur5e",
                                             manifest=manifest, model_path=path, scene_plan=scene)
    changed = copy.deepcopy(scene)
    changed["task_spec"]["start_pose_world"][0] += 0.01
    with pytest.raises(support.DeploymentSupportError, match="scene_plan_digest_mismatch"):
        support.prepare_native_binding(_shortlist(), embodiment_id="universal_robots_ur5e",
                                       manifest=manifest, model_path=path, scene_plan=changed)
    changed["plan_digest"] = canonical_digest(changed, digest_field="plan_digest")
    other = support.prepare_native_binding(_shortlist(), embodiment_id="universal_robots_ur5e",
                                          manifest=manifest, model_path=path, scene_plan=changed)
    assert original["scene_plan_digest"] != other["scene_plan_digest"]
    assert original["task_spec_digest"] != other["task_spec_digest"]
    assert original["preflight_digest"] != other["preflight_digest"]


def test_native_binding_rejects_changed_asset_and_wrong_body_and_unsupported_task(tmp_path):
    path, manifest, scene = _native_inputs(tmp_path)
    with pytest.raises(support.DeploymentSupportError, match="identity_mismatch"):
        support.prepare_native_binding(_shortlist("fanuc_crx_10ia_l"), embodiment_id="fanuc_crx_10ia_l",
                                       manifest=manifest, model_path=path, scene_plan=scene)
    with pytest.raises(support.DeploymentSupportError, match="task_kind_unsupported"):
        support.prepare_native_binding(_shortlist(), embodiment_id="universal_robots_ur5e",
                                       manifest=manifest, model_path=path,
                                       scene_plan={**scene, "task_spec": {"task_kind": "articulated_open_close"}})
    path.write_text(path.read_text() + "\n# changed")
    with pytest.raises(ValueError, match="model_digest_mismatch"):
        support.prepare_native_binding(_shortlist(), embodiment_id="universal_robots_ur5e",
                                       manifest=manifest, model_path=path, scene_plan=scene)


def test_dual_arm_never_silently_inherits_the_singular_gripper_runtime(tmp_path):
    request = _shortlist("i2rt_yam_bimanual", "dreamlabs_dream1")
    with pytest.raises(support.DeploymentSupportError, match="independent_bimanual_tool_channels_required"):
        support.prepare_native_binding(request, embodiment_id="i2rt_yam_bimanual",
                                       manifest={}, model_path=tmp_path / "absent.usd", scene_plan={})
