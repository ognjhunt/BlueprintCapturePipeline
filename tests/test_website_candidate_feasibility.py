"""Measured reach decides infeasibility; the intake ranking orders the plan; the budget decides how far it runs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import team_embodiment
from blueprint_pipeline.beta_policy_candidate_catalog import CATALOG_ENV, load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan, validate_screening_plan
from blueprint_pipeline.beta_policy_screening_team import build_registered_team_spec, team_policy_interface
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture, scene_facts
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from blueprint_pipeline.website_candidate_feasibility import candidate_feasibility
from tests.test_beta_policy_screening_coordinator import Backend, plan as coordinator_plan, seal
from tests.test_beta_policy_screening_launch import _plan
from tests.test_beta_policy_screening_plan import NOW, inputs, scene
from tests.test_team_embodiment import manifest, write_robot_usd

SHORT_REACH_M = 0.2


def registered(tmp_path, monkeypatch, *, count=3, short=(1,), priority=()):
    """Real team robots registered on the reconstructed scene packet; ``short`` ones declare a 0.2 m reach."""
    base, facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    model = write_robot_usd(tmp_path / "robot.usda")
    declared = validate_embodiment_manifest(manifest(model, control_hz=60))
    paths = {reach: launch._write(tmp_path / f"robot-{reach}.manifest.json",
                                  validate_embodiment_manifest(manifest(model, control_hz=60, reach_m=reach)))
             for reach in (declared["reach_m"], SHORT_REACH_M)}
    credential = tmp_path / "operator" / "team-bearer"
    credential.write_text("private-team-token")
    credential.chmod(0o600)
    registrations, rows = {}, []
    for index in range(count):
        key = f"team_policy_{index:02d}"
        rights = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
                  "team_id": "test-team", "checkpoint_id": key, "candidate_id": key, "license_to_run_granted": True,
                  "policy_self_grading_permitted": False, "official_ranking_permitted": False,
                  "secret_material_recorded": False, "adp_qualified_evidence_permitted": False}
        rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
        registration = {"manifest": launch._record(paths[declared["reach_m"]]), "model": launch._record(model),
            "team_id": "test-team", "checkpoint_id": key, "rights_binding": rights,
            "policy_spec": {"policy_id": key, "policy_interface_id": "blueprint_policy_https.v1",
                "endpoint_url": "https://policy.example.com/infer", "metadata_url": "https://policy.example.com/metadata",
                "allowed_origin": "https://policy.example.com", "model_identity_digest": canonical_digest({"policy": key}),
                "embodiment_id": declared["embodiment_id"], "action_chunk_rows": 4, "open_loop_horizon": 4,
                "camera_names": [camera["name"] for camera in declared["cameras"]]}}
        spec, _ = build_registered_team_spec(candidate_id=key, registration=registration,
            scene_plan=facts["scene"]["native_scene_plan"], contract=facts["scene"]["task_spec"]["task_success_contract"],
            prompt=scene_facts(profile)["prompt"], scene_revision_digest=facts["scene"]["scene_revision_digest"])
        if index in short:
            registration["manifest"] = launch._record(paths[SHORT_REACH_M])
        row = deepcopy(load_default_catalog()["candidates"][0])
        row.update(candidate_id=key, display_name=key, interface=team_policy_interface(spec))
        row["runtime"]["minimum_gpu_memory_gb"] = 48
        row["policy"].update(kind="hosted", source="https://policy.example.com", revision="test-release-1",
                             digest=spec["checkpoint_digest"], digest_kind="provider_metadata")
        row["eligibility"].update(commercial_use="allowed", requires_credentials=["TEAM_POLICY_CREDENTIAL"])
        rows.append(row)
        registrations[key] = {"robot_policy": registration, "credential_file": str(credential),
            "credential_origin": "https://policy.example.com", "provider_hard_cap_usd": "1.00",
            "hosted_inference_maximum_usd": "0.00", "hosted_inference_payment": "caller_paid",
            "hosted_max_cost_per_request_usd": "0"}
    catalog = seal_catalog({**load_default_catalog(), "candidates": rows})
    monkeypatch.setenv(CATALOG_ENV, str(launch._write(tmp_path / "team-catalog.json", catalog)))
    profile.update(screening_candidates=registrations, screening_hard_ttl_seconds=1800,
                   screening_candidate_cap_usd="1.00")
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
        source=base["source"], inputs_dir=tmp_path / "team-inputs", priority=priority)
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
        cost_quotes=planning["cost_quotes"], campaign_id="team-screening", priority=planning["candidate_priority"])
    return plan, planning, profile, catalog


def test_infeasible_comes_only_from_measured_station_reach_and_stays_listed(tmp_path, monkeypatch):
    measured = []
    original = team_embodiment.station_reach_blockers

    def spy(scene_plan, robot):
        measured.append((scene_plan.get("plan_digest"), robot["reach_m"]))
        return original(scene_plan, robot)

    monkeypatch.setattr(team_embodiment, "station_reach_blockers", spy)
    plan, planning, _, _ = registered(tmp_path, monkeypatch)
    rows = {row["candidate_id"]: row for row in plan["candidates"]}
    assert rows["team_policy_01"]["status"] == "infeasible"
    assert rows["team_policy_01"]["blockers"] == ["team_embodiment_station_out_of_reach"]
    assert rows["team_policy_00"]["status"] == rows["team_policy_02"]["status"] == "selected"
    feasibility = planning["candidate_feasibility"]
    scene_digest = planning["scene"]["native_scene_plan"]["plan_digest"]
    assert feasibility["scene_plan_digest"] == scene_digest
    assert {key: row["status"] for key, row in feasibility["candidates"].items()} == {
        "team_policy_00": "feasible", "team_policy_01": "infeasible", "team_policy_02": "feasible"}
    # Every registered body was measured on this reconstructed scene's packet; none was inferred.
    assert (scene_digest, SHORT_REACH_M) in measured and (scene_digest, 3.0) in measured
    retained = list((tmp_path / "team-inputs").glob("*.candidate_feasibility.json"))
    assert [json.loads(path.read_text()) for path in retained] == [feasibility]
    assert "team_policy_01" not in planning["cost_quotes"]
    assert validate_screening_plan(plan) == plan


def test_unregistered_bodies_stay_unassessed_never_infeasible(tmp_path, monkeypatch):
    plan, facts = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    feasibility = candidate_feasibility(scene_plan=facts["scene"]["native_scene_plan"], catalog=load_default_catalog(),
                                        profile=profile_for_capture(str(tmp_path / "capture")))
    # A job with no registered team body (Franka-only) keeps exactly the rows it had: nothing is added.
    assert all(row == {"status": "unassessed", "blockers": ["embodiment_reach_manifest_not_registered"]}
               for row in feasibility["candidates"].values())
    assert facts["capabilities"]["candidate_infeasible"] == {}
    assert "infeasible" not in {row["status"] for row in plan["candidates"]}
    assert len(plan["candidates"]) == len(load_default_catalog()["candidates"])


def test_each_body_is_measured_from_its_own_station_and_an_unmodelled_one_stays_unassessed(tmp_path):
    from blueprint_pipeline.team_embodiment import (
        apply_team_embodiment,
        read_model_structure,
        team_embodiment_binding,
        verify_model_file,
    )
    from blueprint_pipeline.team_scene_packet import materialize_team_scene
    from blueprint_pipeline.website_candidate_feasibility import (
        infeasible_blockers, own_stance, own_station, station_scene_plan)
    from tests.test_team_scene_packet import _scene, bodies

    scene = _scene(tmp_path, strategy="pick_and_place")
    body = bodies(tmp_path)
    compiled = materialize_team_scene(
        **{key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")},
        output_root=tmp_path / "team-scene", embodiment_manifests=list(body.values()))
    record = json.loads(Path(compiled["path"]).read_text())
    scene_plan = json.loads((Path(record["base_native_packet"]["path"]).parent
                             / "native_task_arena_scene_plan.v1.json").read_text())

    def registered(name):
        path = launch._write(tmp_path / f"{name}.manifest.json", body[name])
        return {"robot_policy": {"manifest": launch._record(path)}}

    profile = {"team_scene_inputs_path": compiled["path"], "screening_candidates": {
        "long_arm": registered("long"), "short_arm": registered("short"), "rover": registered("wheeled"),
        "g1_humanoid": {"native_g1": {"registered": "with its own scene packet"}}}}
    catalog = {"candidates": [{"candidate_id": key}
                              for key in ("long_arm", "short_arm", "rover", "g1_humanoid", "pi05_droid")]}
    feasibility = candidate_feasibility(scene_plan=scene_plan, catalog=catalog, profile=profile)
    rows, own = feasibility["candidates"], record["embodiment_stations"]
    long = rows["long_arm"]
    # The status is measured where the run stands: the body's own station, which its binding carries;
    # the packet's station is kept beside it for comparison.
    assert (long["status"], long["blockers"], long["station_basis"]) == ("feasible", [], "embodiment_own_station")
    assert long["scene_packet_station"]["status"] in {"feasible", "infeasible"}
    assert long["own_station"]["station"] == own[body["long"]["embodiment_digest"]]["station"]
    assert (long["own_station"]["status"], long["own_station"]["station_basis"]) == ("feasible", "embodiment_own_station")
    # The short arm cannot reach from its own best station either: infeasible, by name, in both records.
    assert rows["short_arm"]["own_station"]["status"] == "infeasible"
    assert rows["short_arm"]["status"] == "infeasible"
    assert infeasible_blockers(feasibility) == {"short_arm": ["team_embodiment_station_out_of_reach"]}
    # A free base has its own floor station but declares no reach; a body without a manifest is not modelled.
    assert rows["rover"]["status"] == "unassessed"
    assert rows["rover"]["blockers"] == ["embodiment_free_base_reach_not_modeled"]
    assert rows["rover"]["own_station"]["station"] == own[body["wheeled"]["embodiment_digest"]]["station"]
    for key in ("g1_humanoid", "pi05_droid"):
        assert rows[key] == {"status": "unassessed", "blockers": ["embodiment_reach_manifest_not_registered"]}
    # The run built from the station plan holds the long arm where it was measured.
    placed = station_scene_plan(scene_plan, long)
    assert placed["robot"]["base_pose_world"]["position_world_m"] == long["own_station"]["station"]["position_world_m"]
    assert placed["plan_digest"] == canonical_digest(placed, digest_field="plan_digest")
    assert station_scene_plan(scene_plan, rows["rover"]) == scene_plan
    model = tmp_path / "robot.usda"
    binding = team_embodiment_binding(body["long"], model_record=verify_model_file(body["long"], model),
                                      structure=read_model_structure(model), scene_plan=placed)
    run = apply_team_embodiment(placed, binding)
    assert run["robot"]["base_mode"] == "fixed_at_station"
    assert run["robot"]["base_pose_world"] == placed["robot"]["base_pose_world"]
    # The run's binding carries the station: every cell of the packet's own plan stands the body there.
    station = own_station(long)
    assert station == {key: long["own_station"]["station"][key] for key in ("position_world_m", "orientation_xyzw")}
    bound = team_embodiment_binding(body["long"], model_record=verify_model_file(body["long"], model),
                                    structure=read_model_structure(model), scene_plan=scene_plan, station=station)
    assert bound["station"] == station
    assert apply_team_embodiment(scene_plan, bound)["robot"]["base_pose_world"] == station
    assert own_station(rows["rover"]) is None and own_station(rows["short_arm"]) is not None
    # A free base is placed by its own floor stance, the one its screening run binds.
    assert own_stance(rows["rover"]) == own[body["wheeled"]["embodiment_digest"]]["station"]
    assert "floor_z_m" in own_stance(rows["rover"]) and own_stance(long) is None
    # A scene compiled for another plan is not read for this one: the packet's own station is measured, and said.
    other = candidate_feasibility(scene_plan={**scene_plan, "plan_digest": "sha256:" + "0" * 64}, catalog=catalog,
                                  profile=profile)
    assert other["candidates"]["long_arm"]["station_basis"] == "scene_packet_station"
    assert "own_station" not in other["candidates"]["long_arm"]


def test_priority_orders_registered_robots_and_the_budget_decides_how_far(tmp_path, monkeypatch):
    priority = ["team_policy_15", "team_policy_01", "not_in_catalog", "team_policy_07"]
    plan, planning, _, _ = registered(tmp_path, monkeypatch, count=16, short=(), priority=priority)
    order = [row["candidate_id"] for row in plan["candidates"]]
    assert planning["candidate_priority"] == ["team_policy_15", "team_policy_01", "team_policy_07"]
    assert order[:3] == ["team_policy_15", "team_policy_01", "team_policy_07"]
    assert order[3:] == sorted(order[3:]) and len(order) == 16
    # Fifteen dollars covers fifteen one-dollar candidates; the lowest priority is deferred with its reason.
    assert [row["status"] for row in plan["candidates"]].count("selected") == 15
    assert plan["candidates"][-1]["candidate_id"] == "team_policy_14"
    assert plan["candidates"][-1]["blockers"] == ["site_screening_budget_insufficient"]


def test_budget_is_spent_in_priority_order_and_every_candidate_keeps_a_status():
    catalog, capabilities, quotes = inputs(40, "2.00")
    ids = [row["candidate_id"] for row in catalog["candidates"]]
    priority = list(reversed(ids))
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                 cost_quotes=quotes, now=NOW, priority=priority)
    assert [row["candidate_id"] for row in value["candidates"]] == priority
    selected = [row["candidate_id"] for row in value["candidates"] if row["status"] == "selected"]
    assert selected == priority[:7] and value["maximum_planned_cost_usd"] == "14.00"
    assert all(row["blockers"] == ["site_screening_budget_insufficient"] for row in value["candidates"][7:])
    default = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                   cost_quotes=quotes, now=NOW)
    assert [row["candidate_id"] for row in default["candidates"]] == sorted(ids)


def test_unranked_learned_policies_are_funded_before_reference_only_scripted_baselines():
    """2026-10-31: 23 scripted baselines sort first by identifier and would take every funded slot."""
    catalog, capabilities, quotes = inputs(8, "4.00")
    rows = deepcopy(catalog["candidates"])
    for row in rows[:4]:
        row["policy"]["kind"] = "scripted"
    catalog = seal_catalog({**catalog, "candidates": rows})
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                 cost_quotes=quotes, now=NOW)
    order = [row["candidate_id"] for row in value["candidates"]]
    scripted = {row["candidate_id"] for row in rows[:4]}
    assert order == sorted(set(order) - scripted) + sorted(scripted)
    funded = [row["candidate_id"] for row in value["candidates"] if row["status"] == "selected"]
    assert funded and not scripted & set(funded)
    # An intake ranking still decides first, scripted or not.
    ranked = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                  cost_quotes=quotes, now=NOW, priority=[rows[0]["candidate_id"]])
    assert ranked["candidates"][0]["candidate_id"] == rows[0]["candidate_id"]


def test_infeasible_and_unavailable_candidates_are_published_with_their_reasons(tmp_path):
    catalog, capabilities, quotes = inputs()
    capabilities["candidate_infeasible"] = {"groot_n17_droid": ["team_embodiment_station_out_of_reach"]}
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                 cost_quotes=quotes, now=NOW, priority=["groot_n17_droid", "pi05_droid"])
    rows = {row["candidate_id"]: row for row in value["candidates"]}
    assert rows["groot_n17_droid"] == {"candidate_id": "groot_n17_droid", "display_name": rows["groot_n17_droid"][
        "display_name"], "configuration_digest": rows["groot_n17_droid"]["configuration_digest"],
        "status": "infeasible", "blockers": ["team_embodiment_station_out_of_reach"]}
    assert rows["pi05_droid"]["status"] == "selected"
    assert {row["status"] for row in value["candidates"]} == {"selected", "infeasible", "unavailable"}
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/beta_policy_screening_plan.v1.schema.json").read_text())
    Draft202012Validator(schema).validate(value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    [published] = coordinator.pending_publications(queue_root=tmp_path)
    shown = {row["candidate_id"]: row for row in published["candidates"]}
    assert set(shown) == set(rows)
    assert shown["groot_n17_droid"]["status"] == "unavailable"
    assert shown["groot_n17_droid"]["blockers"] == [coordinator.INFEASIBLE_BLOCKER, "team_embodiment_station_out_of_reach"]
    assert all(shown[key]["blockers"] for key, row in rows.items() if row["status"] != "selected")


def test_coordinator_starts_new_work_in_the_plan_priority_order(tmp_path):
    value = coordinator_plan(count=3, cost="1.00")
    value["candidates"] = [value["candidates"][2], value["candidates"][0], value["candidates"][1]]
    value = validate_screening_plan(seal(value, "plan_digest"))
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    launched = []

    class Ordered(Backend):
        def launch_candidate(self, **kwargs):
            launched.append(kwargs["candidate"]["candidate_id"])
            return super().launch_candidate(**kwargs)

    for _ in range(3):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path,
                                            executor=Ordered(), max_launches=1)
    assert launched == ["candidate-002", "candidate-000", "candidate-001"]


def test_a_free_base_whose_own_arm_reached_from_its_start_stance_is_feasible_there(tmp_path, monkeypatch):
    from blueprint_pipeline import team_scene_packet
    from tests.test_team_scene_packet import bodies

    wheeled = bodies(tmp_path)["wheeled"]
    path = launch._write(tmp_path / "wheeled.manifest.json", wheeled)
    stance = {"position_world_m": [1.0, 0.0, 0.9], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "floor_z_m": 0.0}

    def recorded(status):
        return {wheeled["embodiment_digest"]: {
            "base_mode": "free", "station": stance, "blockers": [], "unassessed": [],
            "reach": {"rule": "closest_clear_stance_own_arm_reaches_every_phase", "own_arm": {"status": status}}}}
    profile = {"team_scene_inputs_path": str(tmp_path / "inputs.json"),
               "screening_candidates": {"rover": {"robot_policy": {"manifest": launch._record(path)}}}}
    for status, expected in (("passed", ("feasible", [])),
                             ("flagged", ("unassessed", ["embodiment_free_base_reach_not_modeled"]))):
        monkeypatch.setattr(team_scene_packet, "recorded_embodiment_stations",
                            lambda *_, status=status, **__: recorded(status))
        [row] = candidate_feasibility(scene_plan={"plan_digest": "sha256:" + "a" * 64},
                                      catalog={"candidates": [{"candidate_id": "rover"}]},
                                      profile=profile)["candidates"].values()
        assert (row["status"], row["blockers"]) == expected
        assert row["own_station"]["station"] == stance and row["own_station"]["reach"]["own_arm"]["status"] == status


def test_only_an_own_station_the_bodys_arm_reached_from_lifts_the_declared_reach_at_binding() -> None:
    from blueprint_pipeline.website_candidate_feasibility import OWN_STATION, own_station_reached_by_own_arm

    station = {"position_world_m": [1.0, 0.0, 0.75], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

    def row(rule, status, *, basis=OWN_STATION, base_mode="fixed"):
        return {"station_basis": basis, "own_station": {
            "station": station, "station_basis": basis, "base_mode": base_mode,
            "reach": {"rule": rule, "own_arm": {"status": status}}}}
    reached = "closest_clear_stance_own_arm_reaches_every_phase"
    assert own_station_reached_by_own_arm(row(reached, "passed")) is True
    assert own_station_reached_by_own_arm(row(reached, "flagged")) is False
    assert own_station_reached_by_own_arm(row("closest_clear_stance_within_declared_reach", "flagged")) is False
    assert own_station_reached_by_own_arm(row(reached, "passed", basis="scene_packet_station")) is False
    assert own_station_reached_by_own_arm(row(reached, "passed", base_mode="free")) is False
    assert own_station_reached_by_own_arm({}) is False
