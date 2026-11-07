"""The team registry reads a navigation job's scene half; the rigid, family and pair halves read as before.

Part C of docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md. A ``blueprint.website_navigation_scene_inputs.v1``
record (``website_navigation_scene_inputs``) is a profile's fourth scene half. Real OpenUSD packet
compilation with the tiny G1-shaped robot the record's own tests use; no simulator, provider, network
or money.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import team_candidate_quick10_registry as registry
from blueprint_pipeline import website_navigation_scene_inputs as navigation_inputs
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION
from blueprint_pipeline.team_candidate_quick10_registry import (
    REGISTRY_ENV,
    REGISTRY_SCHEMA,
    TeamCandidateRegistryError,
    offer_eligibility,
    profile_for_capture,
    scene_facts,
    write_derived_profile,
)
from blueprint_pipeline.team_candidate_registry_reader import DERIVED_FIELDS, REQUIRED_DERIVED_FIELDS
from blueprint_pipeline.team_evaluation_plan import PURPOSES
from tests.test_website_navigation_scene_inputs import navigation_scene_inputs_arguments

HALVES = ("team_scene_inputs_path", "family_scene_inputs_path", "pair_runtime_inputs_path", "pair_activation_path")


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """The navigation record's writer arguments over the record's own fixture room."""
    return navigation_scene_inputs_arguments(tmp_path, monkeypatch)


def _navigation_profile(record_path: str, record: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"capture_root": "/captures/site-navigation/capture-1", "scene_id": record["scene_id"],
            "task_id": record["task_id"], "scenario_id": "nominal",
            "robot_free_probe_root": str(Path(record["robot_free_probe"]["path"]).parent),
            "scene_revision_digest": record["scene_revision_digest"], "navigation_scene_inputs_path": record_path,
            "open_loop_horizon": 8, **extra}


def _no_franka_cells(monkeypatch) -> None:
    """A navigation record never reaches the Franka DROID cell resolution."""
    from blueprint_pipeline import native_task_arena_policy_canary_worker as canary

    def refuse(*_args, **_kwargs):
        raise AssertionError("a navigation scene resolved a Franka DROID cell")

    monkeypatch.setattr(canary, "_resolved_scene_plan", refuse)


# --- the record's place in the registry -------------------------------------------------------


def test_the_navigation_half_is_a_derived_field_and_its_schema_is_the_writers() -> None:
    assert registry.NAVIGATION_SCENE_INPUTS_SCHEMA == navigation_inputs.SCHEMA
    assert DERIVED_FIELDS[-1] == "navigation_scene_inputs_path"
    assert REQUIRED_DERIVED_FIELDS == ("capture_root", "scene_id", "task_id", "scenario_id",
                                       "robot_free_probe_root", "scene_revision_digest")


def test_a_navigation_record_is_one_scene_half_alone() -> None:
    assert registry._one_scene_half({"navigation_scene_inputs_path": "/navigation.json"})
    for half in ("team_scene_inputs_path", "family_scene_inputs_path"):
        assert not registry._one_scene_half({"navigation_scene_inputs_path": "/navigation.json", half: "/x.json"})
    assert not registry._one_scene_half({"navigation_scene_inputs_path": "/navigation.json",
                                         "pair_runtime_inputs_path": "/p.json", "pair_activation_path": "/a.json"})
    assert not registry._one_scene_half({"navigation_scene_inputs_path": "/navigation.json",
                                         "pair_runtime_inputs_path": "/p.json"})


# --- scene facts ------------------------------------------------------------------------------


def test_scene_facts_read_a_navigation_record_without_a_franka_droid_plan(inputs, monkeypatch) -> None:
    written = navigation_inputs.write_navigation_scene_inputs(**inputs)
    record = written["record"]
    plan = json.loads((Path(record["packet_root"]) / "native_task_arena_scene_plan.v1.json").read_text())
    _no_franka_cells(monkeypatch)

    facts = scene_facts(_navigation_profile(written["path"], record))

    assert facts["inputs"] == navigation_inputs.read_navigation_scene_inputs(written["path"])
    assert facts["packet_dir"] == Path(record["packet_root"])
    assert record["scene_revision_digest"] == plan["plan_digest"]
    assert facts["scene"] == {"scene_revision_digest": plan["plan_digest"], "task_spec": plan["task_spec"],
                              "task_success_contract": record["task_success_contract"],
                              "embodiment_preset_id": "unitree_g1"}
    assert facts["scene"]["task_spec"]["task_kind"] == "goal_region_navigation"
    assert facts["robot_free_probe_root"] == Path(record["robot_free_probe"]["path"]).parent
    assert (facts["robot_free_probe_root"] / gate.RECEIPT_FILENAME).is_file()
    # The checkpoints' trained instruction, never the site's sentence.
    assert facts["prompt"] == PUBLISHED_TASK_INSTRUCTION == "Avoid obstacles and move to the yellow marked area."
    assert facts["runtime_source_packet_receipt"] is None and record["runtime_source"] is None
    assert facts["max_policy_queries"] is None and facts["open_loop_horizon"] == 8
    assert facts["episode_specs_sha256"] == canonical_digest(facts["inputs"])


def test_a_navigation_record_that_does_not_verify_refuses_by_its_readers_name(inputs, tmp_path) -> None:
    written = navigation_inputs.write_navigation_scene_inputs(**inputs)
    record = written["record"]
    tampered = {**record, "task_id": "another-task"}
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(TeamCandidateRegistryError, match="^website_navigation_scene_inputs_seal_invalid$"):
        scene_facts(_navigation_profile(str(path), record))
    with pytest.raises(TeamCandidateRegistryError, match="^website_navigation_scene_inputs_unreadable$"):
        scene_facts(_navigation_profile(str(tmp_path / "missing.json"), record))
    with pytest.raises(TeamCandidateRegistryError, match="^team_candidate_scene_half_ambiguous$"):
        scene_facts(_navigation_profile(written["path"], record, team_scene_inputs_path="/team.json"))


def _registry(tmp_path: Path, monkeypatch) -> Path:
    operator = tmp_path / "operator"
    operator.mkdir()
    value = {"schema_version": REGISTRY_SCHEMA, "profiles": [], "derived_profiles": {
        "root": str(operator / "derived"), "defaults": {"allowed_founder_approvers": ["owner-1"],
                                                        "max_hourly_rate_usd": "1.00", "open_loop_horizon": 8}}}
    value["registry_digest"] = canonical_digest(value, digest_field="registry_digest")
    path = operator / "registry.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    return operator


def test_a_derived_navigation_profile_is_written_read_back_and_never_listed_for_teams(
        inputs, tmp_path, monkeypatch) -> None:
    written = navigation_inputs.write_navigation_scene_inputs(**inputs)
    record = written["record"]
    _registry(tmp_path, monkeypatch)
    capture = tmp_path / "captures" / "capture-1"
    capture.mkdir(parents=True)
    facts = scene_facts(_navigation_profile(written["path"], record))
    row = {key: value for key, value in _navigation_profile(written["path"], record).items()
           if key != "open_loop_horizon"}
    row.update(capture_root=str(capture), robot_free_probe_root=str(facts["robot_free_probe_root"]))
    assert write_derived_profile(row)["status"] == "written"

    profile = profile_for_capture(capture)
    assert profile["navigation_scene_inputs_path"] == written["path"]
    assert profile["robot_free_probe_root"] == str(facts["robot_free_probe_root"])
    assert scene_facts(profile) == facts
    # A team evaluation runs a team's arm on a rigid task's cells; a navigation scene is the G1 lane's.
    for purpose in PURPOSES:
        assert offer_eligibility(profile, purpose) == (None, ["team_candidate_navigation_scene_not_a_team_evaluation"])


# --- beta planning's robot-free gate ----------------------------------------------------------


def _plan_screening(record_path: str, record: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
    from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs

    profile = _navigation_profile(record_path, record, capture_root=str(tmp_path / "capture"),
                                  max_hourly_rate_usd=0.5, screening_candidate_cap_usd=3)
    source = {"task_id": record["task_id"], "scene_revision_digest": record["scene_revision_digest"]}
    return build_planning_inputs(catalog=load_default_catalog(), profile=profile,
                                 capture_root=Path(profile["capture_root"]), source=source,
                                 inputs_dir=tmp_path / "screening-inputs")


def test_build_planning_inputs_robot_free_gate_admits_the_navigation_receipt(inputs, tmp_path, monkeypatch) -> None:
    """The real planner, from the profile through scene facts and its own gate call, to every candidate."""
    written = navigation_inputs.write_navigation_scene_inputs(**inputs)
    record = written["record"]
    _no_franka_cells(monkeypatch)

    planning = _plan_screening(written["path"], record, tmp_path)

    receipt = json.loads(Path(record["robot_free_probe"]["path"]).read_text())
    scene = planning["scene"]
    assert scene["task_kind"] == "goal_region_navigation"
    assert scene["robot_free_probe_status"] == "passed" and receipt["status"] == "passed"
    assert scene["robot_free_probe_digest"] == receipt["receipt_digest"] == record["robot_free_probe"]["receipt_digest"]
    assert scene["scene_revision_digest"] == record["scene_revision_digest"]
    sealed = json.loads(Path(scene["scene_inputs"]["path"]).read_text())
    assert sealed["beta_robot_free_probe_receipt"] == receipt and sealed["beta_task_kind"] == "goal_region_navigation"
    # With no G1 registration in this profile, nothing is bound or quoted; the G1 rows say why.
    capabilities = planning["capabilities"]
    assert capabilities["candidate_bindings"] == {} and planning["cost_quotes"] == {}
    assert {key: value for key, value in capabilities["candidate_blockers"].items() if key.startswith("humanoidarena_")} \
        == dict.fromkeys(("humanoidarena_dp_g1_dex3_sonic", "humanoidarena_pi05_g1_dex3_sonic",
                          "humanoidarena_dp_g1_dex3_sonic_vision_navi", "humanoidarena_pi05_g1_dex3_sonic_vision_navi"),
                         ["g1_beta_scene_packet_missing"])


def test_build_planning_inputs_holds_a_navigation_scene_whose_probe_failed(inputs, tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import native_g1_navigation_goal as navigation_goal

    real = navigation_goal.score_g1_navigation_episode

    def ever_reached(*, task_spec, samples):
        report = dict(real(task_spec=task_spec, samples=samples))
        if report["minimum_distance_m"] <= report["goal"]["acceptance_radius_m"]:
            report["outcome"] = "success"
        return report

    monkeypatch.setattr(navigation_goal, "score_g1_navigation_episode", ever_reached)
    written = navigation_inputs.write_navigation_scene_inputs(**inputs)
    assert written["record"]["robot_free_probe"]["status"] == "failed"
    with pytest.raises(gate.RobotFreeProbeGateError, match="^team_evaluation_robot_free_probe_not_passed$"):
        _plan_screening(written["path"], written["record"], tmp_path)


# --- the rigid, family and pair halves, byte for byte ------------------------------------------
# The three halves' readers as they stood before the navigation half (origin/main 69e48f8): their
# success paths as they were (a local renamed), the refusals left out since only what they return is
# compared. The current module is compared against them on real records.

_LEGACY_DERIVED_FIELDS = ("capture_root", "scene_id", "task_id", "scenario_id", "robot_free_probe_root",
                          "scene_revision_digest", "team_scene_inputs_path", "pair_runtime_inputs_path",
                          "pair_activation_path", "nominal_dynamic_friction", "family_scene_inputs_path")


def _legacy_one_scene_half(profile):
    pair = [bool(profile.get(key)) for key in ("pair_runtime_inputs_path", "pair_activation_path")]
    return (pair in ([True, True], [False, False])
            and sum((bool(profile.get('team_scene_inputs_path')), bool(profile.get('family_scene_inputs_path')),
                     pair == [True, True])) == 1)


def _legacy_scene_inputs(profile):
    if not _legacy_one_scene_half(profile):
        raise TeamCandidateRegistryError("team_candidate_scene_half_ambiguous")
    if profile.get('family_scene_inputs_path'):
        from blueprint_pipeline.website_task_family_scene_inputs import read_family_scene_inputs
        return read_family_scene_inputs(Path(profile['family_scene_inputs_path']))
    if profile.get("team_scene_inputs_path"):
        from blueprint_pipeline.team_scene_packet import read_team_scene_inputs
        return read_team_scene_inputs(Path(profile["team_scene_inputs_path"]))
    return registry.scene_inputs_from_frozen_pair(Path(profile["pair_runtime_inputs_path"]),
                                                  Path(profile["pair_activation_path"]))


def _legacy_scene_facts(profile):
    from blueprint_pipeline.adp009d_policy_episode import maximum_policy_queries_for_task_spec
    from blueprint_pipeline.native_task_arena_policy_canary_worker import _resolved_scene_plan

    half = _legacy_scene_inputs(profile)
    packet_dir = Path(half["base_native_packet"]["path"]).parent
    plan = registry._read(packet_dir / "native_task_arena_scene_plan.v1.json")
    if half.get('schema_version') == 'blueprint.task_family_scene_inputs.v1':
        from blueprint_pipeline.task_family_policy_execution import family_policy_execution_binding
        family_policy_execution_binding(plan)
        horizon = profile.get('open_loop_horizon')
        context = registry._read(Path(half['references']['task_context']['path']))
        return {'inputs': half, 'packet_dir': packet_dir,
            'runtime_source_packet_receipt': Path(half['runtime_source']['path']),
            'scene': {**plan, 'scene_revision_digest': half['scene_revision_digest'], 'task_spec': plan['task_spec'],
                'task_success_contract': half['task_success_contract'],
                'embodiment_preset_id': plan['native_runtime_contract']['robot_configuration']['embodiment_id']},
            'prompt': context.get('description'), 'max_policy_queries': None, 'open_loop_horizon': horizon,
            'episode_specs_sha256': canonical_digest(half)}
    nominal = _resolved_scene_plan(plan, {**registry.NOMINAL_CELL, "control_diagnostic": half["control_diagnostic"]},
                                   task_success_contract=half["task_success_contract"])
    horizon = profile.get("open_loop_horizon")
    return {
        "inputs": half, "packet_dir": packet_dir,
        "runtime_source_packet_receipt": Path(half["runtime_source"]["path"]),
        "scene": {"scene_revision_digest": half.get("scene_revision_digest"),
                  "task_spec": (registry.scored_task_spec(plan["task_spec"], half["task_success_contract"])
                                if (plan.get("task_spec") or {}).get("task_kind") == "rigid_pick_place"
                                else plan.get("task_spec")),
                  "task_success_contract": half["task_success_contract"],
                  "embodiment_preset_id": registry.DROID_POLICY_CANARY_PRESET_ID},
        "prompt": nominal["task_spec"]["prompt"],
        "max_policy_queries": maximum_policy_queries_for_task_spec(nominal["task_spec"], open_loop_horizon=horizon),
        "open_loop_horizon": horizon,
        "episode_specs_sha256": canonical_digest(half),
    }


def _bytes(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def test_every_legacy_scene_half_combination_is_decided_as_before() -> None:
    assert DERIVED_FIELDS[:-1] == _LEGACY_DERIVED_FIELDS
    for present in itertools.product((False, True), repeat=len(HALVES)):
        profile = {key: "/" + key for key, flag in zip(HALVES, present) if flag}
        assert registry._one_scene_half(profile) is _legacy_one_scene_half(profile), profile


def test_rigid_family_and_pair_scene_facts_are_byte_for_byte_unchanged(tmp_path, monkeypatch) -> None:
    from tests.test_provider_runtime_import_closure import _real_canary_pair_inputs
    from tests.test_team_candidate_quick10_launch import _team_scene_record
    from tests.test_website_task_family_scene_inputs import source_fixture

    def never(*_args, **_kwargs):
        raise AssertionError("a rigid, family or pair half read a navigation record")

    monkeypatch.setattr(navigation_inputs, "read_navigation_scene_inputs", never)
    monkeypatch.setattr(registry, "_navigation_scene_facts", never)
    _packet, _runtime, pair_inputs, pair_activation = _real_canary_pair_inputs(tmp_path / "pair")
    packet, runtime, team_inputs, _activation = _real_canary_pair_inputs(tmp_path / "rigid")
    team_half = _team_scene_record(packet, runtime, json.loads(team_inputs.read_text()))
    (tmp_path / "family").mkdir()
    source_fixture(tmp_path / "family", monkeypatch)
    family = profile_for_capture(tmp_path / "family")
    assert family["family_scene_inputs_path"]
    profiles = {
        "pair": {"pair_runtime_inputs_path": str(pair_inputs), "pair_activation_path": str(pair_activation),
                 "open_loop_horizon": 8},
        "rigid": {"team_scene_inputs_path": str(team_half), "open_loop_horizon": 8},
        "family": family,
    }
    for name, profile in profiles.items():
        assert _bytes(registry._scene_inputs(profile)) == _bytes(_legacy_scene_inputs(profile)), name
        facts = registry.scene_facts(profile)
        assert _bytes(facts) == _bytes(_legacy_scene_facts(profile)), name
        assert "robot_free_probe_root" not in facts, name
    assert registry.scene_facts(profiles["rigid"])["scene"]["task_spec"]["task_kind"] == "rigid_pick_place"
