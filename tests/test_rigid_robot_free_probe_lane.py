"""The capped GPU lane that records a prepared rigid scene's robot-free probe, proven without spend.

Beta journey step: L5 team policy evaluation. Owner approval 2026-10-26: under
each capture's website grant, build the scene and run the no-robot check
automatically; the first real run is a capped test.

Real here: the rigid task and its sealed success contract (the rigid relocation
adapter and the contract sealer), the scene packet (the canary fixture carrying
that task), the sealed probe bundle and the Vast adapter's bundle preflight, the
canonical allocator, the lane, the website-grant reservation client, the queue
runner, ingest, scoring, retention and the gate. The GPU host runs the real
probe worker and the real ``record_arena_robot_free_probe``.

Faked, and only these: the team registry read (one profile), the WebApp's
reply, the launch-environment probes, the offer POST, the Vast transport, and
Isaac itself, played by the rigid probe's deterministic flat-support stand-in
(``FlatSupport``). No GPU, provider, network or spend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import adp_isaac_lab_arena_vast as arena_vast
from blueprint_pipeline import agent_execution_offer
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import rigid_robot_free_probe_lane as lane
from blueprint_pipeline import rigid_robot_free_probe_runner as runner
from blueprint_pipeline import rigid_robot_free_probe_worker as worker
from blueprint_pipeline import robot_arm_path_clearance as arm
from blueprint_pipeline import scene_physics_preflight as physics
from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import robot_free_readiness_retention as retention
from blueprint_pipeline import team_candidate_quick10_registry as registry
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_task_arena_execution_contract import native_task_arena_execution_transport_completed
from blueprint_pipeline.native_task_arena_readback import _compose_pose_xyzw
from blueprint_pipeline.native_task_nurec_render_setup import appearance_render_path_from_plan
from blueprint_pipeline.rigid_robot_free_probe_bundle import PROBE_KIND, current_probe_state
from blueprint_pipeline.team_embodiment_runtime import record_arena_robot_free_probe
from blueprint_pipeline.team_robot_free_probe import scored_task_spec
from tests.test_provider_runtime_import_closure import _real_canary_pair_inputs
from tests.test_rigid_robot_free_probe import FlatSupport, _sealed_contract, rigid_case

REVISION = "sha256:" + "4" * 64
COMMIT = "a" * 40
PLAN = "native_task_arena_scene_plan.v1.json"
PACKET_RECEIPT = "native_task_arena_packet_receipt.v1.json"
UNREACHABLE = "robot_rigid_pick_place_unreachable:approach"


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def retask_packet(packet: Path, task_spec: dict[str, Any]) -> None:
    """The fixture packet, compiled for this rigid task: the plan's task and the resealed receipt."""

    plan_path = packet / PLAN
    plan = json.loads(plan_path.read_text())
    plan["task_spec"] = task_spec
    # Its task object spawns where the task starts, as every compiled packet's does (rigid_support_seating).
    offset = task_spec["interaction_affordance"]["asset_root_from_scoring_frame"]["position_m"]
    root = {"position_world_m": [a - b for a, b in zip(task_spec["start_pose_world"][:3], offset, strict=True)],
            "orientation_xyzw": list(task_spec["start_pose_world"][3:])}
    for row in plan["objects"]:
        if row.get("task_subject") is True:
            row["pose_world"] = copy.deepcopy(root)
            row.setdefault("reset_state", {})["root_pose_world"] = copy.deepcopy(root)
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    receipt_path = packet / PACKET_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    data = plan_path.read_bytes()
    for artifact in receipt["artifacts"]:
        if artifact["relative_path"] == PLAN:
            artifact.update(size_bytes=len(data), sha256="sha256:" + hashlib.sha256(data).hexdigest())
    receipt["arena_scene_plan_digest"] = plan["plan_digest"]
    receipt_path.write_text(json.dumps(_sealed(receipt, "receipt_digest"), indent=2, sort_keys=True) + "\n")


def _unreachable_from_the_packets_own_base(packet: Path) -> dict[str, Any]:
    """The arm-path solver's verdict for this packet's own Franka, at the packet's own base pose, picking
    its object up and placing it at its target.

    The fixture's task is a planar push the gate does not measure, and its base stands about 3.4 m from
    the object, so the real solver answers the refusal capture-447ae4f4's probe was sealed with.
    """
    plan = json.loads((packet / PLAN).read_text())
    spec = plan["task_spec"]
    with pytest.MonkeyPatch.context() as patch:
        patch.delitem(sys.modules, "torch", raising=False)  # the lane fixture's stand-in, not scipy's torch
        return arm.rigid_pick_place_clearance(
            robot_id=plan["robot"]["robot_id"], base_pose_world=plan["robot"]["base_pose_world"],
            start_pose_world=spec["start_pose_world"], target_position_world=spec["target_position_world_m"],
            contact_point_m=[0.0, 0.0, 0.0], approach_unit=[0.0, 0.0, 1.0],
            gripper_orientation_xyzw=[1.0, 0.0, 0.0, 0.0], pregrasp_clearance_m=0.1,
            seeds=[list(arm.NOMINAL_SEED_RAD)])


def website_context() -> dict[str, Any]:
    return _sealed({"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
                    "capture_id": "walkthrough-req1", "purpose": "scene_preparation",
                    "description": "Move the mug onto the tray",
                    "confirmed": True, "confirmed_at": "2026-10-26T01:00:00Z", "operator_answers": {},
                    "unresolved": [], "capture_rights": {"derived_scene_generation_allowed": True,
                                                         "consent_revoked": False}}, "context_digest")


GRANT_OWNER = {"user_id": "blueprint-preparation", "organization_id": "blueprint"}


def website_sponsorship(context: dict[str, Any], *, expires_in: float = 20 * 3600,
                        renewal: dict[str, Any] | None = None) -> dict[str, Any]:
    """A preparation grant as the WebApp's signed read returns it (synthetic caps and terms)."""

    expires = time.time() + expires_in
    value = {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint", "purpose": "scene_preparation",
             **{key: context[key] for key in ("request_id", "scene_id", "capture_id")},
             "task_context_digest": context["context_digest"], "policy_digest": "sha256:" + "8" * 64,
             "authoring_provider": "openai", "owner": dict(GRANT_OWNER), "preparation_max_total_spend_usd": 25,
             "upstream_max_spend_usd": 5, "max_total_spend_usd": 20, "max_paid_attempts": 2,
             "expires_at_epoch": expires,
             "consent": {"rights_reference": cross_runtime_canonical_digest(context["capture_rights"]),
                         "provider_terms_reference": "sha256:" + "7" * 64, "accepted_by": GRANT_OWNER["user_id"],
                         "accepted_at_epoch": expires - 86_000, "private_processing_authorized": True,
                         "provider_training_authorized": False, "task_confirmed": context["confirmed"],
                         "spend_authorized": True}}
    if renewal is not None:
        value["renewal"] = renewal
    return _sealed(value, "authority_digest")


class _FakeTorch(types.ModuleType):
    float32 = "float32"

    def __init__(self) -> None:
        super().__init__("torch")

    @staticmethod
    def tensor(data: Any, dtype: Any = None, device: Any = None) -> list[list[float]]:
        return [[float(value) for value in row] for row in data]

    @staticmethod
    def zeros(shape: Any, dtype: Any = None, device: Any = None) -> list[list[float]]:
        return [[0.0] * int(shape[1]) for _ in range(int(shape[0]))]


class _Host:
    """Isaac played by the flat-support stand-in: the object's state, gravity, and the robot never moving."""

    def __init__(self, plan: dict[str, Any], *, support_max_xy: Any = None, readback_fails: bool = False):
        spec = plan["task_spec"]
        self.offset = spec["interaction_affordance"]["asset_root_from_scoring_frame"]
        self.start = list(spec["start_pose_world"])
        self.support = FlatSupport(self.start, **({"support_max_xy": support_max_xy} if support_max_xy else {}))
        self.readback_fails = readback_fails
        self.resets: list[int] = []
        self.closed = False
        host = self

        class Object:
            def write_root_pose_to_sim(self, tensor: Any) -> None:
                root = [float(value) for value in tensor[0]]
                host.support.set_scoring_pose(_compose_pose_xyzw(
                    root, [*host.offset["position_m"], *host.offset["orientation_xyzw"]]))

            def write_root_velocity_to_sim(self, tensor: Any) -> None:
                assert tensor == [[0.0] * 6]

        class Scene(dict):
            def write_data_to_sim(self) -> None:
                return None

            def update(self, dt: float) -> None:
                assert abs(dt * 120 - 1) < 1e-9

        class Sim:
            def step(self, render: bool = True) -> None:
                assert render is False  # the probe never renders
                host.support.step()

        class Unwrapped:
            physics_dt = 1.0 / 120
            device = "cuda:0"
            scene = Scene(task_object=Object())
            sim = Sim()

        class Env:
            unwrapped = Unwrapped()

            def reset(self, seed: int) -> None:
                host.resets.append(seed)
                host.support.set_scoring_pose(host.start)

            def step(self, action: Any) -> None:
                raise AssertionError("the robot was commanded")

            def close(self) -> None:
                host.closed = True

        self.built = types.SimpleNamespace(env=Env(), scene_asset_names={"task_object": "task_object"}, plan=plan)

    def read_task_sample(self) -> dict[str, Any]:
        if self.readback_fails:
            raise RuntimeError("native_task_arena_scene_readback_missing")
        sample = self.support.read_sample()
        return {"task_scoring_pose_world": sample["task_scoring_pose_world"],
                "task_support_contact_peak_force_n": 5.0 if sample["support_contact_active"] else 0.0,
                "task_scene_collision_peak_force_n": 0.0}


def host_runtime(world: dict[str, Any]) -> worker.ProbeRuntime:
    def launch(path: Path, *, device: str, appearance_render_path: str) -> tuple[Any, dict[str, Any]]:
        world["launches"].append({"device": device, "render_path": appearance_render_path})
        return types.SimpleNamespace(close=lambda: None), {"status": "qualified", "render_path": appearance_render_path}

    def build(plan: dict[str, Any], *, device: str, bundle_root: Path, preconstruction_receipt: Any,
              enable_cameras: bool) -> Any:
        assert (Path(bundle_root) / PLAN).is_file()
        assert enable_cameras is False  # the probe reads object state, never an image
        if world["build_error"] is not None:
            raise world["build_error"]
        world["host"] = _Host(plan, **world["host_options"])
        return world["host"].built

    return worker.ProbeRuntime(
        device="cuda:0", resolve_render_path=appearance_render_path_from_plan, launch_isaac=launch,
        prepare_preconstruction=lambda *, expected_device: {"passed": True, "expected_device": expected_device},
        build_environment=build, read_device_binding=lambda built, *, expected_device: {"passed": True},
        make_rigid_task_readback=lambda built: world["host"], record_probe=record_arena_robot_free_probe)


def fake_transport(world: dict[str, Any]):
    """The Vast transport and GPU host: the real worker on the sealed bundle's bytes, then teardown."""

    def run(**kwargs: Any) -> dict[str, Any]:
        world["transport_calls"].append(kwargs)
        if not kwargs["execute"]:
            return {"status": "dry_run_ready", "provider_mutations_performed": 0, "blockers": []}
        assert kwargs["paid_resource_admission_grant"] is not None
        bundle = kwargs["prepared_bundle"]
        attempt_root = Path(kwargs["job_dir"]) / "attempts" / "attempt_001"
        host_dir = attempt_root / "host"
        with zipfile.ZipFile(bundle["bundle_path"]) as archive:
            archive.extractall(host_dir)
        output = attempt_root / "immutable_execution"
        result, _app = worker.run_rigid_robot_free_probe(runtime_dir=host_dir / "provider_runtime",
                                                         output_root=output, runtime=host_runtime(world))
        completed = native_task_arena_execution_transport_completed(
            result, expected_output_filename=worker.RESULT_FILENAME)
        return {"schema_version": kwargs["result_schema_version"], "status": "completed" if completed else "blocked",
                "bundle_sha256": bundle["bundle_sha256"], "native_control_result_path": str(output / worker.RESULT_FILENAME),
                "native_control_result_digest": result["result_digest"], "vast_instance_ids": [4242],
                "provider_mutations_performed": 2, "provider_create_attempted": True,
                "continuing_spend_from_this_run": world["continuing_spend"], "all_staged_objects_absent": True,
                "estimated_cost_usd": 0.31, "candidate_policy_queried": result["candidate_policy_queried"],
                "blockers": [] if completed else [*world["transport_blockers"], *(
                    list(result["blockers"]) or ["rigid_robot_free_probe_runtime_not_completed"])]}

    return run


def build_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **host_options: Any) -> dict[str, Any]:
    (tmp_path / "case").mkdir(parents=True)
    _configured, _references, _docs, spec = rigid_case(tmp_path / "case")
    # The shared revision helper now returns its scored spec. This fixture
    # starts with the revision's raw spec, then independently seals the job's
    # owner contract below. A contract from the helper's original site cannot
    # follow the fixture's reassigned site/task identity.
    spec = {key: value for key, value in spec.items()
            if key not in {"task_success_contract", "task_success_contract_digest"}}
    spec = dict(spec, site_id="site-1", task_id="task-1")
    contract = _sealed_contract(spec)
    packet, runtime_receipt, pair_inputs, _activation = _real_canary_pair_inputs(tmp_path / "scene")
    retask_packet(packet, spec)
    control = json.loads(pair_inputs.read_text())["cells"][0]["control_diagnostic"]
    capture = tmp_path / "captures" / "walkthrough-req1"
    (capture / "pipeline").mkdir(parents=True)
    context = website_context()
    (capture / "pipeline" / "website_task_context.json").write_text(json.dumps(context))
    # The copy written once when preparation started; the precheck reads the WebApp's current grant instead.
    (capture / "pipeline" / "website_scene_sponsorship.json").write_text(json.dumps(website_sponsorship(context)))
    probe_root = tmp_path / "source-launch"
    probe_root.mkdir()
    profile = {"capture_root": str(capture.resolve()), "robot_free_probe_root": str(probe_root.resolve()),
               "scene_id": "site-req1", "task_id": "task-1", "scenario_id": "nominal", "max_hourly_rate_usd": 0.8}
    facts = {"inputs": {"task_success_contract": contract, "control_diagnostic": control}, "packet_dir": packet,
             "runtime_source_packet_receipt": runtime_receipt,
             "scene": {"scene_revision_digest": REVISION, "task_spec": scored_task_spec(spec, contract),
                       "task_success_contract": contract}}
    world: dict[str, Any] = {
        "tmp": tmp_path, "capture": capture, "probe_root": probe_root, "queue": tmp_path / "queue", "packet": packet,
        "spec": spec, "contract": contract, "scored": scored_task_spec(spec, contract), "profile": profile,
        "facts": facts, "context": context, "webapp_calls": [], "webapp_refusal": None, "transport_calls": [],
        "grant": website_sponsorship(context), "grant_reads": [], "grant_failure": None, "webapp_window": False,
        "published": [], "launches": [], "host_options": dict(host_options), "continuing_spend": False,
        "allocator_calls": [], "build_error": None, "transport_blockers": []}
    monkeypatch.setattr(registry, "profile_for_capture", lambda root: dict(profile) if str(
        Path(root).resolve()) == profile["capture_root"] else None)
    monkeypatch.setattr(registry, "scene_facts", lambda _profile: copy.deepcopy(world["facts"]))
    monkeypatch.setattr(agent_execution_offer, "publish_agent_execution_offer",
                        lambda root: world["published"].append(Path(root)) or {"status": "published"})
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": COMMIT}))
    monkeypatch.setattr(lane, "launch_environment_blockers", lambda **_kwargs: [])
    monkeypatch.setattr(arena_vast, "run_arena_native_control_vast", fake_transport(world))
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    for name in ("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG",
                 "BLUEPRINT_ATLAS_OPERATOR_ADMISSION_FILE", "BLUEPRINT_VAST_MAX_ISAAC_DRIVER", runner.QUEUE_ENV,
                 runner.WORK_ENV, "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(lane.ENABLE_ENV, "1")

    def webapp(*, capture_id: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation == "scene-sponsorship":
            # The current grant (the signed non-creating read). Past its own expiry the WebApp
            # answers only while its own verified intent window covers it.
            world["grant_reads"].append({"capture_id": capture_id, "payload": dict(payload)})
            failure = world["grant_failure"]
            if isinstance(failure, BaseException):
                raise failure
            if failure:
                raise ValueError(f"website_control_scene-sponsorship_http_409:{failure}")
            if world["grant"]["expires_at_epoch"] <= time.time() and not world["webapp_window"]:
                raise ValueError("website_control_scene-sponsorship_http_409:consent_expired")
            return copy.deepcopy(world["grant"])
        world["webapp_calls"].append({"capture_id": capture_id, "operation": operation, "payload": payload})
        if world["webapp_refusal"]:
            raise ValueError(f"website_control_{operation}_http_409:{world['webapp_refusal']}")
        return {**payload["spend"], "schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                "external_disclosure_allowed": True, "expires_at_epoch": time.time() + 3 * 3600,
                "sponsorship_digest": "sha256:" + "5" * 64}

    monkeypatch.setattr(website_task_context, "website_webapp_request", webapp)
    return world


def runner_hooks(world: dict[str, Any]) -> dict[str, Any]:
    def run(argv: list[str], *, attempt: Path) -> int:
        world["allocator_calls"].append(list(argv))
        return allocator.main(argv)

    return {"allocator_runner": run, "source_commit": lambda: COMMIT}


def staged_attempt(world: dict[str, Any], index: int = 0, *, expected: str = "probe_required") -> dict[str, Any]:
    """One staged attempt, exactly as the runner stages it (inputs, bundle, authority, argv)."""

    state = current_probe_state(world["capture"])
    assert state["status"] == expected, state
    job = runner._job_dir(world["queue"], state)
    job.mkdir(parents=True, exist_ok=True)
    return runner.stage_attempt(job, index, state=state, commit=COMMIT)


def _receipt(world: dict[str, Any]) -> dict[str, Any] | None:
    return gate.retained_probe_receipt(world["probe_root"])


# --- binding to the scored specification -------------------------------------------------------


def test_the_probe_is_planned_against_the_specification_teams_are_scored_with(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.rigid_robot_free_probe_bundle import RigidRobotFreeProbeBundleError, probe_inputs
    from blueprint_pipeline.team_robot_free_probe import probe_request

    world = build_world(tmp_path, monkeypatch)
    state = current_probe_state(world["capture"])
    assert state["status"] == "probe_required"
    binding = state["request"]["binding"]
    # The scene's task under the job's sealed success contract, never the configured revision's own spec.
    assert binding["task_spec_digest"] == canonical_digest(world["scored"])
    assert binding["task_spec_digest"] != canonical_digest(world["spec"])
    assert binding["configured_scene_revision_digest"] == REVISION
    assert binding["scene_context"] == worker.SCENE_CONTEXT and binding["scene_collisions_modeled"] is True
    inputs = probe_inputs(state=state)
    assert inputs["probe_request"] == state["request"]
    assert inputs["scene_plan"]["scenario"]["seed"] == registry.NOMINAL_CELL["seed"]
    assert inputs["scene_plan"]["task_spec"]["task_success_contract"] == world["contract"]
    # A request bound to the revision's specification (the old Isaac preparation's binding) is refused:
    # its receipt would refuse every team with task_spec_mismatch.
    revision_bound = probe_request(task_spec=world["spec"], scene_revision_digest=REVISION)
    assert revision_bound is not None
    with pytest.raises(RigidRobotFreeProbeBundleError, match="request_not_bound_to_scored_spec"):
        probe_inputs(state={**state, "request": revision_bound})


def test_worker_constants_are_pinned_to_their_control_plane_owners() -> None:
    from blueprint_pipeline import robot_free_readiness_retention, team_robot_free_probe

    assert worker.REQUEST_SCHEMA == team_robot_free_probe.REQUEST_SCHEMA
    assert worker.SCENE_CONTEXT == team_robot_free_probe.SCENE_CONTEXT
    assert worker.TRACE_FILENAME == robot_free_readiness_retention.TRACE_FILENAME
    assert PROBE_KIND == lane.PROBE_KIND == "rigid-robot-free-probe"


# --- allocator routing and refusals before spend ----------------------------------------------


def test_the_canonical_allocator_routes_only_this_probe_kind_to_its_isolated_lane() -> None:
    from blueprint_pipeline import policy_canary_allocator_lane
    from blueprint_pipeline.native_task_arena_allocator_dispatch import isolated_native_task_arena_lane

    assert allocator.RIGID_ROBOT_FREE_PROBE_KIND == PROBE_KIND
    assert isolated_native_task_arena_lane(PROBE_KIND) is lane.run_rigid_robot_free_probe_allocator_lane
    assert (isolated_native_task_arena_lane(policy_canary_allocator_lane.PROBE_KIND)
            is policy_canary_allocator_lane.run_policy_canary_allocator_lane)
    with pytest.raises(ValueError, match="isolated_native_task_arena_lane_unknown"):
        isolated_native_task_arena_lane("native-task-arena-construction")


def _run_allocator(world: dict[str, Any], argv: list[str]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    code = allocator.main(argv)
    attempt = Path(argv[argv.index("--adapter-output") + 1]).parent
    return code, json.loads((attempt / "paid_admission.json").read_text()), json.loads(
        (attempt / runner.RESULT_FILENAME).read_text())


def _with(argv: list[str], flag: str, value: str) -> list[str]:
    changed = list(argv)
    changed[changed.index(flag) + 1] = value
    return changed


@pytest.mark.parametrize("case, blocker, reserved", [
    ("disabled", "rigid_robot_free_probe_lane_disabled", False),
    ("context_missing", "rigid_robot_free_probe_website_grant_missing", False),
    ("grant_missing", "rigid_robot_free_probe_website_grant_missing", False),
    ("grant_expired", "rigid_robot_free_probe_website_grant_expired", False),
    ("grant_ending", "rigid_robot_free_probe_website_grant_expired", False),
    ("webapp_expired", "rigid_robot_free_probe_website_grant_expired", True),
    ("webapp_exhausted", "rigid_robot_free_probe_website_grant_exhausted", True),
    ("webapp_missing", "rigid_robot_free_probe_website_grant_missing", True),
    ("decided", "rigid_robot_free_probe_receipt_already_decided", False),
    ("over_cap", "rigid_robot_free_probe_spend_cap_above_ceiling", False),
    ("over_rate", "rigid_robot_free_probe_hourly_rate_above_ceiling", False),
    ("over_ttl", "rigid_robot_free_probe_hard_ttl_invalid", False),
    ("driver_ceiling_off", "rigid_robot_free_probe_isaac_driver_ceiling_not_enforced", False),
    ("stale_request", "rigid_robot_free_probe_request_stale", False),
])
def test_the_lane_refuses_before_any_spend(tmp_path, monkeypatch, case, blocker, reserved) -> None:
    world = build_world(tmp_path, monkeypatch)
    staged = staged_attempt(world)
    argv = staged["argv"]
    pipeline = world["capture"] / "pipeline"
    if case == "disabled":
        monkeypatch.delenv(lane.ENABLE_ENV)
    elif case == "context_missing":
        (pipeline / "website_task_context.json").unlink()
    elif case == "grant_missing":
        world["grant_failure"] = "website_scene_sponsorship_missing"
    elif case == "grant_expired":
        world["grant"] = website_sponsorship(world["context"], expires_in=-60)
    elif case == "grant_ending":
        world["grant"] = website_sponsorship(world["context"], expires_in=600)
    elif case.startswith("webapp_"):
        world["webapp_refusal"] = {"webapp_expired": "consent_expired",
                                   "webapp_exhausted": "website_scene_preparation_budget_exhausted",
                                   "webapp_missing": "website_scene_sponsorship_missing"}[case]
    elif case == "decided":
        # Decided for this request: bound to its scene revision and task specification.
        binding = current_probe_state(world["capture"])["request"]["binding"]
        (world["probe_root"] / gate.RECEIPT_FILENAME).write_text(json.dumps(
            {"status": "failed", **{field: binding[field] for field in retention.RECEIPT_BINDING_FIELDS}}))
    elif case == "over_cap":
        argv = _with(argv, "--adp-max-spend-usd", "2.0")
    elif case == "over_rate":
        argv = _with(argv, "--adp-max-hourly-rate-usd", "0.9")
    elif case == "over_ttl":
        argv = _with(argv, "--adp-hard-ttl-seconds", "7200")
    elif case == "driver_ceiling_off":
        monkeypatch.setenv("BLUEPRINT_VAST_MAX_ISAAC_DRIVER", "0")
    elif case == "stale_request":
        world["facts"]["scene"]["scene_revision_digest"] = "sha256:" + "6" * 64
    code, admission, result = _run_allocator(world, argv)
    assert code == 2
    assert blocker in admission["blockers"] and blocker in result["blockers"], (admission, result)
    assert admission["status"] == "blocked" and result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert world["transport_calls"] == []
    # A refusal the lane can know locally never reaches the WebApp; a WebApp refusal reserves nothing.
    assert bool(world["webapp_calls"]) is reserved
    assert not (staged["root"] / "allocator" / lane.FUNDING_FILENAME).exists()
    assert not (staged["root"] / "allocator" / lane.CONSUMPTION_FILENAME).exists()


# --- the capture's current website grant ------------------------------------------------------

INTAKE_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"


def _stage_intent(root: Path, grant: dict[str, Any], *, name: str) -> dict[str, Any]:
    """A sealed scene-preparation intent for this capture, issued under ``grant`` (synthetic request)."""

    from blueprint_pipeline.task_evaluation_scene_intent_contracts import INTENT_SCHEMA, _seal

    request = {"submission_id": grant["capture_id"], "owner": dict(grant["owner"]),
               "consent": copy.deepcopy(grant["consent"]), "task": {"subject": {}},
               "source": {"content_digest": "sha256:" + "3" * 64},
               "execution": {"purpose": "scene_preparation", "policy_candidates": [], "max_retries": 0,
                             "max_total_spend_usd": grant["max_total_spend_usd"],
                             "max_paid_attempts": grant["max_paid_attempts"],
                             "expires_at_epoch": grant["expires_at_epoch"]}}
    intent = _seal({"schema_version": INTENT_SCHEMA, "intent_id": name, "request": request,
                    "authenticated_issuer": "webapp", "accepted_at_epoch": grant["consent"]["accepted_at_epoch"],
                    "source_content_digest": request["source"]["content_digest"],
                    "task_content_digest": "sha256:" + "2" * 64, "provider_mutation_performed": False}, "intent_digest")
    (root / name).mkdir(parents=True)
    (root / name / "intent.json").write_text(json.dumps(intent))
    return intent


def _extend(root: Path, intent: dict[str, Any], until: float) -> dict[str, Any]:
    """The operator door's sealed, time-only extension (the real writer)."""

    from blueprint_pipeline.task_evaluation_scene_execution_window import ACK, extend_scene_execution_window_by_operator

    return extend_scene_execution_window_by_operator(
        queue_root=root, intent_id=intent["intent_id"], intent_digest=intent["intent_digest"], expires_at_epoch=until,
        authorization_reference="synthetic owner approval (time only)", ack=ACK, door_caller="operator",
        door_request_id="20261103T160000Z-extend-scene-execution-window-0123abcd", trusted_clients={"webapp"},
        capture_rights={"consent_revoked": False, "derived_scene_generation_allowed": True}, apply=True)


def test_the_precheck_reads_the_webapps_current_grant_not_the_copy_from_preparation_start(
        tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    pipeline = world["capture"] / "pipeline"
    # capture-447ae4f4: the copy written when preparation started is the original grant, long expired;
    # the sponsor has since renewed it on the WebApp.
    stale = website_sponsorship(world["context"], expires_in=-2 * 86400)
    (pipeline / "website_scene_sponsorship.json").write_text(json.dumps(stale))
    world["grant"] = website_sponsorship(world["context"], renewal={
        "parent_authority_digest": stale["authority_digest"], "parent_expires_at_epoch": stale["expires_at_epoch"],
        "approved_by": GRANT_OWNER["user_id"], "approval_reference": "synthetic owner approval"})
    assert lane.retained_website_grant(world["capture"]) == (world["context"], [])
    (read,) = world["grant_reads"]
    # The non-creating read: it can never create or renew a grant.
    assert read == {"capture_id": "walkthrough-req1", "payload": {
        "request_id": "req1", "scene_id": "site-req1", "purpose": "scene_preparation",
        "expected_task_context_digest": world["context"]["context_digest"], "create": False}}
    (pipeline / "website_scene_sponsorship.json").unlink()
    assert lane.retained_website_grant(world["capture"]) == (world["context"], [])
    # The queued probe is then funded once from that grant and decided.
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world))[0]["status"] == "decided"
    assert [call["operation"] for call in world["webapp_calls"]] == ["preparation-spend"]


def test_an_owner_approved_window_of_the_grants_own_intent_keeps_it_funding_the_probe(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    intents = tmp_path / "intents"
    intents.mkdir()
    monkeypatch.setenv(INTAKE_ROOT_ENV, str(intents))
    # The grant's own 24 h window ended an hour ago.
    grant = world["grant"] = website_sponsorship(world["context"], expires_in=-3600)
    bound = _stage_intent(intents, grant, name="scene-" + "1" * 64)
    assert lane.retained_website_grant(world["capture"])[1] == ["rigid_robot_free_probe_website_grant_expired"]
    # From here the WebApp honours its own mirrored window; the Pipeline still checks its own copy of it.
    world["webapp_window"] = True
    # The parent grant's intent carries a longer window, and a revoked intent another: neither extends this grant.
    parent = _stage_intent(intents, website_sponsorship(world["context"], expires_in=-2 * 86400),
                           name="scene-" + "2" * 64)
    _extend(intents, parent, time.time() + 3 * 86400)
    revoked = _stage_intent(intents, grant, name="scene-" + "3" * 64)
    _extend(intents, revoked, time.time() + 2 * 86400)
    (intents / revoked["intent_id"] / "revoked.json").write_text("{}")
    assert lane.retained_website_grant(world["capture"])[1] == ["rigid_robot_free_probe_website_grant_expired"]
    # A window ending before the shortest run could finish is still expired.
    _extend(intents, bound, time.time() + 600)
    assert lane.retained_website_grant(world["capture"])[1] == ["rigid_robot_free_probe_website_grant_expired"]
    _extend(intents, bound, time.time() + 6 * 3600)
    assert lane.retained_website_grant(world["capture"]) == (world["context"], [])
    # The WebApp's admission follows the window past the grant's own expiry, and so does the run's deadline.
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world))[0]["status"] == "decided"
    (attempt,) = runner._attempts(runner._job_dir(world["queue"], current_probe_state(world["capture"])))
    funding = json.loads((attempt / "allocator" / lane.FUNDING_FILENAME).read_text())
    assert funding["expires_at_epoch"] > grant["expires_at_epoch"]
    assert world["transport_calls"][0]["hard_deadline_epoch"] == funding["expires_at_epoch"]


@pytest.mark.parametrize("failure, blocker", [
    ("consent_expired", "rigid_robot_free_probe_website_grant_expired"),
    ("website_scene_sponsorship_missing", "rigid_robot_free_probe_website_grant_missing"),
    ("website_task_context_changed", "rigid_robot_free_probe_website_grant_refused:website_task_context_changed"),
    (ValueError("website_control_scene-sponsorship_http_503:website_scene_sponsorship_unavailable"),
     "rigid_robot_free_probe_website_grant_unavailable:website_scene_sponsorship_unavailable"),
    (ValueError("website_control_scene-sponsorship_http_429:unclassified"),
     "rigid_robot_free_probe_website_grant_unavailable:unclassified"),
    (ValueError("website_scene_sponsorship_binding_invalid"),
     "rigid_robot_free_probe_website_grant_invalid:website_scene_sponsorship_binding_invalid"),
    (OSError("connection refused"), "rigid_robot_free_probe_website_grant_unavailable:connection refused"),
    (TimeoutError(), "rigid_robot_free_probe_website_grant_unavailable:TimeoutError"),
])
def test_a_failed_grant_read_is_a_typed_blocker(tmp_path, monkeypatch, failure, blocker) -> None:
    world = build_world(tmp_path, monkeypatch)
    world["grant_failure"] = failure
    assert lane.retained_website_grant(world["capture"]) == (None, [blocker])


def test_an_unreachable_webapp_defers_the_probe_and_stages_nothing(tmp_path, monkeypatch) -> None:
    from urllib.error import URLError

    world = build_world(tmp_path, monkeypatch)
    world["grant_failure"] = URLError("connection refused")
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    for _ in range(runner.MAX_UNFUNDED_ATTEMPTS + 1):
        row = runner.run_queue(world["queue"], hooks=runner_hooks(world))[0]
        assert row["status"] == "deferred", row
        assert [code for code in row["blockers"] if code.startswith("rigid_robot_free_probe_website_grant_")] == [
            "rigid_robot_free_probe_website_grant_unavailable:<urlopen error connection refused>"]
    # Nothing was staged, so an outage never uses up the scene's bounded attempts.
    assert runner._attempts(runner._job_dir(world["queue"], current_probe_state(world["capture"]))) == []
    assert world["allocator_calls"] == [] and world["webapp_calls"] == [] and world["transport_calls"] == []
    world["grant_failure"] = None
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world))[0]["status"] == "decided"


def test_one_authority_funds_and_allocates_at_most_once(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    staged = staged_attempt(world)
    code, admission, result = _run_allocator(world, staged["argv"])
    assert code == 0 and result["status"] == "completed", result
    funding = json.loads((staged["root"] / "allocator" / lane.FUNDING_FILENAME).read_text())
    assert admission["allocation_binding_digest"] == funding["allocation_binding_digest"]
    assert admission["website_preparation_admission_digest"] == canonical_digest(funding)
    spend = world["webapp_calls"][0]["payload"]["spend"]
    assert world["webapp_calls"][0]["operation"] == "preparation-spend"
    assert (spend["resource_class"], spend["provider"], spend["maximum_cost_usd"], spend["request_count"]) == (
        "gpu_render", "vast", lane.HARD_CAP_USD, 1)
    assert spend["task_context_digest"] == world["context"]["context_digest"]
    call = world["transport_calls"][0]
    assert call["provider_bundle_kind"] == "native_task_arena" and call["candidate_policy_query_expected"] is False
    assert (call["max_hourly_rate_usd"], call["hard_cap_usd"], call["hard_ttl_seconds"]) == (0.8, 1.0, 3600)
    assert call["stale_offer_create_retry_limit"] == 0 and call["require_independent_watchdog"] is True
    assert call["allowed_geolocation_country_codes"] == ("US",) and call["require_known_supported_isaac_driver"]
    assert call["instance_label_prefix"] == lane.INSTANCE_LABEL_PREFIX
    assert call["instance_label_exact"].startswith(lane.INSTANCE_LABEL_PREFIX)
    assert call["min_gpu_ram_mb"] >= 24_000 and call["minimum_driver_version"].startswith("580.")
    # The watchdog deadline never outlives the website grant that funds the run.
    assert call["hard_deadline_epoch"] == funding["expires_at_epoch"]
    assert call["expected_provider_download_bytes"] > 0 and call["expected_provider_upload_bytes"] > 0
    # Zero retry: the same authority never funds or allocates again.
    code, admission, result = _run_allocator(world, staged["argv"])
    assert code == 2 and "rigid_robot_free_probe_authority_already_consumed" in result["blockers"]
    assert len(world["webapp_calls"]) == 1 and len(world["transport_calls"]) == 1


def test_a_dry_run_reserves_nothing_and_rents_nothing(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    staged = staged_attempt(world)
    dry = [value for value in staged["argv"] if value != "--execute"]
    code, admission, result = _run_allocator(world, dry)
    assert code == 0 and result["status"] == "dry_run_ready" and admission["status"] == "admitted"
    assert world["webapp_calls"] == [] and world["transport_calls"][0]["execute"] is False
    assert not (staged["root"] / "allocator" / lane.CONSUMPTION_FILENAME).exists()


def test_the_canary_preflight_runs_every_check_dry_and_never_touches_the_queue(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    scratch = tmp_path / "canary-preflight"
    outcome = runner.preflight_capture(world["capture"], scratch_dir=scratch, hooks=runner_hooks(world))
    assert outcome["status"] == "ready" and outcome["allocator_status"] == "dry_run_ready", outcome
    assert outcome["precheck_blockers"] == [] and "--execute" in outcome["execute_argv"]
    assert [call["execute"] for call in world["transport_calls"]] == [False]
    assert world["webapp_calls"] == [] and not world["queue"].exists()
    assert not (Path(outcome["attempt_root"]) / "allocator" / lane.CONSUMPTION_FILENAME).exists()
    assert runner.preflight_capture(world["capture"], scratch_dir=scratch, hooks=runner_hooks(world))[
        "reason"] == "rigid_robot_free_probe_preflight_scratch_not_empty"
    monkeypatch.delenv(lane.ENABLE_ENV)
    blocked = runner.preflight_capture(world["capture"], scratch_dir=tmp_path / "off", hooks=runner_hooks(world))
    assert blocked["status"] == "blocked" and "rigid_robot_free_probe_lane_disabled" in blocked["precheck_blockers"]
    assert "rigid_robot_free_probe_lane_disabled" in blocked["allocator_blockers"]


# --- queue, ingest, retries and decided receipts ---------------------------------------------


def test_a_queued_rigid_scene_is_probed_once_and_its_passed_receipt_admits_teams(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    queued = runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert queued["status"] == "queued" and queued["lane_enabled"] is True
    rows = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert rows[0]["status"] == "decided", rows
    receipt = _receipt(world)
    assert receipt["status"] == "passed" and receipt["learned_policy_evaluation_admitted"] is True
    # The receipt admits a scored team run: same revision, scored spec and success contract.
    gate.require_robot_free_probe(
        receipt, scene_revision_digest=REVISION, task_spec_digest=canonical_digest(world["scored"]),
        task_kind="rigid_pick_place",
        task_success_contract_digest=gate.task_success_contract_binding("rigid_pick_place", world["contract"]))
    assert receipt["proof_boundary"]["scene_context"] == worker.SCENE_CONTEXT
    # The recorded trace is retained once, immutably, beside the source launch.
    trace_path = world["probe_root"] / worker.TRACE_FILENAME
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "recorded" and receipt["physics_probe_digest"] == trace["trace_digest"]
    assert trace_path.stat().st_mode & 0o777 == 0o440
    assert world["published"] == [world["capture"].resolve()]
    # The GPU host built the scene once, left the robot idle, rendered nothing and loaded no policy.
    assert world["host"].resets == [0] and world["host"].closed is True
    attempt = runner._attempts(runner._job_dir(world["queue"], current_probe_state(world["capture"])))
    native = json.loads(next((attempt[0] / "allocator").rglob(worker.RESULT_FILENAME)).read_text())
    assert native["policy_loaded"] is False and native["candidate_policy_queried"] is False
    assert native["robot_commanded"] is False and native["scored_here"] is False
    assert len(world["allocator_calls"]) == 1 and len(world["webapp_calls"]) == 1
    assert not list((world["queue"] / "pending").glob("*.json"))
    # Decided: nothing re-queues or re-spends.
    again = runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert again["status"] == "receipt_decided" and again["queued"] is False
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world)) == []
    assert len(world["allocator_calls"]) == 1


def test_with_a_work_root_the_bundle_and_allocator_job_land_there_and_the_queue_keeps_the_records(
        tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    work_root = tmp_path / "work-volume" / "rigid-robot-free-probe-work"
    work_root.mkdir(parents=True)
    monkeypatch.setenv(runner.WORK_ENV, str(work_root))
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    rows = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert rows[0]["status"] == "decided", rows
    assert _receipt(world)["status"] == "passed"
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    [attempt] = runner._attempts(job)
    work = work_root / "jobs" / job.name / "attempts" / attempt.name
    record = json.loads((attempt / runner.ATTEMPT_FILENAME).read_text())
    assert record[runner.WORK_DIR_FIELD] == str(work) and runner.attempt_work_dir(attempt) == work
    # The bulk is on the work root; the attempt's records and receipts stay in the queue.
    assert (work / "bundle").is_dir() and (work / "allocator" / lane.FUNDING_FILENAME).is_file()
    assert not (attempt / "bundle").exists() and not (attempt / "allocator").exists()
    for name in (runner.ATTEMPT_FILENAME, runner.AUTHORITY_FILENAME, runner.INGEST_FILENAME,
                 runner.RESULT_FILENAME, "paid_admission.json"):
        assert (attempt / name).is_file(), name
    argv = world["allocator_calls"][0]
    assert argv[argv.index("--adp-job-dir") + 1] == str(work / "allocator")
    assert argv[argv.index("--adapter-output") + 1] == str(attempt / runner.RESULT_FILENAME)
    assert json.loads((attempt / runner.INGEST_FILENAME).read_text())["funded"] is True
    # One funded attempt is counted from the work root, as from the queue before.
    assert runner._counts(job) == (1, 0)


def test_a_work_root_attempt_whose_volume_is_gone_holds_and_a_legacy_attempt_keeps_its_queue_paths(
        tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    legacy = staged_attempt(world)["root"]
    assert runner.WORK_DIR_FIELD not in json.loads((legacy / runner.ATTEMPT_FILENAME).read_text())
    assert runner.attempt_work_dir(legacy) == legacy and (legacy / "bundle").is_dir()

    state = current_probe_state(world["capture"])
    job = runner._job_dir(world["queue"], state)
    # A configured root that is not there stages nothing (configuration, not a staging refusal).
    monkeypatch.setenv(runner.WORK_ENV, str(tmp_path / "absent"))
    with pytest.raises(runner.RigidRobotFreeProbeWorkRootError):
        runner.new_attempt_work_dir(job, 1)
    work_root = tmp_path / "work"
    work_root.mkdir()
    monkeypatch.setenv(runner.WORK_ENV, str(work_root))
    work = runner.new_attempt_work_dir(job, 1)
    staged = runner.stage_attempt(job, 1, state=state, commit=COMMIT, work_dir=work)
    attempt = staged["root"]
    assert runner.attempt_work_dir(attempt) == work and (work / "bundle").is_dir()
    # An allocator ran and returned, but the volume holding its job directory is not mounted:
    # whether it was funded cannot be read, so the capture waits instead of funding another.
    (attempt / runner.INTENT_FILENAME).write_text("{}")
    (attempt / runner.RESULT_FILENAME).write_text(json.dumps({"status": "blocked", "blockers": ["x"]}))
    work.rename(work.with_name("moved"))
    assert runner.settle_attempt(attempt, state=state)["state"] in runner.WAITING_STATES
    assert not (attempt / runner.INGEST_FILENAME).exists()
    # A recorded work dir that is not this attempt's is refused, never followed.
    record = json.loads((attempt / runner.ATTEMPT_FILENAME).read_text())
    (attempt / runner.ATTEMPT_FILENAME).write_text(json.dumps({**record, runner.WORK_DIR_FIELD: str(legacy)}))
    with pytest.raises(runner.RigidRobotFreeProbeIngestError, match="work_dir_invalid"):
        runner.attempt_work_dir(attempt)


def test_a_scene_that_cannot_hold_its_goal_fails_and_that_receipt_is_final(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch, support_max_xy=(3.05, -1.4))
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    rows = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert rows[0]["status"] == "decided" and _receipt(world)["status"] == "failed"
    assert _receipt(world)["learned_policy_evaluation_admitted"] is False
    with pytest.raises(gate.RobotFreeProbeGateError, match="not_passed"):
        gate.require_robot_free_probe(_receipt(world), scene_revision_digest=REVISION)
    # The lane never pays to try again, and a later passing trace never replaces the sealed failure.
    from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
    from blueprint_pipeline.team_robot_free_probe import retain_recorded_probe

    staged = staged_attempt(world, index=1, expected="receipt_decided")  # as an operator could, by hand
    code, _admission, result = _run_allocator(world, staged["argv"])
    assert code == 2 and "rigid_robot_free_probe_receipt_already_decided" in result["blockers"]
    assert len(world["transport_calls"]) == 1 and len(world["webapp_calls"]) == 1
    request = current_probe_state(world["capture"])["request"]
    passing = record_rigid_robot_free_probe(scene=FlatSupport(world["spec"]["start_pose_world"]),
                                            plan=request["plan"], binding=request["binding"])
    retained = retain_recorded_probe(profile=world["profile"], facts=world["facts"], trace=passing)
    assert retained["receipt_status"] == "passed"  # it would have passed ...
    assert retained["write"] == "kept_existing_decided" and _receipt(world)["status"] == "failed"  # ... but never replaces


def test_an_undecided_run_is_retried_once_then_decided(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch, readback_fails=True)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    original = hooks["allocator_runner"]

    def heal_after_first(argv: list[str], *, attempt: Path) -> int:
        code = original(argv, attempt=attempt)
        world["host_options"] = {}  # the second host reads its scene
        return code

    hooks["allocator_runner"] = heal_after_first
    rows = runner.run_queue(world["queue"], hooks=hooks)
    assert rows[0]["status"] == "decided" and _receipt(world)["status"] == "passed", rows
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    first, second = runner._attempts(job)
    first_ingest = json.loads((first / runner.INGEST_FILENAME).read_text())
    assert first_ingest["state"] == "undecided" and first_ingest["trace_status"] == "failed"
    assert first_ingest["retention"]["receipt_status"] == "probe_unavailable"
    assert json.loads((second / runner.INGEST_FILENAME).read_text())["state"] == "decided"
    # The failed recording never reached the immutable trace file; the passing one did.
    assert json.loads((world["probe_root"] / worker.TRACE_FILENAME).read_text())["status"] == "recorded"
    assert len(world["allocator_calls"]) == 2 and len(world["webapp_calls"]) == 2


def test_a_capped_canary_pass_starts_one_attempt_and_seals_nothing(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch, readback_fails=True)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    row = runner.run_queue(world["queue"], capture_root=world["capture"], hooks=runner_hooks(world),
                           max_attempts=1)[0]
    assert row["status"] == "deferred" and row["reason"] == "attempt_limit_reached_for_this_pass", row
    assert row["last"]["state"] == "undecided" and len(world["allocator_calls"]) == 1
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    assert not (job / runner.TERMINAL_FILENAME).exists()
    world["host_options"] = {}
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world))[0]["status"] == "decided"
    assert len(world["allocator_calls"]) == 2 and _receipt(world)["status"] == "passed"
    with pytest.raises(ValueError, match="max_attempts_invalid"):
        runner.run_queue(world["queue"], max_attempts=0)


def test_two_undecided_funded_attempts_exhaust_the_scene(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch, readback_fails=True)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    rows = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert rows[0]["status"] == "attempts_exhausted", rows
    assert rows[0]["detail"]["funded_attempts"] == runner.MAX_FUNDED_ATTEMPTS == 2
    assert len(world["allocator_calls"]) == 2
    assert _receipt(world)["status"] == "probe_unavailable"
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world)) == []


def test_an_allocated_run_without_provider_zero_holds_the_scene(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    world["continuing_spend"] = None
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    first = runner.run_queue(world["queue"], hooks=hooks)
    assert first[0]["status"] == "waiting" and first[0]["reason"] == "provider_zero_unproven", first
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "provider_zero_unproven"
    assert len(world["allocator_calls"]) == 1 and _receipt(world) is None
    assert not (world["probe_root"] / worker.TRACE_FILENAME).exists()


def test_a_launched_allocator_without_a_result_is_never_followed_by_another(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    hooks["allocator_runner"] = lambda argv, *, attempt: world["allocator_calls"].append(argv) or 0
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "unknown"
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "unknown"
    assert len(world["allocator_calls"]) == 1 and world["transport_calls"] == []


@pytest.mark.parametrize(("refusal", "blocker"), [
    # The upstream cap and the job's dollar ceiling share the WebApp's code (websiteSceneSponsorship.ts).
    ("website_scene_preparation_budget_exhausted", "rigid_robot_free_probe_website_grant_exhausted"),
    ("provider_terms_not_configured_or_changed",
     "rigid_robot_free_probe_website_grant_refused:provider_terms_not_configured_or_changed"),
])
def test_a_webapp_money_refusal_defers_without_counting_and_stages_nothing_until_its_wait(
        tmp_path, monkeypatch, refusal, blocker) -> None:
    world = build_world(tmp_path, monkeypatch)
    world["webapp_refusal"] = refusal
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    clock = [time.time()]
    hooks["now"] = lambda: clock[0]
    row = runner.run_queue(world["queue"], hooks=hooks)[0]
    assert row["status"] == "deferred" and blocker in row["blockers"], row
    assert len(world["allocator_calls"]) == 1
    clock[0] = time.time()
    # Every later wake-up inside the wait stages nothing, reserves nothing and seals nothing
    # (capture-447ae4f4: four such refusals sealed attempts_exhausted).
    for _ in range(runner.MAX_UNFUNDED_ATTEMPTS + 2):
        row = runner.run_queue(world["queue"], hooks=hooks)[0]
        assert (row["status"], row["reason"]) == ("deferred", runner.GRANT_REFUSED_STATE), row
        assert 0 < row["retry_after_seconds"] <= runner.GRANT_REFUSAL_RETRY_SECONDS and blocker in row["blockers"]
    assert len(world["allocator_calls"]) == 1 and len(world["webapp_calls"]) == 1
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    assert runner._counts(job) == (0, 0) and not (job / runner.TERMINAL_FILENAME).exists()
    (attempt,) = runner._attempts(job)
    assert json.loads((attempt / runner.INGEST_FILENAME).read_text())["state"] == runner.GRANT_REFUSED_STATE
    # After the wait one more attempt asks the WebApp; still refused, still uncounted.
    clock[0] += runner.GRANT_REFUSAL_RETRY_SECONDS + 1
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["status"] == "deferred"
    assert len(world["allocator_calls"]) == 2 and runner._counts(job) == (0, 0)
    # The owner lifts the stop (a raised cap or ceiling): the next attempt after the wait is funded and decided.
    world["webapp_refusal"] = None
    clock[0] += runner.GRANT_REFUSAL_RETRY_SECONDS + 1
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["status"] == "decided"
    assert _receipt(world)["status"] == "passed" and len(world["transport_calls"]) == 1


def test_a_refusal_before_funding_that_is_not_the_webapps_defers_to_the_next_wake_up_and_is_bounded(
        tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    # The WebApp already holds this exact binding: one binding is never paid twice, and it is not a money stop.
    monkeypatch.setattr(lane, "reserve_probe_funding", lambda **_kwargs: (
        None, ["rigid_robot_free_probe_website_grant_already_reserved"]))
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    for attempt in range(runner.MAX_UNFUNDED_ATTEMPTS - 1):
        row = runner.run_queue(world["queue"], hooks=hooks)[0]
        assert row["status"] == "deferred", row
        assert "rigid_robot_free_probe_website_grant_already_reserved" in row["blockers"]
        assert len(world["allocator_calls"]) == attempt + 1  # one attempt per wake-up, never a tight loop
    final = runner.run_queue(world["queue"], hooks=hooks)[0]
    assert final["status"] == "attempts_exhausted" and final["detail"]["unfunded_attempts"] == 4, final
    assert world["transport_calls"] == [] and _receipt(world) is None


# --- a staging refusal under an older release is retried once by each newly deployed one -------------

LEGACY_REFUSAL = "native_task_arena_bundle_robot_arm_path_blocked:" + UNREACHABLE
PHYSICS_REFUSAL = "native_task_arena_bundle_scene_physics_blocked:task_object:scene_physics_rest_overlap:1.3cm"


def _legacy_staging_refusal(world: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """The queue exactly as the runner left capture-447ae4f4's probe at 2026-11-03T17:54:40Z.

    ``stage_attempt`` made ``attempts/00`` and wrote its probe input; the bundle then refused (the
    packet's Franka could not reach the approach), so no bundle, authority, argv or intent exists. The
    runner sealed the job ``staging_refused``, before terminals named their release, and retired the
    entry to ``done/``.
    """
    from blueprint_pipeline.rigid_robot_free_probe_bundle import probe_inputs

    queued = runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    state = current_probe_state(world["capture"])
    job = runner._job_dir(world["queue"], state)
    attempt = job / "attempts" / "00"
    attempt.mkdir(parents=True)
    (job / "runner.lock").touch()
    runner._write_exclusive(attempt / worker.INPUTS_FILENAME, probe_inputs(state=state))
    terminal = _sealed({"schema_version": runner.TERMINAL_SCHEMA, "status": "staging_refused",
                        "capture_root": state["capture_root"], "probe_request_digest": state["request"]["request_digest"],
                        "detail": {"reason": LEGACY_REFUSAL}, "sealed_at_iso": "2026-11-03T17:54:40Z"},
                       "terminal_digest")
    runner._write_exclusive(job / runner.TERMINAL_FILENAME, terminal)
    (world["queue"] / "done").mkdir()
    os.replace(queued["path"], world["queue"] / "done" / Path(queued["path"]).name)
    return job, terminal


def test_the_refused_attempt_of_a_superseded_refusal_counts_toward_no_limit(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    job, terminal = _legacy_staging_refusal(world)
    attempt = job / "attempts" / "00"
    assert sorted(path.name for path in attempt.iterdir()) == [worker.INPUTS_FILENAME]
    name = Path(next((world["queue"] / "done").glob("*.json"))).name
    # Without a known running release nothing is retried.
    assert runner.requeue_new_release(world["queue"], None) == []
    unknown = {**runner_hooks(world), "source_commit": lambda: "not-a-release"}
    assert runner.run_queue(world["queue"], hooks=unknown) == []
    # A known release returns the retired entry to pending once ...
    (moved,) = runner.requeue_new_release(world["queue"], COMMIT)
    assert moved == {"entry": name, "reason": "new_release", runner.RELEASE_COMMIT_FIELD: COMMIT,
                     "terminal_status": "staging_refused", "terminal_release_commit": None}
    assert (world["queue"] / "pending" / name).is_file() and not (world["queue"] / "done" / name).exists()
    assert runner.requeue_new_release(world["queue"], COMMIT) == []
    # ... and sets the terminal aside under the job lock; an interrupted pass finishes the move.
    marker = runner._supersede_for_release(job, COMMIT)
    (job / runner.SUPERSEDED_DIRNAME / f"{terminal['terminal_digest'][7:23]}.terminal.json").rename(
        job / runner.TERMINAL_FILENAME)
    assert runner._supersede_for_release(job, "b" * 40) == marker
    assert runner._supersede_for_release(job, COMMIT) is None and not (job / runner.TERMINAL_FILENAME).exists()
    # The refused attempt launched nothing: neither funded nor a refusal before funding, and the next
    # attempt takes the next index.
    ingest = json.loads((attempt / runner.INGEST_FILENAME).read_text())
    assert ingest["state"] == runner.NOT_STAGED_STATE and ingest["funded"] is False and ingest["allocated"] is False
    assert ingest["reason"] == LEGACY_REFUSAL
    assert runner.settle_attempt(attempt, state=current_probe_state(world["capture"]))["state"] == "not_staged"
    assert runner._counts(job) == (0, 0) and runner.NOT_STAGED_STATE not in runner.UNFUNDED_STATES
    assert runner._attempts(job) == [attempt]


def test_a_staging_refusal_sealed_by_an_older_release_is_retried_once_on_a_new_one(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    job, terminal = _legacy_staging_refusal(world)
    sealed = (job / runner.TERMINAL_FILENAME).read_bytes()
    # The deployed fix: the packet's arm still cannot reach the tray, and the probe now runs.
    verdict = _unreachable_from_the_packets_own_base(world["packet"])
    monkeypatch.setattr(arm, "packet_arm_path_clearance", lambda packet_dir: dict(verdict))
    (row,) = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert row["status"] == "decided" and _receipt(world)["status"] == "passed", row
    marker = row["superseded_terminal"]
    assert marker[runner.RELEASE_COMMIT_FIELD] == COMMIT and marker["superseded_release_commit"] is None
    assert marker["superseded_terminal_digest"] == terminal["terminal_digest"]
    assert marker["superseded_detail"] == {"reason": LEGACY_REFUSAL}
    # History is kept: the sealed refusal sits beside its marker, byte for byte.
    history = job / runner.SUPERSEDED_DIRNAME
    assert (history / f"{terminal['terminal_digest'][7:23]}.terminal.json").read_bytes() == sealed
    first, second = runner._attempts(job)
    assert [first.name, second.name] == ["00", "01"]
    assert json.loads((first / runner.INGEST_FILENAME).read_text())["state"] == runner.NOT_STAGED_STATE
    assert json.loads((second / runner.INGEST_FILENAME).read_text())["state"] == "decided"
    assert runner._counts(job) == (1, 0) and len(world["allocator_calls"]) == 1
    receipt = json.loads((second / "bundle" / runner.BUNDLE_RECEIPT_FILENAME).read_text())
    assert receipt["evidence_only_checks"]["robot_arm_path_clearance"] == "blocked"
    final = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    assert final["status"] == "decided" and final[runner.RELEASE_COMMIT_FIELD] == COMMIT
    assert final[runner.RELEASE_REQUEUE_FIELD] == [COMMIT]
    # Decided: neither this release nor a later one retries it.
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world)) == []
    assert runner.run_queue(world["queue"], hooks={**runner_hooks(world), "source_commit": lambda: "b" * 40}) == []
    assert len(world["allocator_calls"]) == 1


def test_a_staging_refusal_is_final_under_its_release_and_retried_once_by_a_new_one(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    old, new = "c" * 40, COMMIT
    admitted = physics.require_scene_physics_admission

    def overlap(packet_dir: Path, **_options: Any) -> dict[str, Any]:
        raise physics.ScenePhysicsAdmissionBlocked(["task_object:scene_physics_rest_overlap:1.3cm"])

    monkeypatch.setattr(physics, "require_scene_physics_admission", overlap)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    under_old = {**runner_hooks(world), "source_commit": lambda: old}
    (row,) = runner.run_queue(world["queue"], hooks=under_old)
    assert row["status"] == "staging_refused" and row[runner.RELEASE_COMMIT_FIELD] == old, row
    assert row["detail"] == {"reason": PHYSICS_REFUSAL, "attempt": "00"}
    assert runner.RELEASE_REQUEUE_FIELD not in row
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    # Final under the release that sealed it, however its entry comes back.
    assert runner.run_queue(world["queue"], hooks=under_old) == []
    assert runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])["status"] == "queued"
    (row,) = runner.run_queue(world["queue"], hooks=under_old)
    assert row["status"] == "staging_refused" and "superseded_terminal" not in row
    assert [attempt.name for attempt in runner._attempts(job)] == ["00"]
    # A newly deployed release retries it once; the physics refusal stands, so it seals again under that one.
    (row,) = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert row["status"] == "staging_refused" and row["superseded_terminal"]["superseded_release_commit"] == old
    assert row[runner.RELEASE_COMMIT_FIELD] == new and row[runner.RELEASE_REQUEUE_FIELD] == [old, new]
    assert row["detail"] == {"reason": PHYSICS_REFUSAL, "attempt": "01"}
    assert [json.loads((attempt / runner.INGEST_FILENAME).read_text())["state"]
            for attempt in runner._attempts(job)] == [runner.NOT_STAGED_STATE] * 2
    assert runner._counts(job) == (0, 0)
    # Never twice, even once its cause is gone, and never by a release that already tried it.
    monkeypatch.setattr(physics, "require_scene_physics_admission", admitted)
    for hooks in (runner_hooks(world), under_old):
        assert runner.run_queue(world["queue"], hooks=hooks) == []
        runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
        (row,) = runner.run_queue(world["queue"], hooks=hooks)
        assert row["status"] == "staging_refused" and "superseded_terminal" not in row, row
    assert len(runner._attempts(job)) == 2 and len(list((job / runner.SUPERSEDED_DIRNAME).glob("*.terminal.json"))) == 1
    # The next release runs it: both refused attempts count toward no limit, and it takes index 02.
    later = "d" * 40
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": later}))
    (row,) = runner.run_queue(world["queue"], hooks={**runner_hooks(world), "source_commit": lambda: later})
    assert row["status"] == "decided" and _receipt(world)["status"] == "passed", row
    assert [attempt.name for attempt in runner._attempts(job)] == ["00", "01", "02"]
    assert runner._counts(job) == (1, 0) and len(world["allocator_calls"]) == 1
    assert row[runner.RELEASE_REQUEUE_FIELD] == [old, new, later]


def _half(world: dict[str, Any], support: dict[str, Any] | None) -> None:
    """The capture's compiled team half record, as ``team_scene_packet`` writes its support's (or none)."""

    path = world["tmp"] / "team_scene_inputs.v1.json"
    path.write_text(json.dumps({"base_native_packet": {"path": str(world["packet"] / PACKET_RECEIPT)},
                                **({"task_support_pad": support} if support is not None else {})}))
    world["profile"]["team_scene_inputs_path"] = str(path)


def test_a_half_its_support_check_holds_stages_nothing_and_its_recompiled_request_runs(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.rigid_support_pad import UNMODELLED_HALF

    world = build_world(tmp_path, monkeypatch)
    # As capture-447ae4f4's half was compiled, before its support was modelled: refused while staging.
    _half(world, None)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert row["status"] == "staging_refused", row
    assert row["detail"]["reason"] == "rigid_robot_free_probe_" + UNMODELLED_HALF
    # The CPU rest check's own hold is refused the same way: nothing reserved on the grant, no GPU rented.
    reason = "rigid_support_start_rest_off_start:0.5mm/0.021rad"
    _half(world, {"status": "held", "reasons": [reason]})
    with pytest.raises(runner.RigidRobotFreeProbeBundleError, match="rigid_robot_free_probe_" + reason):
        staged_attempt(world, 1)
    assert world["allocator_calls"] == [] and world["webapp_calls"] == []
    held = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    sealed = json.loads((held / runner.TERMINAL_FILENAME).read_text())
    # The recompiled half seats its object on the planarized support (this stand-in's support is flat
    # already): a new task, so a new request and its own job. The old request's sealed terminal stays
    # where it is and blocks nothing.
    spec = copy.deepcopy(world["spec"])
    spec["interaction_affordance"]["support_alignment"]["start_support"] = {
        "basis": "scene_collider_highest_point_under_collision_footprint", "collider_top_z_m": 0.7545}
    retask_packet(world["packet"], spec)
    world["facts"]["scene"]["task_spec"] = scored_task_spec(spec, world["contract"])
    _half(world, {"status": "consistent", "reasons": []})
    assert runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])["status"] == "queued"
    (row,) = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    assert job != held and row["status"] == "decided" and _receipt(world)["status"] == "passed", row
    assert json.loads((held / runner.TERMINAL_FILENAME).read_text()) == sealed
    assert len(world["allocator_calls"]) == 1


# --- an attempt budget our own runtime spent is re-driven once per newly deployed release ------------

# capture-447ae4f4's first GPU run (Vast 59017963, 2026-11-03): Arena's DROID rig has no such setter.
CAMERA_CRASH = AttributeError("'DroidCameraCfgWithExteriorRight' object has no attribute 'set_use_tiled_camera'")
CRASH_BLOCKERS = ["provider_runtime_result_missing_from_output_zip",
                  "rigid_robot_free_probe_failed_at_simulation_app_launched:AttributeError:"
                  "'DroidCameraCfgWithExteriorRight' object has no attribute 'set_use_tiled_camera'"]


def _crashing(world: dict[str, Any]) -> None:
    world["build_error"] = CAMERA_CRASH
    world["transport_blockers"] = [CRASH_BLOCKERS[0]]


def _healed(world: dict[str, Any]) -> None:
    world["build_error"] = None
    world["transport_blockers"] = []


def _under(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, release: str) -> dict[str, Any]:
    """The runner's hooks with ``release`` deployed: the allocator's checkout is that release too."""
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": release}))
    return {**runner_hooks(world), "source_commit": lambda: release}


def _work_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    work = (tmp_path / "work").resolve()
    work.mkdir()
    monkeypatch.setenv(runner.WORK_ENV, str(work))
    return work


def test_the_probe_result_is_one_the_provider_output_inspection_reads() -> None:
    from blueprint_pipeline.wam_provider_output import RUNTIME_RESULT_FILENAMES, inspect_provider_runtime_output_archive
    import io

    # Unlisted, every probe run (a recorded one too) was blocked provider_runtime_result_missing_from_output_zip.
    assert worker.RESULT_FILENAME in RUNTIME_RESULT_FILENAMES
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(worker.RESULT_FILENAME, json.dumps({"status": "completed", "blockers": []}))
    with zipfile.ZipFile(buffer) as archive:
        inspection = inspect_provider_runtime_output_archive(archive, zip_path="out.zip", zip_size_bytes=1)
    assert inspection["runtime_result_present"] is True and inspection["runtime_result_status"] == "completed"


def test_an_exhausted_probe_our_runtime_crashed_is_redriven_once_by_a_new_release(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    work = _work_root(tmp_path, monkeypatch)
    old, new = "c" * 40, COMMIT
    _crashing(world)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, old))
    assert row["status"] == "attempts_exhausted" and row[runner.RELEASE_COMMIT_FIELD] == old, row
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    # The queue exactly as the runner leaves capture-447ae4f4's probe: two funded attempts, each
    # allocated and blocked by the crash, their bulk on the work volume, then attempts_exhausted.
    attempts = runner._attempts(job)
    assert [attempt.name for attempt in attempts] == ["00", "01"]
    sealed = {attempt.name: (attempt / runner.INGEST_FILENAME).read_bytes() for attempt in attempts}
    for attempt in attempts:
        ingest = json.loads(sealed[attempt.name])
        assert {key: ingest[key] for key in ("state", "funded", "allocated", "allocator_status")} == {
            "state": "allocated_blocked", "funded": True, "allocated": True, "allocator_status": "blocked"}, ingest
        assert ingest["allocator_blockers"] == CRASH_BLOCKERS
        assert runner.attempt_work_dir(attempt) == work / "jobs" / job.name / "attempts" / attempt.name
        assert runner._funded(attempt)
    assert row["detail"]["funded_attempts"] == 2 and row["detail"]["unfunded_attempts"] == 0
    assert len(world["allocator_calls"]) == 2
    terminal = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    state = current_probe_state(world["capture"])
    # Final under the release that sealed it (the website job is held by name); not under a new one.
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, old)) == []
    assert runner.final_terminal(state, queue_root=world["queue"], release=old) == terminal
    assert runner.final_terminal(state, queue_root=world["queue"], release=new) is None
    # The deployed fix: one more funded attempt, at the lane's caps, on the next index.
    _healed(world)
    (row,) = runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, new))
    assert row["status"] == "decided" and _receipt(world)["status"] == "passed", row
    marker = row["superseded_terminal"]
    assert marker["superseded_status"] == "attempts_exhausted" and marker["superseded_release_commit"] == old
    assert marker[runner.RELEASE_COMMIT_FIELD] == new
    assert marker["superseded_terminal_digest"] == terminal["terminal_digest"]
    history = job / runner.SUPERSEDED_DIRNAME
    assert json.loads((history / f"{terminal['terminal_digest'][7:23]}.terminal.json").read_text()) == terminal
    first, second, third = runner._attempts(job)
    assert {attempt.name: (attempt / runner.INGEST_FILENAME).read_bytes() for attempt in (first, second)} == sealed
    assert json.loads((third / runner.INGEST_FILENAME).read_text())["state"] == "decided"
    staged = json.loads((third / runner.AUTHORITY_FILENAME).read_text())
    assert (staged["hard_cap_usd"], staged["max_hourly_rate_usd"], staged["hard_ttl_seconds"]) == (
        lane.HARD_CAP_USD, lane.MAX_HOURLY_RATE_USD, lane.HARD_TTL_SECONDS) == (1.0, 0.8, 3600)
    assert runner._counts(job) == (3, 0) and len(world["allocator_calls"]) == 3
    final = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    assert final["status"] == "decided" and final[runner.RELEASE_REQUEUE_FIELD] == [old, new]


def test_a_runtime_redrive_is_once_per_release_and_two_at_most(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    _work_root(tmp_path, monkeypatch)
    _crashing(world)
    releases = ["c" * 40, "d" * 40, "e" * 40, "f" * 40]
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, releases[0]))
    assert row["status"] == "attempts_exhausted" and len(world["allocator_calls"]) == 2
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    for index, release in enumerate(releases[1:3], start=1):
        # Each new release funds exactly one more attempt; the crash stands, so it seals again under that one.
        (row,) = runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, release))
        assert row["status"] == "attempts_exhausted" and row[runner.RELEASE_COMMIT_FIELD] == release, row
        assert row["detail"]["funded_attempts"] == 2 + index and len(world["allocator_calls"]) == 2 + index
        assert row[runner.RELEASE_REQUEUE_FIELD] == releases[:index + 1]
        # Never twice under the same release, however its entry comes back.
        assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, release)) == []
        runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
        (again,) = runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, release))
        assert again["status"] == "attempts_exhausted" and "superseded_terminal" not in again, again
        assert len(world["allocator_calls"]) == 2 + index
    # Two extra attempts spent: no later release funds another, even once the crash is fixed.
    _healed(world)
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, releases[3])) == []
    state = current_probe_state(world["capture"])
    assert runner.final_terminal(state, queue_root=world["queue"], release=releases[3])["status"] == "attempts_exhausted"
    assert [attempt.name for attempt in runner._attempts(job)] == ["00", "01", "02", "03"]
    assert runner._counts(job) == (4, 0) and len(world["allocator_calls"]) == 4 == 2 + runner.MAX_RUNTIME_REDRIVES
    assert len(list((job / runner.SUPERSEDED_DIRNAME).glob("*.supersession.json"))) == 2


def test_an_exhausted_probe_that_is_not_our_runtimes_failure_is_never_redriven(tmp_path, monkeypatch) -> None:
    # Two recordings the scorer could not decide: a scene outcome, not a crash of ours.
    world = build_world(tmp_path, monkeypatch, readback_fails=True)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert row["status"] == "attempts_exhausted" and len(world["allocator_calls"]) == 2, row
    world["host_options"] = {}
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, "d" * 40)) == []
    assert len(world["allocator_calls"]) == 2
    # Only an allocated attempt blocked by our own runtime alone counts as ours.
    crash = {"state": "allocated_blocked", "funded": True, "allocated": True, "allocator_blockers": CRASH_BLOCKERS}
    assert runner._runtime_failure(crash)
    assert runner._runtime_failure({**crash, "allocator_blockers": CRASH_BLOCKERS[:1]})
    assert runner._runtime_failure({**crash, "allocator_blockers": CRASH_BLOCKERS[1:]})
    for record in (
        {**crash, "allocator_blockers": [*CRASH_BLOCKERS, "vast_offer_search_no_capacity"]},  # provider capacity
        {**crash, "allocator_blockers": []},
        {**crash, "state": "undecided"},  # a scorer outcome
        {**crash, "state": "funded_unallocated"},  # funded, never allocated: the provider's
        {"state": runner.GRANT_REFUSED_STATE, "funded": False, "allocated": False,
         "allocator_blockers": ["rigid_robot_free_probe_website_grant_exhausted"]},  # a money refusal
    ):
        assert not runner._runtime_failure(record), record


def test_an_allocator_that_cannot_start_allocates_nothing_and_defers(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])

    def cannot_start(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("interpreter missing")

    monkeypatch.setattr(subprocess, "Popen", cannot_start)
    row = runner.run_queue(world["queue"], hooks={"source_commit": lambda: COMMIT})[0]
    assert row["status"] == "deferred" and row["blockers"] == ["allocator_spawn_failed"], row
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    (attempt,) = runner._attempts(job)
    assert json.loads((attempt / runner.INGEST_FILENAME).read_text())["state"] == "not_launched"
    assert world["webapp_calls"] == [] and world["transport_calls"] == []


def test_ingest_refuses_a_trace_that_is_not_this_requests(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    staged = staged_attempt(world)
    code, _admission, result = _run_allocator(world, staged["argv"])
    assert code == 0
    native_path = Path(result["native_control_result_path"])
    native = json.loads(native_path.read_text())
    # A trace bound to the revision's own specification, re-sealed as a forger would.
    trace = dict(native["trace"], binding={**native["trace"]["binding"],
                                           "task_spec_digest": canonical_digest(world["spec"])})
    trace = _sealed(trace, "trace_digest")
    forged = _sealed({**native, "trace": trace, "trace_digest": trace["trace_digest"]}, "result_digest")
    native_path.write_text(json.dumps(forged))
    (native_path.parent / worker.TRACE_FILENAME).write_text(json.dumps(trace))
    state = current_probe_state(world["capture"])
    with pytest.raises(runner.RigidRobotFreeProbeIngestError, match="result_digest_invalid"):
        runner.verified_trace(result, attempt=staged["root"], request=state["request"])
    (staged["root"] / runner.RESULT_FILENAME).write_text(json.dumps({**result,
                                                                    "native_control_result_digest": forged["result_digest"]}))
    settled = runner.settle_attempt(staged["root"], state=state)
    assert settled["state"] == "undecided" and "trace_binding_mismatch" in settled["ingest_refusal"]
    assert _receipt(world) is None and not (world["probe_root"] / worker.TRACE_FILENAME).exists()


def test_a_different_trace_beside_the_source_launch_is_a_conflict_never_overwritten(tmp_path) -> None:
    root = tmp_path / "launch"
    root.mkdir()
    assert runner.retain_trace_file(root, {"trace_digest": "sha256:1"}) == "written"
    assert runner.retain_trace_file(root, {"trace_digest": "sha256:1"}) == "unchanged"
    with pytest.raises(runner.RigidRobotFreeProbeIngestError, match="trace_conflict"):
        runner.retain_trace_file(root, {"trace_digest": "sha256:2"})
    assert json.loads((root / worker.TRACE_FILENAME).read_text()) == {"trace_digest": "sha256:1"}


# --- the trace and receipt beside the launch are the current request's --------------------------------


def _superseded_request_evidence(world: dict[str, Any], *, decided: bool = False) -> dict[str, Any]:
    """Request A's trace and receipt beside the launch, as the lane retained them before the scene was recompiled.

    capture-447ae4f4, 2026-11-03 22:54Z: request bdfe6a51 (the bumpy-table compile #3349 fixed) recorded a trace
    whose untouched object rested 14.7 mm off the task's start, scored ``probe_unavailable``, beside the launch
    request 9319f26e later probed. Request A here is the same task compiled with its start 0.5 mm higher, so
    its binding and plan are not the current request's.
    """
    from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
    from blueprint_pipeline.rigid_robot_free_readiness_score import score_rigid_robot_free_readiness
    from blueprint_pipeline.team_robot_free_probe import probe_request
    from tests.test_rigid_support_seating import _OffsetSupport

    spec = copy.deepcopy(world["scored"])
    start = list(spec["start_pose_world"])
    spec["start_pose_world"] = [*start[:2], start[2] + 0.0005, *start[3:]]
    request = probe_request(task_spec=spec, scene_revision_digest=REVISION)
    current = current_probe_state(world["capture"])["request"]
    assert request["request_digest"] != current["request_digest"] and request["binding"] != current["binding"]
    scene = (FlatSupport(spec["start_pose_world"], support_max_xy=(3.05, -1.4)) if decided
             else _OffsetSupport(spec["start_pose_world"], start_offset=-0.0147))
    trace = record_rigid_robot_free_probe(scene=scene, plan=request["plan"], binding=request["binding"])
    receipt = score_rigid_robot_free_readiness(task_spec=spec, physics_probe=trace,
                                               configured_scene_revision_digest=REVISION)
    assert trace["status"] == "recorded" and receipt["status"] == ("failed" if decided else "probe_unavailable")
    assert runner.retain_trace_file(world["probe_root"], trace, request=request) == "written"
    assert retention.retain_receipt(world["probe_root"], receipt, supersede_other_request=True) == "written"
    return {"request": request, "trace": trace, "receipt": receipt,
            "trace_bytes": (world["probe_root"] / worker.TRACE_FILENAME).read_bytes(),
            "receipt_bytes": (world["probe_root"] / gate.RECEIPT_FILENAME).read_bytes()}


def _kept(world: dict[str, Any], name: str, data: bytes) -> dict[str, Any]:
    """The history copy of ``name`` holding ``data``, byte for byte; return its marker."""
    digest = hashlib.sha256(data).hexdigest()
    kept = world["probe_root"] / retention.HISTORY_DIRNAME / f"{digest}.{name}"
    assert kept.read_bytes() == data
    marker = json.loads(kept.with_name(kept.name + ".supersession.json").read_text())
    assert marker["schema_version"] == retention.SUPERSESSION_SCHEMA and marker["file"] == name
    assert marker["sha256"] == "sha256:" + digest and marker["history_file"] == kept.name
    assert marker["marker_digest"] == canonical_digest(marker, digest_field="marker_digest")
    return marker


def _admits(world: dict[str, Any], receipt: dict[str, Any]) -> None:
    gate.require_robot_free_probe(
        receipt, scene_revision_digest=REVISION, task_spec_digest=canonical_digest(world["scored"]),
        task_kind="rigid_pick_place",
        task_success_contract_digest=gate.task_success_contract_binding("rigid_pick_place", world["contract"]))


def test_a_completed_run_refused_beside_a_superseded_requests_trace_is_reingested_once_at_no_cost(
        tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    _work_root(tmp_path, monkeypatch)
    old, new = "c" * 40, COMMIT
    superseded = _superseded_request_evidence(world)
    # The release that sealed capture-447ae4f4's job (88497b47) kept one trace per launch.
    per_request = runner.retain_trace_file
    monkeypatch.setattr(runner, "retain_trace_file", lambda root, trace, **_request: per_request(root, trace))
    # Its attempts, exactly: 00 crashed inside our runtime (funded), 01 the WebApp refused (money,
    # uncounted), 02 completed with this request's trace and was refused beside request A's.
    _crashing(world)
    hooks = _under(world, monkeypatch, old)
    clock = [time.time()]
    hooks["now"] = lambda: clock[0]
    allocate = hooks["allocator_runner"]

    def crash_then_refuse(argv: list[str], *, attempt: Path) -> int:
        code = allocate(argv, attempt=attempt)
        _healed(world)
        world["webapp_refusal"] = "website_scene_preparation_budget_exhausted"
        return code

    hooks["allocator_runner"] = crash_then_refuse
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=hooks)
    assert row["status"] == "deferred" and "rigid_robot_free_probe_website_grant_exhausted" in row["blockers"], row
    hooks["allocator_runner"], world["webapp_refusal"] = allocate, None
    clock[0] = time.time() + runner.GRANT_REFUSAL_RETRY_SECONDS + 1
    (row,) = runner.run_queue(world["queue"], hooks=hooks)
    assert row["status"] == "trace_conflict" and row[runner.RELEASE_COMMIT_FIELD] == old, row
    assert row["detail"]["ingest_refusal"] == "rigid_robot_free_probe_trace_conflict"
    assert row["detail"]["allocator_status"] == "completed" and row["detail"]["funded"] is True
    monkeypatch.setattr(runner, "retain_trace_file", per_request)
    state = current_probe_state(world["capture"])
    job = runner._job_dir(world["queue"], state)
    attempts = runner._attempts(job)
    assert [attempt.name for attempt in attempts] == ["00", "01", "02"]
    assert [json.loads((attempt / runner.INGEST_FILENAME).read_text())["state"] for attempt in attempts] == [
        "allocated_blocked", runner.GRANT_REFUSED_STATE, "trace_conflict"]
    refused = (attempts[2] / runner.INGEST_FILENAME).read_bytes()
    terminal = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    native = json.loads(next((runner.attempt_work_dir(attempts[2]) / "allocator").rglob(
        worker.RESULT_FILENAME)).read_text())
    # Request A's trace and undecided receipt are still beside the launch; neither is request B's.
    assert (world["probe_root"] / worker.TRACE_FILENAME).read_bytes() == superseded["trace_bytes"]
    assert (world["probe_root"] / gate.RECEIPT_FILENAME).read_bytes() == superseded["receipt_bytes"]
    assert state["status"] == "probe_required" and state["receipt"] is None and state["receipt_status"] is None
    assert state["superseded_receipt"]["receipt_digest"] == superseded["receipt"]["receipt_digest"]
    assert not list((world["queue"] / "pending").glob("*.json")) and list((world["queue"] / "done").glob("*.json"))
    spent = (len(world["allocator_calls"]), len(world["webapp_calls"]), len(world["transport_calls"]))
    assert spent == (3, 3, 2)  # three allocator runs, three reservations asked (one refused), two GPU runs
    # Final under the release that sealed it; not under a newly deployed one.
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, old)) == []
    assert runner.final_terminal(state, queue_root=world["queue"], release=old) == terminal
    assert runner.final_terminal(state, queue_root=world["queue"], release=new) is None
    assert runner._superseded_trace_conflict(job, terminal) == attempts[2]
    # The deployed fix ingests attempt 02's retained result once: nothing is allocated, reserved or run.
    (row,) = runner.run_queue(world["queue"], hooks={**_under(world, monkeypatch, new), "now": hooks["now"]})
    assert row["status"] == "decided", row
    assert (len(world["allocator_calls"]), len(world["webapp_calls"]), len(world["transport_calls"])) == spent
    assert runner._attempts(job) == attempts and runner._counts(job) == (2, 0)
    receipt = _receipt(world)
    assert receipt["status"] == "passed" and receipt["physics_probe_digest"] == native["trace_digest"]
    assert receipt["checks"]["zero_action_not_success"] is True and receipt["checks"]["direct_placement_success"] is True
    _admits(world, receipt)
    # Request B's trace is the one beside the launch now; request A's trace and receipt are kept.
    assert json.loads((world["probe_root"] / worker.TRACE_FILENAME).read_text()) == native["trace"]
    trace_marker = _kept(world, worker.TRACE_FILENAME, superseded["trace_bytes"])
    assert trace_marker["superseded_binding"] == superseded["request"]["binding"]
    assert trace_marker["superseded_digest"] == superseded["trace"]["trace_digest"]
    assert trace_marker["superseded_by"] == {"probe_request_digest": state["request"]["request_digest"],
                                             "trace_digest": native["trace_digest"]}
    receipt_marker = _kept(world, gate.RECEIPT_FILENAME, superseded["receipt_bytes"])
    assert receipt_marker["superseded_status"] == "probe_unavailable"
    assert receipt_marker["superseded_by"]["receipt_digest"] == receipt["receipt_digest"]
    # The job's history: the marker names the attempt it re-ingested, and its refused record sits beside it.
    marker = row["superseded_terminal"]
    assert marker["superseded_status"] == "trace_conflict" and marker["superseded_release_commit"] == old
    assert marker[runner.RELEASE_COMMIT_FIELD] == new and marker[runner.REINGEST_FIELD] == "02"
    assert marker["superseded_ingest_digest"] == terminal["detail"]["record_digest"]
    history, name = job / runner.SUPERSEDED_DIRNAME, terminal["terminal_digest"][7:23]
    assert (history / f"{name}.attempt-02.ingest.json").read_bytes() == refused
    assert json.loads((history / f"{name}.terminal.json").read_text()) == terminal
    reingested = json.loads((attempts[2] / runner.INGEST_FILENAME).read_text())
    assert reingested["state"] == "decided" and reingested["trace_write"] == "written_superseding"
    assert reingested["retention"]["write"] == "superseded_other_request"
    assert reingested["trace_digest"] == native["trace_digest"]
    final = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    assert final["status"] == "decided" and final[runner.RELEASE_REQUEUE_FIELD] == [old, new]
    # Once: neither this release nor a later one reads it again, and nothing more is spent.
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, new)) == []
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, "d" * 40)) == []
    assert (len(world["allocator_calls"]), len(world["webapp_calls"]), len(world["transport_calls"])) == spent


def test_a_different_trace_of_the_attempts_own_request_stays_a_final_conflict(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe

    world = build_world(tmp_path, monkeypatch)
    old = "c" * 40
    request = current_probe_state(world["capture"])["request"]
    # Another recording of this very request (its object falls off the support's edge) is already retained.
    own = record_rigid_robot_free_probe(scene=FlatSupport(world["spec"]["start_pose_world"], support_max_xy=(3.05, -1.4)),
                                        plan=request["plan"], binding=request["binding"])
    assert runner.retain_trace_file(world["probe_root"], own, request=request) == "written"
    retained = (world["probe_root"] / worker.TRACE_FILENAME).read_bytes()
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, old))
    assert row["status"] == "trace_conflict" and row["detail"]["ingest_refusal"].endswith("trace_conflict"), row
    state = current_probe_state(world["capture"])
    job = runner._job_dir(world["queue"], state)
    terminal = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    # A real conflict: no release re-ingests it, and the retained trace is never overwritten or set aside.
    assert runner._superseded_trace_conflict(job, terminal) is None
    assert runner.run_queue(world["queue"], hooks=_under(world, monkeypatch, COMMIT)) == []
    assert runner.final_terminal(state, queue_root=world["queue"], release=COMMIT) == terminal
    assert (world["probe_root"] / worker.TRACE_FILENAME).read_bytes() == retained
    assert not (world["probe_root"] / retention.HISTORY_DIRNAME).exists() and _receipt(world) is None
    assert len(world["allocator_calls"]) == 1
    with pytest.raises(runner.RigidRobotFreeProbeIngestError, match="trace_binding_mismatch"):
        runner.retain_trace_file(world["probe_root"], own, request={**request, "plan": {}})


def test_a_decided_receipt_of_a_superseded_request_never_shadows_the_current_one(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    superseded = _superseded_request_evidence(world, decided=True)
    # Request A's sealed failure is not request B's verdict: B is still to be probed, and the gate never admits A's.
    state = current_probe_state(world["capture"])
    assert state["status"] == "probe_required" and state["receipt"] is None and state["receipt_status"] is None
    assert state["superseded_receipt"] == {
        "receipt_digest": superseded["receipt"]["receipt_digest"], "status": "failed",
        "task_spec_digest": superseded["request"]["binding"]["task_spec_digest"],
        "configured_scene_revision_digest": REVISION}
    with pytest.raises(gate.RobotFreeProbeGateError):
        _admits(world, _receipt(world))
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    (row,) = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert row["status"] == "decided" and len(world["allocator_calls"]) == 1, row
    receipt = _receipt(world)
    assert receipt["status"] == "passed" and receipt["task_spec_digest"] == canonical_digest(world["scored"])
    _admits(world, receipt)
    assert _kept(world, gate.RECEIPT_FILENAME, superseded["receipt_bytes"])["superseded_status"] == "failed"
    assert _kept(world, worker.TRACE_FILENAME, superseded["trace_bytes"])["superseded_status"] == "recorded"
    # Decided for request B itself: never replaced, by any writer.
    retained = (world["probe_root"] / gate.RECEIPT_FILENAME).read_bytes()
    assert retention.retain_receipt(world["probe_root"], {**superseded["receipt"], **{
        field: receipt[field] for field in retention.RECEIPT_BINDING_FIELDS}},
        supersede_other_request=True) == "kept_existing_decided"
    assert (world["probe_root"] / gate.RECEIPT_FILENAME).read_bytes() == retained


def test_only_the_current_requests_writer_sets_a_decided_receipt_of_another_request_aside(tmp_path) -> None:
    root = tmp_path / "launch"
    root.mkdir()
    bound = {"configured_scene_revision_digest": REVISION, "replacement_asset_digest": None}
    a = {**bound, "status": "failed", "task_spec_digest": "sha256:" + "a" * 64, "receipt_digest": "sha256:1"}
    b = {**bound, "status": "passed", "task_spec_digest": "sha256:" + "b" * 64, "receipt_digest": "sha256:2"}
    assert retention.retain_receipt(root, a) == "written"
    sealed = (root / gate.RECEIPT_FILENAME).read_bytes()
    # Scene preparation's retention scores the revision, not a probe request: it never sets a decided receipt aside.
    assert retention.retain_receipt(root, b) == "kept_existing_decided"
    assert (root / gate.RECEIPT_FILENAME).read_bytes() == sealed
    assert retention.retain_receipt(root, b, supersede_other_request=True) == "superseded_other_request"
    assert json.loads((root / gate.RECEIPT_FILENAME).read_text()) == b
    kept = root / retention.HISTORY_DIRNAME / f"{hashlib.sha256(sealed).hexdigest()}.{gate.RECEIPT_FILENAME}"
    assert kept.read_bytes() == sealed
    # An undecided receipt a decided one replaces is kept too; the same bytes set aside twice stay one copy.
    unavailable = {**b, "status": "probe_unavailable", "receipt_digest": "sha256:3"}
    other = root / "other"
    other.mkdir()
    for _ in range(2):
        assert retention.retain_receipt(other, unavailable) in {"written", "unchanged"}
        (other / gate.RECEIPT_FILENAME).chmod(0o640)
        assert retention.set_aside(other, gate.RECEIPT_FILENAME, superseded_by={"why": "test"})["file"] == (
            gate.RECEIPT_FILENAME)
    assert len(list((other / retention.HISTORY_DIRNAME).glob(f"*.{gate.RECEIPT_FILENAME}"))) == 1
    assert retention.set_aside(other, gate.RECEIPT_FILENAME, superseded_by={}) is None
    assert retention.retain_receipt(other, unavailable) == "written"
    assert retention.retain_receipt(other, b) == "replaced_unavailable"
    assert retention.receipt_bound_to_request(b, {"binding": {**bound, "task_spec_digest": b["task_spec_digest"]}})
    assert not retention.receipt_bound_to_request(a, {"binding": {**bound, "task_spec_digest": b["task_spec_digest"]}})


# --- the hook: enqueue launches nothing --------------------------------------------------------


def test_enqueue_launches_nothing_is_idempotent_and_never_raises(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    monkeypatch.delenv(lane.ENABLE_ENV)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("enqueue started a process or an allocation")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(allocator, "main", forbidden)
    first = runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert first["status"] == "queued" and first["lane_enabled"] is False
    assert runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])["status"] == (
        "already_queued")
    assert [path.relative_to(world["queue"]).as_posix() for path in world["queue"].rglob("*") if path.is_file()] == [
        f"pending/{Path(first['path']).name}"]
    # Switched off, the runner touches nothing.
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world)) == [{"status": "lane_disabled", "pending": 1}]
    assert not (world["queue"] / "jobs").exists() and world["webapp_calls"] == []
    # A capture without a profile, or a task the probe cannot carry, queues nothing; failures never raise.
    assert runner.enqueue_rigid_robot_free_probe(tmp_path / "elsewhere", queue_root=world["queue"])["status"] == (
        "profile_missing")
    world["facts"]["scene"]["task_spec"] = {"task_kind": "articulated_open_close"}
    assert runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])["status"] == (
        "unavailable_for_task")
    monkeypatch.setattr(registry, "scene_facts", lambda _profile: (_ for _ in ()).throw(RuntimeError("boom")))
    refused = runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert refused["status"] == "refused" and refused["queued"] is False


def test_the_real_registry_queues_nothing_for_a_task_the_probe_cannot_carry(tmp_path, monkeypatch) -> None:
    packet, _runtime, pair_inputs, pair_activation = _real_canary_pair_inputs(tmp_path / "scene")
    capture = tmp_path / "capture"
    capture.mkdir()
    probe = tmp_path / "probe"
    probe.mkdir()
    value = _sealed({"schema_version": registry.REGISTRY_SCHEMA, "profiles": [{
        "capture_root": str(capture.resolve()), "task_id": "pick_place", "scenario_id": "nominal",
        "pair_runtime_inputs_path": str(pair_inputs), "pair_activation_path": str(pair_activation),
        "robot_free_probe_root": str(probe), "open_loop_horizon": 8, "max_hourly_rate_usd": 0.8,
        "allowed_founder_approvers": ["founder@blueprint"], "scene_id": "site-req1"}]}, "registry_digest")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value))
    path.chmod(0o644)
    monkeypatch.setenv(registry.REGISTRY_ENV, str(path))
    # The fixture's frozen task carries no rigid contract the probe can plan, so nothing is queued.
    outcome = runner.enqueue_rigid_robot_free_probe(capture, queue_root=tmp_path / "queue")
    assert outcome == {"status": "unavailable_for_task", "queued": False, "capture_root": str(capture.resolve())}
    assert not (tmp_path / "queue").exists()


# --- watchdog scope, inventory and deploy units ------------------------------------------------


def test_the_lane_label_is_watchdog_scoped_and_its_exact_names_pass_the_gate() -> None:
    from blueprint_pipeline.groot_oscar_runpod_watchdog import CANARY_NAME_PREFIXES
    from blueprint_pipeline.vast_independent_watchdog_control import validate_independent_vast_watchdog_names

    assert lane.INSTANCE_LABEL_PREFIX in CANARY_NAME_PREFIXES
    name = lane.resource_name_for("job-a00")
    assert validate_independent_vast_watchdog_names(pod_name_prefix=lane.INSTANCE_LABEL_PREFIX,
                                                    resource_name_exact=name) == (lane.INSTANCE_LABEL_PREFIX, name)
    assert lane.resource_name_for("job-a01") != name


def test_the_paid_resource_inventory_and_reachability_ledger_name_the_lane() -> None:
    root = Path(__file__).resolve().parents[1]
    relative = "src/blueprint_pipeline/rigid_robot_free_probe_lane.py"
    manifest = json.loads((root / "docs/architecture/paid-resource-mutation-surfaces.json").read_text())
    for key in ("require_paid_resource_admission", "build_paid_lane_admission"):
        assert relative in manifest["issuer_allowlist"][key]
    row = next(item for item in manifest["surfaces"] if item["path"] == relative)
    source = (root / relative).read_text()
    assert row["classification"] == "canonical_adapter"
    assert all(marker in source for marker in row["required_markers"])
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_paid_resource_allocator",
                                                  root / "scripts/verify_paid_resource_allocator.py")
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    assert relative in verifier.APPROVED_ADMISSION_ISSUERS and relative in verifier.APPROVED_LANE_ADMISSION_BUILDERS
    ledger = (root / "docs/arm_decision_proof_v1/LIVE_LANE_REACHABILITY.md").read_text()
    assert f"`{PROBE_KIND}` — `not_a_website_lane`, automatic under the capture's grant" in ledger


def test_the_runner_units_are_automatic_under_capture_grants_and_allow_opt_out(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    service = (root / "deploy/systemd/blueprint-rigid-robot-free-probe-runner.service").read_text()
    path_unit = (root / "deploy/systemd/blueprint-rigid-robot-free-probe-runner.path").read_text()
    assert f"PathChanged={runner.DEFAULT_QUEUE}/pending" in path_unit
    # A deferring pass leaves its entry pending; a level trigger would re-run it in a loop.
    assert not any(line.startswith(("PathExists", "DirectoryNotEmpty")) for line in path_unit.splitlines())
    assert "Unit=blueprint-rigid-robot-free-probe-runner.service" in path_unit
    assert "blueprint-rigid-robot-free-probe-runner.path.json" in path_unit  # operator-door hold
    assert "-m blueprint_pipeline.rigid_robot_free_probe_runner condition" in service
    assert "-m blueprint_pipeline.rigid_robot_free_probe_runner run-queue" in service
    assert service.index("ExecCondition=") < service.index("ExecStart=")
    for setting in ("KillMode=process", "User=blueprint", "ProtectSystem=strict", "NoNewPrivileges=true",
                    "BLUEPRINT_VAST_WATCHDOG_CALLER_EXIT_SURVIVAL=systemd_dispatcher_kill_mode_process",
                    f"BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE={runner.DEFAULT_QUEUE}",
                    "Environment=BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT="
                    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents"):
        assert setting in service
    # The service default implements the owner's automatic-check direction.
    # Operator env files can disable it; every pass keeps the canary attempt cap.
    assert f"Environment={lane.ENABLE_ENV}=1" in service
    assert "EnvironmentFile=-/etc/blueprint/rigid-robot-free-probe.env" in service
    assert "run-queue --max-attempts 1" in service
    assert "paid_resource_allocator" not in service
    timer = (root / "deploy/systemd/blueprint-rigid-robot-free-probe-runner.timer").read_text()
    assert "OnUnitInactiveSec=5min" in timer
    assert "blueprint-rigid-robot-free-probe-runner.timer.json" in timer
    import ast

    tree = ast.parse((root / "scripts/deploy_control_plane_commit.py").read_text())
    wanted = {"DEFAULT_DEPLOYED_SYSTEMD_UNITS", "DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS",
              "DEFAULT_ALWAYS_ARM_PATH_UNITS"}
    lists = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in wanted}
    assert set(lists) == wanted
    units = {f"blueprint-rigid-robot-free-probe-runner.{suffix}" for suffix in ("service", "path")}
    assert units <= set(lists["DEFAULT_DEPLOYED_SYSTEMD_UNITS"])
    assert "blueprint-rigid-robot-free-probe-runner.path" in lists["DEFAULT_ALWAYS_ARM_AUTHORITY_GATED_PATH_UNITS"]
    assert not units & set(lists["DEFAULT_ALWAYS_ARM_PATH_UNITS"])  # never filed as no-spend
    mount = (root / "deploy/host/mount_work_volume.sh").read_text()
    assert all(f"  {unit}\n" in mount for unit in units)  # stopped for a work-volume move
    monkeypatch.delenv(lane.ENABLE_ENV, raising=False)
    assert runner.main(["condition"]) == 1
    monkeypatch.setenv(lane.ENABLE_ENV, "true")
    assert runner.main(["condition"]) == 0
    monkeypatch.setenv(lane.ENABLE_ENV, "off")
    assert runner.main(["condition"]) == 1


# --- a staging refusal is also set aside on the shared schedule, without a deploy ---------------------


def test_a_staging_refusal_is_retried_on_the_shared_schedule_without_a_deploy_and_then_waits_for_one(
        tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS

    world = build_world(tmp_path, monkeypatch)
    admitted = physics.require_scene_physics_admission

    def overlap(packet_dir: Path, **_options: Any) -> dict[str, Any]:
        raise physics.ScenePhysicsAdmissionBlocked(["task_object:scene_physics_rest_overlap:1.3cm"])

    monkeypatch.setattr(physics, "require_scene_physics_admission", overlap)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    clock = [time.time()]
    hooks = {**_under(world, monkeypatch, COMMIT), "now": lambda: clock[0]}
    (row,) = runner.run_queue(world["queue"], hooks=hooks)
    assert row["status"] == "staging_refused" and row[runner.RELEASE_COMMIT_FIELD] == COMMIT, row
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    state = current_probe_state(world["capture"])
    history = job / runner.SUPERSEDED_DIRNAME
    for retry, wait in enumerate(REFUSAL_RETRY_DELAYS_SECONDS, start=1):
        sealed = json.loads((job / runner.TERMINAL_FILENAME).read_text())
        # Not final while a timed retry is left; nothing moves before its wait since the terminal was sealed.
        assert runner.final_terminal(state, queue_root=world["queue"], release=COMMIT) is None
        clock[0] = runner._when(sealed["sealed_at_iso"]) + wait - 5
        assert runner.run_queue(world["queue"], hooks=hooks) == []
        clock[0] += 10
        (row,) = runner.run_queue(world["queue"], hooks=hooks)
        # Set aside through the same supersession as a release's, its marker naming the clock and its count.
        marker = row["superseded_terminal"]
        assert (marker["trigger"], marker["refusal_retry"]) == (runner.REFUSAL_RETRY_TRIGGER, retry)
        assert (marker["superseded_status"], marker[runner.RELEASE_COMMIT_FIELD]) == ("staging_refused", COMMIT)
        assert json.loads((history / f"{sealed['terminal_digest'][7:23]}.terminal.json").read_text()) == sealed
        assert row["status"] == "staging_refused" and row["detail"]["attempt"] == f"{retry:02d}", row
    # Six timed set-asides, each attempt recorded not staged: nothing launched, funded or allocated.
    assert len(list(history.glob("*.supersession.json"))) == 6 and runner._refusal_retries(job) == 6
    assert [json.loads((attempt / runner.INGEST_FILENAME).read_text())["state"]
            for attempt in runner._attempts(job)] == [runner.NOT_STAGED_STATE] * 7
    assert runner._counts(job) == (0, 0) and world["allocator_calls"] == [] and world["webapp_calls"] == []
    # Spent: final under this release, however long it waits, even once its cause is gone.
    terminal = json.loads((job / runner.TERMINAL_FILENAME).read_text())
    assert runner.final_terminal(state, queue_root=world["queue"], release=COMMIT) == terminal
    monkeypatch.setattr(physics, "require_scene_physics_admission", admitted)
    clock[0] += 10 ** 7
    assert runner.run_queue(world["queue"], hooks=hooks) == []
    # A newly deployed release still gets its own try (the count is not reset), and the probe runs once,
    # through the same funded-attempt limits and allocator.
    later = "d" * 40
    (row,) = runner.run_queue(world["queue"], hooks={**_under(world, monkeypatch, later), "now": lambda: clock[0]})
    assert row["status"] == "decided" and _receipt(world)["status"] == "passed", row
    assert "trigger" not in row["superseded_terminal"] and runner._refusal_retries(job) == 6
    assert runner._counts(job) == (1, 0) and len(world["allocator_calls"]) == 1


def _staging_refusal_sealed(world: dict[str, Any], *, status: str = "staging_refused",
                            reason: str = PHYSICS_REFUSAL, release: str | None = COMMIT) -> tuple[Path, dict]:
    """A sealed terminal, retired to ``done/``, a day after it was sealed (any timed wait has passed)."""

    job, legacy = _legacy_staging_refusal(world)
    terminal = _sealed({**{key: value for key, value in legacy.items() if key != "terminal_digest"},
                        "status": status, "detail": {"reason": reason},
                        **({runner.RELEASE_COMMIT_FIELD: release} if release else {})}, "terminal_digest")
    (job / runner.TERMINAL_FILENAME).unlink()
    runner._write_exclusive(job / runner.TERMINAL_FILENAME, terminal)
    return job, terminal


DAY_AFTER = runner._when("2026-11-04T17:54:40Z")


@pytest.mark.parametrize(("status", "reason"), [
    # Privacy, rights or money in the recorded reason: never the clock's.
    ("staging_refused", "rigid_robot_free_probe_dataset_license_unverified"),
    ("staging_refused", "rigid_robot_free_probe_site_consent_withdrawn"),
    ("staging_refused", "rigid_robot_free_probe_upstream_budget_exhausted"),
    ("staging_refused", "rigid_robot_free_probe_hard_cap_usd_exceeded"),
    ("staging_refused", ""),
    # Every other terminal keeps today's rules.
    ("attempts_exhausted", PHYSICS_REFUSAL),
    ("trace_conflict", PHYSICS_REFUSAL),
    ("decided", PHYSICS_REFUSAL),
    (runner.NO_START_REPEATED_STATUS, PHYSICS_REFUSAL),
])
def test_the_clock_sets_aside_only_a_staging_refusal_that_is_no_hard_stop(tmp_path, monkeypatch, status,
                                                                           reason) -> None:
    world = build_world(tmp_path, monkeypatch)
    job, terminal = _staging_refusal_sealed(world, status=status, reason=reason)
    assert runner.requeue_new_release(world["queue"], COMMIT, now=DAY_AFTER) == []
    assert runner._supersede_for_release(job, COMMIT, DAY_AFTER) is None
    assert json.loads((job / runner.TERMINAL_FILENAME).read_text()) == terminal
    assert runner.final_terminal(current_probe_state(world["capture"]), queue_root=world["queue"],
                                 release=COMMIT) == terminal


def test_a_due_staging_refusal_is_set_aside_by_the_clock_only_under_a_known_release(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    job, terminal = _staging_refusal_sealed(world)
    state = current_probe_state(world["capture"])
    # Without a known running release nothing is retried, by a release or by the clock: it is final.
    assert runner.requeue_new_release(world["queue"], None, now=DAY_AFTER) == []
    assert runner._supersede_for_release(job, None, DAY_AFTER) is None
    assert runner.final_terminal(state, queue_root=world["queue"], release=None) == terminal
    # Not yet due: sealed 2026-11-03T17:54:40Z, its first wait is 15 minutes.
    assert runner.requeue_new_release(world["queue"], COMMIT, now=runner._when("2026-11-03T18:09:39Z")) == []
    (moved,) = runner.requeue_new_release(world["queue"], COMMIT, now=runner._when("2026-11-03T18:09:40Z"))
    assert (moved["reason"], moved["terminal_status"]) == (runner.REFUSAL_RETRY_TRIGGER, "staging_refused")
    marker = runner._supersede_for_release(job, COMMIT, DAY_AFTER)
    assert (marker["trigger"], marker["refusal_retry"], marker["superseded_terminal_digest"]) == (
        runner.REFUSAL_RETRY_TRIGGER, 1, terminal["terminal_digest"])
    assert not (job / runner.TERMINAL_FILENAME).exists()


def _marker(job: Path, name: str, **fields: Any) -> None:
    history = job / runner.SUPERSEDED_DIRNAME
    history.mkdir(exist_ok=True)
    (history / f"{name}.supersession.json").write_text(json.dumps(
        {"schema_version": runner.SUPERSESSION_SCHEMA, "superseded_at_iso": "2026-11-03T00:00:00Z", **fields}))


@pytest.mark.parametrize("history", ["spent", "unreadable", "not_an_object"])
def test_a_spent_or_unreadable_timed_count_never_sets_aside_and_a_new_release_does_not_reset_it(
        tmp_path, monkeypatch, history) -> None:
    world = build_world(tmp_path, monkeypatch)
    job, terminal = _staging_refusal_sealed(world)
    # A release's own marker is no timed retry: it leaves the count as it is.
    _marker(job, "0" * 16, superseded_status="staging_refused", **{runner.RELEASE_COMMIT_FIELD: "c" * 40})
    assert runner._refusal_retries(job) == 0
    if history == "spent":
        for index in range(6):
            _marker(job, f"{index + 1:016x}", trigger=runner.REFUSAL_RETRY_TRIGGER, refusal_retry=index + 1)
    else:
        (job / runner.SUPERSEDED_DIRNAME / ("f" * 16 + ".supersession.json")).write_text(
            "{not json" if history == "unreadable" else "[]")
    assert runner._refusal_retries(job) == 6
    assert runner.requeue_new_release(world["queue"], COMMIT, now=DAY_AFTER + 10 ** 7) == []
    assert runner._supersede_for_release(job, COMMIT, DAY_AFTER + 10 ** 7) is None
    state = current_probe_state(world["capture"])
    assert runner.final_terminal(state, queue_root=world["queue"], release=COMMIT) == terminal
    if history == "spent":
        # A new release gets its once-per-release try; the timed count stays spent after it.
        (moved,) = runner.requeue_new_release(world["queue"], "b" * 40, now=DAY_AFTER)
        assert moved["reason"] == "new_release"
        assert "trigger" not in runner._supersede_for_release(job, "b" * 40, DAY_AFTER)
        assert runner._refusal_retries(job) == 6
