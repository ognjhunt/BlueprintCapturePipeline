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
from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import team_candidate_quick10_registry as registry
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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


def website_context() -> dict[str, Any]:
    return _sealed({"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
                    "capture_id": "walkthrough-req1", "description": "Move the mug onto the tray",
                    "confirmed": True, "confirmed_at": "2026-10-26T01:00:00Z", "operator_answers": {},
                    "unresolved": []}, "context_digest")


def website_sponsorship(context: dict[str, Any], *, expires_in: float = 20 * 3600) -> dict[str, Any]:
    return _sealed({"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint",
                    **{key: context[key] for key in ("request_id", "scene_id", "capture_id")},
                    "task_context_digest": context["context_digest"],
                    "expires_at_epoch": time.time() + expires_in}, "authority_digest")


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

    def build(plan: dict[str, Any], *, device: str, bundle_root: Path, preconstruction_receipt: Any) -> Any:
        assert (Path(bundle_root) / PLAN).is_file()
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
                "blockers": [] if completed else list(result["blockers"]) or ["rigid_robot_free_probe_runtime_not_completed"]}

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
        "published": [], "launches": [], "host_options": dict(host_options), "continuing_spend": False,
        "allocator_calls": []}
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
                 "BLUEPRINT_ATLAS_OPERATOR_ADMISSION_FILE", "BLUEPRINT_VAST_MAX_ISAAC_DRIVER", runner.QUEUE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(lane.ENABLE_ENV, "1")

    def webapp(*, capture_id: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
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
    ("grant_missing", "rigid_robot_free_probe_website_grant_missing", False),
    ("grant_expired", "rigid_robot_free_probe_website_grant_expired", False),
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
    elif case == "grant_missing":
        (pipeline / "website_scene_sponsorship.json").unlink()
    elif case == "grant_expired":
        (pipeline / "website_scene_sponsorship.json").write_text(
            json.dumps(website_sponsorship(world["context"], expires_in=600)))
    elif case.startswith("webapp_"):
        world["webapp_refusal"] = {"webapp_expired": "consent_expired",
                                   "webapp_exhausted": "website_scene_preparation_budget_exhausted",
                                   "webapp_missing": "website_scene_sponsorship_missing"}[case]
    elif case == "decided":
        (world["probe_root"] / gate.RECEIPT_FILENAME).write_text(json.dumps({"status": "failed"}))
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
    def uncertain_create(argv, *, attempt):
        world["allocator_calls"].append(argv)
        runner._write_exclusive(attempt / "invocation/provider_create_intent.json", {"outcome": "unknown"})
        return 0
    hooks["allocator_runner"] = uncertain_create
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "unknown"
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "unknown"
    assert len(world["allocator_calls"]) == 1 and world["transport_calls"] == []


def test_a_refusal_before_funding_defers_to_the_next_wake_up_and_is_bounded(tmp_path, monkeypatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    world["webapp_refusal"] = "website_scene_preparation_budget_exhausted"
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    for attempt in range(runner.MAX_UNFUNDED_ATTEMPTS - 1):
        row = runner.run_queue(world["queue"], hooks=hooks)[0]
        assert row["status"] == "deferred", row
        assert "rigid_robot_free_probe_website_grant_exhausted" in row["blockers"]
        assert len(world["allocator_calls"]) == attempt + 1  # one attempt per wake-up, never a tight loop
    final = runner.run_queue(world["queue"], hooks=hooks)[0]
    assert final["status"] == "attempts_exhausted" and final["detail"]["unfunded_attempts"] == 4, final
    assert world["transport_calls"] == [] and _receipt(world) is None


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


def test_default_preflight_honors_dry_argv_with_paid_manifest_present(tmp_path, monkeypatch) -> None:
    """Exercise the default wrapper, not the test hook that calls allocator.main."""
    from blueprint_pipeline import allocator_invocation as jobs

    world = build_world(tmp_path, monkeypatch)
    commands = []

    class DryProcess:
        pid = 12345

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert command[1:3] == ["-m", "blueprint_pipeline.paid_resource_allocator"]
            assert "--execute" not in command
            self.code = allocator.main(command[3:])

        def wait(self):
            return self.code

    monkeypatch.setattr(runner.subprocess, "Popen", DryProcess)
    monkeypatch.setattr(jobs, "start_invocation", lambda *a, **k: pytest.fail("dry preflight started paid wrapper"))
    result = runner.preflight_capture(world["capture"], scratch_dir=tmp_path / "scratch",
                                      hooks={"source_commit": lambda: COMMIT})
    attempt = Path(result["attempt_root"])
    assert result["status"] == "ready" and len(commands) == 1
    assert "--execute" in jobs.manifest(attempt / "invocation/manifest.json")["argv"]
    assert not (attempt / "invocation/provider_create_intent.json").exists()
    assert not (attempt / "allocator" / lane.FUNDING_FILENAME).exists()
    assert world["webapp_calls"] == []
    assert world["transport_calls"] and all(call["execute"] is False for call in world["transport_calls"])


def test_default_paid_wrapper_requires_exact_retained_argv(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import allocator_invocation as jobs

    world = build_world(tmp_path, monkeypatch)
    staged = staged_attempt(world)
    monkeypatch.setattr(jobs, "start_invocation", lambda *a, **k: pytest.fail("changed paid arguments started"))
    with pytest.raises(ValueError, match="invocation_argv_mismatch"):
        runner.run_allocator(_with(staged["argv"], "--adp-max-spend-usd", "0.01"), attempt=staged["root"])
    assert world["webapp_calls"] == [] and world["transport_calls"] == []


@pytest.mark.parametrize("remote_status", ["reserved", "not_reserved"])
def test_funding_response_loss_retains_exact_request_and_never_opens_new_attempt(
        tmp_path, monkeypatch, remote_status) -> None:
    from blueprint_pipeline import allocator_invocation as jobs
    from blueprint_pipeline import probe_funding_recovery as recovery
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest

    class SimulatedPowerLoss(BaseException):
        pass

    world = build_world(tmp_path, monkeypatch)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    staged = staged_attempt(world)
    attempt, grants, reads = staged["root"], [], []
    runner._write_exclusive(attempt / runner.INTENT_FILENAME, {"observed_at_iso": "test"})

    def remote(*, capture_id, operation, payload):
        request = recovery.request_intent(attempt / "allocator")
        assert request["payload"] == {key: value for key, value in payload.items() if key != "create"}
        assert request["capture_id"] == capture_id and operation == "preparation-spend"
        if payload.get("create") is not False:
            grants.append(payload["spend"])
            raise SimulatedPowerLoss("remote transaction committed before response retention")
        reads.append(payload)
        command = payload["spend"]
        return {"schema_version": "website_preparation_spend_observation.v1", "capture_id": capture_id,
                "request_id": payload["request_id"], "scene_id": payload["scene_id"],
                "allocation_binding_digest": command["allocation_binding_digest"],
                "command_digest": cross_runtime_canonical_digest(command), "status": remote_status,
                "dispatch_authorized": False, "observed_at_epoch": time.time(),
                **({"admission": {**command, "schema_version": "paid_lane_admission.v1", "status": "admitted",
                    "blockers": [], "external_disclosure_allowed": True, "expires_at_epoch": time.time() + 60}}
                   if remote_status == "reserved" else {})}

    monkeypatch.setattr(website_task_context, "website_webapp_request", remote)
    with pytest.raises(SimulatedPowerLoss):
        jobs.run_invocation(attempt / "invocation/manifest.json", allocator=allocator.main)
    assert len(grants) == 1 and not (attempt / "allocator" / lane.FUNDING_FILENAME).exists()
    assert not (attempt / "invocation/provider_create_intent.json").exists()
    settled = runner.settle_attempt(attempt, state=current_probe_state(world["capture"]))
    assert settled["state"] == "unknown" and settled["last_remote_status"] == remote_status
    assert settled["allocation_binding_digest"] == grants[0]["allocation_binding_digest"]
    assert not (attempt / runner.INGEST_FILENAME).exists()
    assert runner._counts(attempt.parent.parent) == (1, 0)
    assert len(reads) == 1 and reads[0]["create"] is False
    _, blocked = lane.reserve_probe_funding(task_context=world["context"],
        binding_digest=grants[0]["allocation_binding_digest"], hard_cap_usd=lane.HARD_CAP_USD,
        job_dir=attempt / "allocator")
    assert blocked == ["rigid_robot_free_probe_funding_request_reconciliation_required"]
    assert jobs.run_invocation(attempt / "invocation/manifest.json", allocator=lambda argv: pytest.fail(
        "fenced old allocator restarted")) == 2
    monkeypatch.setattr(allocator, "main", lambda *a, **k: pytest.fail("response loss allowed another dispatch"))
    row = runner.run_queue(world["queue"], hooks={"source_commit": lambda: COMMIT})[0]
    assert row["status"] == "waiting" and row["next_action"].startswith("reconcile_exact_funding_")
    assert len(runner._attempts(attempt.parent.parent)) == 1 and len(grants) == len(reads) == 1
    assert world["transport_calls"] == []


def test_old_unfunded_ingest_cannot_hide_legacy_unknown_funding(tmp_path):
    from blueprint_pipeline import allocator_invocation as jobs
    attempt = tmp_path / "attempts" / "00"
    attempt.mkdir(parents=True)
    path = jobs.retain_invocation(attempt_dir=attempt, binding={"provider": "vast"}, argv=["gpu-canary"])
    (path.parent / "started.json").write_text("{}")
    retained = runner._record_ingest(attempt, {"state": "not_launched", "funded": False, "allocated": False})
    outcome = runner.settle_attempt(attempt, state={})
    assert outcome["state"] == "unknown" and outcome["next_action"] == "recover_exact_legacy_funding_request"
    assert runner._counts(tmp_path) == (1, 0)
    assert json.loads((attempt / runner.INGEST_FILENAME).read_text()) == retained


@pytest.mark.parametrize("pid_receipt_survives", [True, False])
def test_frozen_pre_aut08_bare_allocator_response_loss_never_becomes_unfunded(
        tmp_path, monkeypatch, pid_receipt_survives):
    world = build_world(tmp_path, monkeypatch)
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    staged = staged_attempt(world)
    attempt = staged["root"]
    # The historical runner never created a named invocation. Retain the
    # unused modern staging bytes separately so this exercises its bare path.
    (attempt / "invocation").rename(attempt / "modern_staging_unused")
    runner._write_exclusive(attempt / runner.INTENT_FILENAME, {"observed_at_iso": "legacy"})
    fixture = Path(__file__).parent / "fixtures/aut08_legacy_probe_funding.py.txt"
    legacy_namespace = dict(vars(lane))
    exec(compile(fixture.read_text(), str(fixture), "exec"), legacy_namespace)
    monkeypatch.setattr(lane, "reserve_probe_funding", legacy_namespace["reserve_probe_funding"])
    remote_admissions = []

    def commit_then_lose_response(**kwargs):
        remote_admissions.append(kwargs["payload"]["spend"])
        raise OSError("legacy funding response lost after commit")

    class BareProcess:
        pid = 12346

        def __init__(self, command, **kwargs):
            assert command[1:3] == ["-m", "blueprint_pipeline.paid_resource_allocator"]
            self.code = allocator.main(command[3:])

        def wait(self):
            return self.code

    monkeypatch.setattr(website_task_context, "website_webapp_request", commit_then_lose_response)
    monkeypatch.setattr(runner.subprocess, "Popen", BareProcess)
    assert runner.run_allocator(staged["argv"], attempt=attempt) == 2
    if not pid_receipt_survives:
        (attempt / runner.PROCESS_FILENAME).rename(attempt / "lost_pid_receipt_unused.json")
    assert len(remote_admissions) == 1 and not (attempt / "invocation").exists()
    from blueprint_pipeline import probe_funding_recovery as recovery
    assert not (attempt / "allocator" / lane.FUNDING_FILENAME).exists()
    assert not (attempt / "allocator" / recovery.REQUEST_FILENAME).exists()
    state = current_probe_state(world["capture"])
    assert runner.settle_attempt(attempt, state=state)["state"] == "unknown"
    assert not (attempt / runner.INGEST_FILENAME).exists()
    retained_bad_ingest = runner._record_ingest(attempt, {"state": "unfunded_blocked", "funded": False,
                                                       "allocated": False})
    assert runner.settle_attempt(attempt, state=state)["state"] == "unknown"
    assert json.loads((attempt / runner.INGEST_FILENAME).read_text()) == retained_bad_ingest
    assert runner._counts(attempt.parent.parent) == (1, 0)
    row = runner.run_queue(world["queue"], hooks={"source_commit": lambda: COMMIT})[0]
    assert row["status"] == "waiting" and row["next_action"] == "recover_exact_legacy_funding_request"
    assert len(remote_admissions) == 1 and len(runner._attempts(attempt.parent.parent)) == 1
    assert world["transport_calls"] == []


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
                    f"BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE={runner.DEFAULT_QUEUE}"):
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
