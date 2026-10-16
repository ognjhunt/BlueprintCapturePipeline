"""Hermetic launch rehearsal: one team evaluation request, executor claim to scoring statement.

Beta journey step: L5 team policy evaluation. A free-beta request for a team's
scored evaluation (200 hidden cells) is staged by the real agent-run executor
(``poll_once``) through the real WebApp client methods, the real operator
registry and eligibility rule, the real admission, staging, bundle builder and
static preflight; run by the real team evaluation runner (``run_queue``)
through the real canonical allocator (``paid_resource_allocator gpu-canary``),
lane and team transport, the first shard alone and then the rest behind its
gate (``first_shard_gate``); closed by the real closeout and
terminal; and reported by the real ``report_completed`` on the next poll.

Faked, and only these: the WebApp's HTTP replies, the credential broker's
reply (``CheckpointPolicyCredentialClient.access``), the team's HTTPS service,
provider-zero inventory, the launch environment's watchdog and spend-lock
probes, and the Vast transport plus GPU host (``run_arena_native_control_vast``).
That last fake reads the token only through the real runtime-secret reader,
verifies the sealed bundle with the real worker's static preflight, and has the
real team client complete its identity handshake with that token. The ten
episode records it returns stand in for the GPU worker, whose real team run is
pinned in ``tests/test_team_candidate_quick10_rehearsal.py``.

The rehearsal scene is rigid. Its sealed probe receipt is read through the
gate's retention API; until the rigid probe lands on this branch, the gate's
task-kind set is widened for this scene only and every other check runs (the
receipt takes the rigid schema and checks once the gate has them). No GPU,
provider, network or spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import pytest

import blueprint_pipeline.droid_https_policy_client as https_client
from blueprint_pipeline import agent_run_executor as executor
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import policy_canary_allocator_lane as lane
from blueprint_pipeline import robot_free_probe_gate as probe_gate
from blueprint_pipeline import team_candidate_quick10_launch as launch
from blueprint_pipeline import team_candidate_quick10_vast as team_vast
from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline.agent_execution_offer import publish_agent_execution_offer
from blueprint_pipeline.agent_run_result_scoring import validate_scoring_statement
from blueprint_pipeline.checkpoint_policy_credentials import bearer_lease_id
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_worker import (
    preflight_policy_canary_runtime_directory,
)
from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
from blueprint_pipeline.team_candidate_quick10_admission import RIGHTS_SCHEMA
from blueprint_pipeline.team_candidate_quick10_closeout import read_team_candidate_terminal
from blueprint_pipeline.team_candidate_quick10_launch import CREDENTIAL_ENV, TERMINAL_FILENAME
from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from blueprint_pipeline.team_robot_free_probe import scored_task_spec
from tests.test_agent_run_executor import _row
from tests.test_provider_runtime_import_closure import _real_canary_pair_inputs

TOKEN = "team-bearer-0123456789abcdef0123456789"
ORIGIN = "https://policy.acme-robotics.example"
ENDPOINT = ORIGIN + "/v1/act"
CREDENTIAL_REF = "policy-credential-1b4e28ba-2fa1-11d2-883f-0016d3cca427"
TEAM, CHECKPOINT, CANDIDATE = "acme", "pi05-ft-7", "team-acme.pi05-ft-7"
MODEL_DIGEST = "sha256:" + "7" * 64
SALT = "5" * 64


def _verdict(cell_id: str) -> int:
    return int(hashlib.sha256(cell_id.encode()).hexdigest(), 16) % 10


def _succeeds(cell_id: str) -> bool:
    """The stand-in worker's verdict: seven in ten cells succeed, deterministically by cell."""
    return _verdict(cell_id) < 7


def _times_out(cell_id: str) -> bool:
    """One failing cell in ten is the team's service not answering: a failed episode, never a void."""
    return _verdict(cell_id) == 9


ZERO = {"schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1", "status": "provider_zero_confirmed",
        "provider": "vast", "api_confirmed": True, "live_instance_count": 0, "provider_zero_verified": True,
        "blockers": []}


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _metadata() -> dict[str, Any]:
    return {"schema_version": https_client.METADATA_SCHEMA, "policy_id": "acme-pi05-ft-7",
            "policy_interface_id": "openpi_droid_https.v1", "action_space": "joint_position",
            "action_chunk_rows": 10, "model_identity_digest": MODEL_DIGEST}


class _TeamService:
    """The team's HTTPS service: answers only its own origin, only with the bearer token."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.metadata = metadata or _metadata()

    def transport(self, *, url: str, method: str, body: Any, headers: dict[str, str]) -> bytes:
        assert headers["Authorization"] == "Bearer " + TOKEN and url == ENDPOINT and method == "GET"
        self.calls.append(("launch_metadata", method))
        return json.dumps(self.metadata).encode()

    def __call__(self, url: str, **kwargs: Any) -> Any:
        assert kwargs["allowed_origins"] == (ORIGIN,) and kwargs["max_redirects"] == 0
        assert kwargs["headers"]["Authorization"] == "Bearer " + TOKEN and url == ENDPOINT
        self.calls.append(("provider_" + kwargs["method"].lower(), kwargs["method"]))

        class Response:
            body = json.dumps(self.metadata).encode()
        return Response()


class _WebApp(executor.AgentRunWebAppClient):
    """The real client's methods over a fake WebApp: claim, run state, results and settlements."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row, self.owner, self.posted, self.blocked = row, None, [], []

    def _json(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if path.startswith("/api/internal/pipeline/agent-runs?"):
            return {"runs": [self.row] if self.owner is None else []}
        if path.endswith("/started"):
            self.owner = payload["pipeline_run_id"]
            return {"pipeline_run_id": self.owner}
        if path.startswith("/api/internal/pipeline/agent-runs/"):
            return {"state": "requested", "money_resolved": False, "cancellation_requested": False,
                    "dispatch": {"pipeline_run_id": self.owner},
                    "execution_admission_digest": self.row["execution_admission_digest"]}
        self.posted.append((path, payload))
        if payload.get("blocked_before_any_episode") or payload.get("execution_status") == "blocked":
            self.blocked.append(payload["reason"])
        return {"ok": True}


class _Broker:
    """``CheckpointPolicyCredentialClient``'s single-use bearer lease, as the WebApp must answer it."""

    accessed: ClassVar[list[str]] = []
    leases: ClassVar[list[str]] = []

    def __init__(self, *, request: dict[str, Any], payload: dict[str, Any], pipeline_run_id: str) -> None:
        assert payload["credential_ref"] == CREDENTIAL_REF and pipeline_run_id.startswith("agent-attempt-")
        self.request, self.pipeline_run_id = request, pipeline_run_id

    def release_bearer_lease(self, *, lease_id: str) -> dict[str, Any]:
        assert lease_id == bearer_lease_id(credential_ref=CREDENTIAL_REF, job_id=self.request["job_id"],
                                           pipeline_run_id=self.pipeline_run_id)
        if lease_id in _Broker.leases:  # the WebApp releases a lease's token once
            raise ValueError("checkpoint_policy_credential_delivery_unavailable")
        _Broker.accessed.append(self.request["job_id"])
        _Broker.leases.append(lease_id)
        return {"ok": True, "credential_ref": CREDENTIAL_REF, "kind": "bearer",
                "credential": {"job_id": self.request["job_id"], "endpoint_url": ENDPOINT, "bearer_token": TOKEN},
                "lease": {"lease_id": lease_id, "single_use": True, "status": "released", "release_count": 1}}


def _provider_session_result(runtime: Path, spec: dict[str, Any], *, probe_status: str = "passed",
                             service_down: bool = False) -> dict[str, Any]:
    """Stand in for the GPU worker's simulator-scored episode records (shape of the real worker's)."""

    inputs = json.loads((runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json").read_text())
    probe = {"robot_free_probe_trace": {"trace_digest": "sha256:" + "6" * 64}, "robot_free_probe_status": probe_status}
    if spec.get("robot_free_probe") is not None and probe_status != "passed":
        # The worker's cell-0 gate: no team episode on a scene whose check did not pass.
        blocked = {"schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
                   "run_kind": inputs["run_kind"], "claim_ceiling": "diagnostic_policy_execution",
                   "task_success_contract": inputs["task_success_contract"],
                   "task_success_contract_digest": inputs["task_success_contract_digest"], "episodes": [],
                   "blockers": ["team_scene_robot_free_probe_not_passed"], **probe}
        return _sealed(blocked, "result_digest")
    episodes = []
    for cell in inputs["cells"]:
        succeeded = _succeeds(cell["cell_id"])
        digests = {field: canonical_digest({"cell": cell["cell_id"], "field": field}) for field in (
            "lossless_frame_manifest_digest", "review_video_digest", "returned_action_sequence_digest",
            "action_delivery_readback_digest", "state_trace_digest", "contact_force_digest",
            "task_object_trajectory_digest", "deterministic_score_digest")}
        row = {
            "candidate_id": spec["candidate_id"], "cell_id": cell["cell_id"], "seed": cell["seed"],
            "cell_spec_digest": cell["cell_spec_digest"], "resolved_scenario": cell["resolved_scenario"],
            "resolved_scenario_digest": cell["resolved_scenario_digest"], "run_kind": inputs["run_kind"],
            "claim_ceiling": "diagnostic_policy_execution", "ranking_eligible": False, "status": "completed",
            "checkpoint_digest": spec["checkpoint_digest"], "runtime_identity_digest": spec["runtime_identity_digest"],
            "candidate_policy_queried": True, "actions_reached_robot": True, "arm_moved": True,
            "scoring_authority": "deterministic_simulator_state", "visual_evidence": {"status": "complete"},
            "episode": {"score": {"status": "scored", "task_succeeded": succeeded}}, **digests}
        if service_down or _times_out(cell["cell_id"]):
            # The worker's retained row when the team's service stops answering mid-episode.
            row.update(status="failed", episode_failure_stage="after_first_observation", episode={},
                       candidate_policy_query_attempted=True, policy_response_status="timeout",
                       failure_message="team_policy_https_timeout", first_observation_retained=True,
                       policy_outcome_interpretable=False,
                       visual_evidence={"status": "complete", "human_review_available": True,
                                        "episode_terminal_status": "failed_after_first_observation",
                                        "terminal_observation_invented": False})
        episodes.append(row)
    result = {"schema_version": "native_task_arena_policy_canary_session_result.v1",
              "status": "runtime_completed_unqualified_pending_closeout", "run_kind": inputs["run_kind"],
              "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": inputs["candidate_ids"],
              "task_success_contract": inputs["task_success_contract"],
              "task_success_contract_digest": inputs["task_success_contract_digest"],
              "episodes_per_policy": len(episodes), "learned_policy_rollout_count": len(episodes),
              "provider_allocations_observed": None, "retry_cap": 0,
              "warm_session_open_count": 1, "episodes": episodes, "session_closeout": {"status": "pending"},
              "scene_promotion_performed": False, "official_ranking_performed": False,
              "candidate_policy_queried": True, "blockers": [],
              **(probe if spec.get("robot_free_probe") is not None else {})}
    return _sealed(result, "result_digest")


class _Provider:
    """Vast plus the GPU host, behind the real team transport's one call."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: _TeamService) -> None:
        self.tmp_path, self.monkeypatch, self.service, self.calls = tmp_path, monkeypatch, service, []
        self.probe_status, self.service_down = "passed", False

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        assert set(kwargs["runtime_secret_file_paths"]) == {CREDENTIAL_ENV}
        assert kwargs["forward_hf_token"] is False and kwargs["execute"] is True
        assert kwargs["paid_resource_admission_grant"] is not None
        # The provider reads the token only through the real runtime-secret channel reader.
        secrets = vast_provider_adapter._runtime_secret_file_values(kwargs["runtime_secret_file_paths"])
        assert secrets == {CREDENTIAL_ENV: TOKEN}
        bundle_path = Path(kwargs["prepared_bundle"]["bundle_path"])
        host = self.tmp_path / "gpu-host"
        with zipfile.ZipFile(bundle_path) as archive:
            assert not any(TOKEN.encode() in archive.read(member) for member in archive.infolist())
            archive.extractall(host)
        runtime = host / "provider_runtime"
        assert preflight_policy_canary_runtime_directory(runtime)["status"] == "passed"
        # The host bootstrap: a 0600 file in a 0700 directory, its path exported under the spec's name.
        secret_root = host / ".blueprint-runtime-secrets"
        secret_root.mkdir(mode=0o700)
        (secret_root / CREDENTIAL_ENV.lower()).write_text(secrets[CREDENTIAL_ENV])
        (secret_root / CREDENTIAL_ENV.lower()).chmod(0o600)
        self.monkeypatch.setenv(CREDENTIAL_ENV, str(secret_root / CREDENTIAL_ENV.lower()))
        spec = json.loads((runtime / "runtime_inputs" / f"policy_execution_spec.{CANDIDATE}.json").read_text())
        assert canary_policy_client(spec).preflight_readiness()["identity_verified"] is True
        attempt = Path(kwargs["job_dir"]) / "attempt-1"
        attempt.mkdir(parents=True)
        result = _provider_session_result(runtime, spec, probe_status=self.probe_status,
                                          service_down=self.service_down)
        # The host is released with its instance; the next shard gets a fresh one.
        self.monkeypatch.delenv(CREDENTIAL_ENV)
        shutil.rmtree(host)
        result_path = attempt / "native_task_arena_policy_canary_session_result.v1.json"
        result_path.write_text(json.dumps(result))
        return {"status": "completed", "blockers": [], "vast_instance_ids": [7001], "provider_mutations_performed": 1,
                "provider_create_attempted": True, "continuing_spend_from_this_run": False,
                "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
                                      "all_staged_objects_absent": True},
                "bundle_sha256": kwargs["prepared_bundle"]["bundle_sha256"],
                "native_control_result_path": str(result_path), "native_control_result_digest": result["result_digest"]}


def _team_scene_record(packet: Path, runtime: Path, inputs: dict[str, Any]) -> Path:
    """What ``team_scene_packet`` seals over a prepared scene's own compiled packet (no prior run)."""

    from blueprint_pipeline import team_scene_packet as scene_packet

    # A compiled plan names its scene and task, as its receipt does.
    receipt_path, plan_path = (packet / "native_task_arena_packet_receipt.v1.json",
                               packet / "native_task_arena_scene_plan.v1.json")
    receipt = json.loads(receipt_path.read_text())
    plan = {**json.loads(plan_path.read_text()), "scene_id": receipt["scene_id"], "task_id": receipt["task_id"]}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    for artifact in receipt["artifacts"]:
        if artifact["relative_path"] == plan_path.name:
            artifact.update(size_bytes=plan_path.stat().st_size,
                            sha256="sha256:" + hashlib.sha256(plan_path.read_bytes()).hexdigest())
    receipt.update(arena_scene_plan_digest=plan["plan_digest"], receipt_digest="")
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    def record(path):
        return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
    value = {"schema_version": scene_packet.SCHEMA, "claim_ceiling": scene_packet.CLAIM_CEILING,
             **{key: inputs[key] for key in ("scene_revision_digest", "configuration_digest", "plan_digest",
                                             "task_success_contract", "task_success_contract_digest")},
             "base_native_packet": record(receipt_path), "runtime_source": record(runtime),
             "construction_result": record(receipt_path), "execution_authority": scene_packet.EXECUTION_AUTHORITY,
             "control_diagnostic": scene_packet.PENDING_CONTROLS, "robot_station": {"stance_digest": "fixture"},
             "droid_profile_digest": "sha256:" + "0" * 64, "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path = packet.parent / scene_packet.RECORD_FILENAME
    path.write_text(json.dumps(value))
    return path


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, approver: str = "founder@blueprint",
           robot: dict[str, Any] | None = None, derived: bool = False,
           scene_half: str = "frozen_pair") -> dict[str, Any]:
    capture = tmp_path / "capture"
    capture.mkdir()
    packet, runtime, pair_inputs, pair_activation = _real_canary_pair_inputs(tmp_path / "scene")
    inputs = json.loads(pair_inputs.read_text())
    task_spec = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text())["task_spec"]
    operator = tmp_path / "operator"
    operator.mkdir()
    # The receipt as the scene's probe retains it beside its source launch, read through the gate.
    rigid_schema = getattr(probe_gate, "RIGID_RECEIPT_SCHEMA", None)
    checks = (dict.fromkeys(probe_gate.REQUIRED_CHECKS[task_spec["task_kind"]], True) if rigid_schema
              else {"zero_action_not_success": True, "direct_drive_success": True})
    probe = _sealed({"schema_version": rigid_schema or probe_gate.RECEIPT_SCHEMA, "status": "passed",
                     "configured_scene_revision_digest": inputs["scene_revision_digest"],
                     # Bound to the specification a team is scored with (team_robot_free_probe).
                     "task_spec_digest": canonical_digest(scored_task_spec(task_spec, inputs["task_success_contract"])),
                     "learned_policy_evaluation_admitted": True,
                     "task_success_contract_digest": inputs["task_success_contract_digest"],
                     "checks": checks, "proof_boundary": {"robot_embodiment_used": False}}, "receipt_digest")
    (operator / "probe").mkdir()
    (operator / "probe" / probe_gate.RECEIPT_FILENAME).write_text(json.dumps(probe))
    rights = _sealed({"schema_version": RIGHTS_SCHEMA, "status": "admitted", "team_id": TEAM,
                      "candidate_id": CANDIDATE, "checkpoint_id": CHECKPOINT, "license_to_run_granted": True,
                      "policy_self_grading_permitted": False, "official_ranking_permitted": False,
                      "secret_material_recorded": False, "adp_qualified_evidence_permitted": False}, "rights_digest")
    (operator / "rights.json").write_text(json.dumps(rights))
    registry = _sealed({"schema_version": REGISTRY_SCHEMA, "profiles": [{
        "capture_root": str(capture.resolve()), "task_id": "pick_place", "scenario_id": "nominal",
        "pair_runtime_inputs_path": str(pair_inputs), "pair_activation_path": str(pair_activation),
        "robot_free_probe_root": str(operator / "probe"), "open_loop_horizon": 8,
        "max_hourly_rate_usd": 0.8, "allowed_founder_approvers": ["founder@blueprint"],
        "scene_id": "site-req1", "scored_matrix_salt": SALT,
        "team_rights": {f"{TEAM}/{CHECKPOINT}": str(operator / "rights.json")}}]}, "registry_digest")
    if derived:
        # Nothing about this scene is written by hand: the operator names only who may fund a run, at
        # what rate, each team's rights and the salt key; the scene's row is written when its inputs land.
        scene_row = registry["profiles"][0]
        (operator / "salt.key").write_text("k" * 48)
        (operator / "salt.key").chmod(0o600)
        registry = _sealed({"schema_version": REGISTRY_SCHEMA, "profiles": [], "derived_profiles": {
            "root": str(operator / "derived"), "scored_matrix_salt_key_path": str(operator / "salt.key"),
            "defaults": {key: scene_row[key] for key in ("allowed_founder_approvers", "max_hourly_rate_usd",
                                                         "open_loop_horizon", "team_rights")}}}, "registry_digest")
    (operator / "registry.json").write_text(json.dumps(registry))
    (operator / "registry.json").chmod(0o644)
    monkeypatch.setenv(REGISTRY_ENV, str(operator / "registry.json"))
    if derived:
        from blueprint_pipeline.team_candidate_quick10_registry import write_derived_profile

        half = ({"team_scene_inputs_path": str(_team_scene_record(packet, runtime, inputs))}
                if scene_half == "team_scene" else
                {key: scene_row[key] for key in ("pair_runtime_inputs_path", "pair_activation_path")})
        written = write_derived_profile({**{key: scene_row[key] for key in (
            "capture_root", "scene_id", "task_id", "scenario_id", "robot_free_probe_root")}, **half,
            "scene_revision_digest": inputs["scene_revision_digest"]})
        assert written["status"] == "written", written
    if not rigid_schema:  # before the rigid probe lands, its kind is admitted for this rigid scene only
        monkeypatch.setattr(probe_gate, "PROBE_TASK_KINDS", frozenset({"articulated_open_close", task_spec["task_kind"]}))
    row = _row(capture)
    admission = row["execution_admission"]
    canonical = admission["canonical_execution_request"]
    canonical["customer"]["id"] = row["team_id"] = admission["binding"]["team_id"] = TEAM
    row["checkpoint"]["checkpoint_id"] = admission["binding"]["checkpoint_id"] = CHECKPOINT
    canonical["robot_profile"]["robot_profile_id"] = CHECKPOINT
    episodes = 200
    canonical["policy_package"] = {"policy_api_endpoint": {
        "endpoint_url": ENDPOINT, "execution_profile": "team_evaluation_scored_v1",
        "policy_interface_id": "openpi_droid_https.v1", "credential_ref": CREDENTIAL_REF, "credential_kind": "bearer"}}
    canonical["execution_authorization"].update(episodes=episodes, max_cost_usd=20)
    admission["funding"].update(cap_usd=20, approved_by=approver)
    row["quoted_episodes"] = episodes
    frozen = json.dumps(admission, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    row.update(execution_admission_canonical_json=frozen,
               execution_admission_digest="sha256:" + hashlib.sha256(frozen.encode()).hexdigest())
    service = _TeamService()
    if robot is not None:  # the team's own arm, declared in the request and named by its service
        canonical["policy_package"]["policy_api_endpoint"].update(
            policy_interface_id="blueprint_policy_https.v1", robot_embodiment=robot["manifest"])
        frozen = json.dumps(admission, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        row.update(execution_admission_canonical_json=frozen,
                   execution_admission_digest="sha256:" + hashlib.sha256(frozen.encode()).hexdigest())
        declared = validate_embodiment_manifest(robot["manifest"])
        service = _TeamService({"schema_version": "blueprint.policy_https_metadata.v1", "policy_id": "acme-arm",
                                "policy_interface_id": "blueprint_policy_https.v1", "action_space": "joint_position",
                                "action_chunk_rows": declared["action_chunk_rows"],
                                "model_identity_digest": MODEL_DIGEST,
                                "embodiment_id": declared["embodiment_id"]})
    monkeypatch.setattr(https_client, "fetch_bounded_https", service)
    provider = _Provider(tmp_path, monkeypatch, service)
    monkeypatch.setattr(team_vast, "run_arena_native_control_vast", provider)
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": "a" * 40}))
    monkeypatch.setattr(lane, "_launch_environment_blockers", lambda *args: [])
    monkeypatch.delenv(lane.REQUIRE_INTEGRATION_CANARY_ENV, raising=False)
    allocator_calls: list[list[str]] = []

    def start_allocator(argv: list[str], *, log_path: Path) -> int:
        token_file = Path(argv[argv.index("--team-candidate-credential-file") + 1])
        assert token_file.stat().st_mode & 0o777 == 0o600 and token_file.read_text().strip() == TOKEN
        assert TOKEN not in " ".join(argv) and "--execute" in argv
        allocator_calls.append(list(argv))
        log_path.write_text("")
        allocator.main(argv)  # the runner polls it; this stand-in returns before the first poll
        return 2**22 + 7  # no such process

    _Broker.accessed, _Broker.leases = [], []
    zeros: list[dict[str, Any]] = []
    queue = tmp_path / "team-evaluations"
    def provider_zero() -> dict[str, Any]:
        zeros.append(dict(ZERO))
        return dict(ZERO)

    hooks = {"credential_client": _Broker, "metadata_transport": service.transport, "queue_root": queue,
             "provider_zero": lambda: world["provider_zero"](), "implementation_commit": "a" * 40}
    if robot is not None:
        hooks["model_fetch"] = lambda url, target: shutil.copyfile(robot["model"], target)
    runner = {"allocator_starter": start_allocator, "provider_zero": lambda: world["provider_zero"](),
              "sleep": lambda _seconds: None,
              "execution_monitor": lambda control: {"ok": True, "run_id": control["run_id"],
                                                     "execution_deadline_at_ms": 4_000_000_000_000}}
    world = {"capture": capture, "row": row, "client": _WebApp(row), "hooks": hooks, "runner": runner,
             "probe_receipt": probe, "operator": operator,
             "queue": queue, "provider": provider, "provider_zero": provider_zero, "service": service,
             "allocator_calls": allocator_calls, "zeros": zeros,
             "job_dir": capture / "pipeline" / "robot_eval_jobs" / canonical["job_id"]}
    return world


def _staged_run_reported(world: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dispatcher stages; the runner runs every shard; the next poll reports the terminal."""
    staged = executor.poll_once(client=world["client"], capture_root=world["capture"],
                                team_launch_hooks=world["hooks"])
    launch.run_queue(world["queue"], hooks=world["runner"])
    reported = executor.poll_once(client=world["client"], capture_root=world["capture"],
                                  team_launch_hooks=world["hooks"])
    return staged, reported


def _every_file(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def test_a_request_runs_200_hidden_cells_first_shard_alone_and_reports_one_scored_result(
        tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    staged, summary = _staged_run_reported(world)
    assert staged["claimed"] == 1 and staged["staged"] == 1 and staged["pending"] == 1, staged
    assert summary["completed"] == 1, summary
    assert _Broker.accessed == ["canonical-job-1"]
    # The admission seals the very lease the WebApp released once.
    admission = json.loads((world["job_dir"] / "team_candidate_admission.json").read_text())
    assert _Broker.leases == [admission["credential"]["lease_id"]]
    # Eight GPU sessions of 25 cells, each its own allocation inside its share of the cap; the
    # first ran alone and its gate opened the rest.
    calls = world["allocator_calls"]
    assert len(calls) == 8 and len(world["provider"].calls) == 8 and "/shards/00/" in " ".join(calls[0])
    gate = json.loads((world["job_dir"] / launch.FIRST_SHARD_GATE_FILENAME).read_text())
    assert gate["status"] == "passed" and gate["episodes"] == 25
    caps = [float(argv[argv.index("--adp-max-spend-usd") + 1]) for argv in calls]
    assert sum(caps) <= 20.0 and set(caps) == {2.5}
    # The terminal the executor reported, and the scoring statement it sent.
    (path, result), (_settlement_path, settlement) = world["client"].posted
    assert path == "/api/internal/pipeline/agent-run-results" and settlement["rate_usd"] == 0
    outcomes = result["episode_outcomes"]
    assert result["episodes_run"] == 200 and len({row["cell_id"] for row in outcomes}) == 200
    assert result["episodes_succeeded"] == sum(_succeeds(row["cell_id"]) and not _times_out(row["cell_id"])
                                               for row in outcomes)
    # A cell whose policy service stopped answering is the team's failure, reported, never voided.
    timed_out = [row for row in outcomes if _times_out(row["cell_id"])]
    assert timed_out and all((row["outcome"], row["reason"]) == ("failure", "policy_service_did_not_answer")
                             for row in timed_out)
    # Cells stay hidden: the team learns each cell's family, outcome and reason, never its draw.
    assert all("parameters" not in row for row in outcomes)
    assert {"canonical_anchor", "placement_approach", "illumination", "camera_sensor",
            "pairwise_stress", "held_out_composition"} <= {row["family"] for row in outcomes}
    assert result["evaluation_purpose"] == "scored"
    scoring = validate_scoring_statement(result["scoring"], episodes_run=200)
    assert (scoring["runtime"], scoring["success_source"]) == ("team_evaluation_scored", "simulator_scored")
    assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", str(scoring[key])) for key in (
        "success_contract_digest", "scene_revision_digest", "robot_free_probe_receipt_digest",
        "team_candidate_admission_digest", "session_result_digest"))
    assert scoring["scored_episode_count"] == 200 and scoring["claim_ceiling"] == "diagnostic_policy_execution"
    job = world["job_dir"]
    terminal = json.loads((job / TERMINAL_FILENAME).read_text())
    admission = json.loads((job / "team_candidate_admission.json").read_text())
    assert terminal["status"] == "completed" and terminal["admission_digest"] == admission["admission_digest"]
    assert scoring["team_candidate_admission_digest"] == admission["admission_digest"]
    assert admission["candidate_id"] == CANDIDATE and admission["adp_qualified_evidence_eligible"] is False
    # The canonical allocator admitted exactly this team run, inside the free-beta cap.
    paid = json.loads(next(job.glob("shards/00/stage-00/paid_admission.json")).read_text())
    assert paid["status"] == "admitted" and paid["program_id"] == "blueprint-free-beta-evaluated-pilot"
    assert paid["allocation_binding"]["candidate_ids"] == [CANDIDATE]
    # The token never touched disk outside the deleted private file, nor any receipt, log or post.
    assert not (job / ".team-credential").exists()
    assert not [path for path in _every_file(job) if TOKEN.encode() in path.read_bytes()]
    assert TOKEN not in json.dumps(world["client"].posted)
    assert ("provider_get", "GET") in world["service"].calls
    # A second pass neither re-launches nor re-reports.
    launch.run_queue(world["queue"], hooks=world["runner"])
    again = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert again["completed"] == 0 and len(world["allocator_calls"]) == 8 and len(world["client"].posted) == 2


def test_a_policy_failing_at_the_interface_stops_after_the_first_shard(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    world["provider"].service_down = True  # every episode: the team's service never answered
    _staged, summary = _staged_run_reported(world)
    # One GPU session spent, not eight; the team learns why.
    assert len(world["allocator_calls"]) == 1 and summary["blocked"] == 1, summary
    [reason] = world["client"].blocked
    assert reason == "team_policy_interface_failures:25_episodes:policy_service_did_not_answer=25"
    [(_path, settlement)] = world["client"].posted
    assert settlement["episodes_run"] == 25 and settlement["blocked_before_any_episode"] is False
    assert settlement["execution_progress"] == {"observed_attempted_episode_count": 25,
        "scored_episode_count": 25, "succeeded_episode_count": 0, "failed_episode_count": 25,
        "void_episode_count": 0, "measurement_complete": True}


def test_an_unadmitted_funding_approver_never_reaches_the_credential_or_the_allocator(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch, approver="someone-else")
    summary = executor.poll_once(client=world["client"], capture_root=world["capture"],
                                 team_launch_hooks=world["hooks"])
    assert summary["blocked"] == 1 and summary["completed"] == 0
    assert launch.run_queue(world["queue"], hooks=world["runner"]) == []
    assert world["client"].blocked == ["team_candidate_founder_funding_approval_not_admitted"]
    assert _Broker.accessed == [] and world["allocator_calls"] == [] and world["provider"].calls == []
    terminal = json.loads((world["job_dir"] / TERMINAL_FILENAME).read_text())
    assert terminal["status"] == "blocked" and terminal["provider_zero_not_applicable"] is True


def _probe_retention(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *, passes: bool) -> list[Any]:
    """Closeout's retention of the trace the first cell recorded, with the scorer's verdict stubbed.

    The trace's real scoring is pinned in tests/test_team_robot_free_probe.py; here the scene is
    decided whichever way the test needs, written where the gate reads it.
    """

    from blueprint_pipeline import team_robot_free_probe

    receipt_path = world["operator"] / "probe" / probe_gate.RECEIPT_FILENAME
    receipt_path.unlink()
    retained: list[Any] = []

    def retain(*, profile: dict[str, Any], facts: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
        receipt = dict(world["probe_receipt"])
        if not passes:
            receipt.update(status="failed", learned_policy_evaluation_admitted=False,
                           checks={**receipt["checks"], "direct_placement_success": False})
        receipt = _sealed({key: value for key, value in receipt.items() if key != "receipt_digest"},
                          "receipt_digest")
        receipt_path.write_text(json.dumps(receipt))
        retained.append(trace)
        return {"receipt_status": receipt["status"], "receipt_digest": receipt["receipt_digest"],
                "probe_error": None, "write": "written", "offer_status": "published"}

    monkeypatch.setattr(team_robot_free_probe, "retain_recorded_probe", retain)
    # This fixture scene's spec is not one the rigid probe can plan; stand in the control plane's plan.
    monkeypatch.setattr(team_robot_free_probe, "probe_request", lambda **kwargs: _sealed(
        {"schema_version": team_robot_free_probe.REQUEST_SCHEMA, "plan": {"stand_in": True},
         "binding": {"configured_scene_revision_digest": kwargs["scene_revision_digest"]},
         "task_spec": kwargs["task_spec"], "request_digest": ""}, "request_digest"))
    return retained


def test_a_rigid_scene_with_a_missing_probe_is_refused_before_paid_allocation(
        tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    retained = _probe_retention(world, monkeypatch, passes=True)
    staged, summary = _staged_run_reported(world)
    assert staged["blocked"] + summary["blocked"] == 1 and world["allocator_calls"] == [] and retained == []
    assert world["client"].blocked == ["agent_execution_team_candidate_not_offered"]
    from blueprint_pipeline.team_candidate_quick10_registry import (
        offer_eligibility,
        profile_for_capture,
    )

    assert offer_eligibility(profile_for_capture(world["capture"]))[1] == [
        "team_evaluation_robot_free_probe_receipt_missing"]


def test_a_rigid_scene_with_an_unavailable_probe_is_refused_before_paid_allocation(
        tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    receipt_path = world["operator"] / "probe" / probe_gate.RECEIPT_FILENAME
    unavailable = {**world["probe_receipt"], "status": "probe_unavailable",
                   "learned_policy_evaluation_admitted": False}
    receipt_path.write_text(json.dumps(_sealed(unavailable, "receipt_digest")))
    staged, summary = _staged_run_reported(world)
    assert world["allocator_calls"] == [] and staged["blocked"] + summary["blocked"] == 1, summary
    assert world["client"].blocked == ["agent_execution_team_candidate_not_offered"]
    # Every next team is also refused before spending.
    from blueprint_pipeline.team_candidate_quick10_registry import (
        offer_eligibility,
        profile_for_capture,
    )

    assert offer_eligibility(profile_for_capture(world["capture"]))[1] == [
        "team_evaluation_robot_free_probe_not_passed"]


def test_a_token_in_the_sealed_bundle_stops_the_launch_before_the_allocator(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    real_metadata = _metadata()
    # A team that declares its token as its policy id would put it in the sealed spec.
    world["hooks"]["metadata_transport"] = lambda **_kwargs: json.dumps({**real_metadata, "policy_id": TOKEN}).encode()
    executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    launch.run_queue(world["queue"], hooks=world["runner"])
    assert world["allocator_calls"] == [] and world["provider"].calls == []
    assert not (world["job_dir"] / ".team-credential").exists()
    failure = json.loads((world["job_dir"] / "team_candidate_quick10_failure.json").read_text())
    assert failure["blockers"] == ["team_candidate_secret_in_bundle"]
    assert world["client"].blocked == ["team_candidate_secret_in_bundle"]


def test_rights_and_provider_zero_are_checked_before_releasing_the_single_use_lease(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    world["provider_zero"] = lambda: {**ZERO, "provider_zero_verified": False}
    executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert _Broker.accessed == [] and world["allocator_calls"] == []
    assert world["client"].blocked == ["team_candidate_initial_provider_zero_unproven"]


def test_a_metadata_repair_reuses_one_private_delivery_without_spending_another_attempt(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    calls = []
    def metadata(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("synthetic transport failure")
        return world["service"].transport(**kwargs)
    world["hooks"]["metadata_transport"] = metadata
    first = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert first["pending"] == 1 and world["allocator_calls"] == [] and len(_Broker.leases) == 1
    assert not (world["job_dir"] / TERMINAL_FILENAME).exists()
    delivery = world["job_dir"] / ".team-credential" / "delivery.json"
    assert delivery.stat().st_mode & 0o777 == 0o600
    executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    launch.run_queue(world["queue"], hooks=world["runner"])
    final = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert final["completed"] == 1 and len(_Broker.leases) == 1 and len(world["allocator_calls"]) == 8
    assert not delivery.exists()


def test_prelaunch_retries_are_bounded_and_cancelled_deliveries_are_removed(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    def unavailable(**_kwargs):
        raise OSError("synthetic metadata unavailable")
    world["hooks"]["metadata_transport"] = unavailable
    for _ in range(launch.MAX_PRELAUNCH_ATTEMPTS):
        executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert len(_Broker.leases) == 1 and world["allocator_calls"] == []
    assert len(list(world["job_dir"].glob("preflight_attempt_*.json"))) == launch.MAX_PRELAUNCH_ATTEMPTS
    assert not (world["job_dir"] / ".team-credential").exists()
    assert (world["job_dir"] / TERMINAL_FILENAME).is_file()


def test_cancelling_during_prelaunch_repair_removes_the_private_delivery(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    def unavailable(**_kwargs):
        raise OSError("synthetic metadata unavailable")
    world["hooks"]["metadata_transport"] = unavailable
    executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert (world["job_dir"] / ".team-credential").is_dir()
    request = world["row"]["execution_admission"]["canonical_execution_request"]
    launch.cancel_unlaunched_team_candidate_request(request=request, job_dir=world["job_dir"])
    assert not (world["job_dir"] / ".team-credential").exists()
    assert json.loads((world["job_dir"] / TERMINAL_FILENAME).read_text())["execution_progress"]["measurement_complete"] is True


def test_cancellation_before_the_first_allocation_closes_without_spend(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    world["runner"]["execution_monitor"] = lambda _control: {"cancellation_requested": True}
    _, summary = _staged_run_reported(world)
    assert summary["blocked"] == 1 and world["allocator_calls"] == []
    assert world["client"].blocked == ["team_evaluation_cancellation_requested"]
    assert not (world["job_dir"] / ".team-credential").exists()


def test_founder_hold_stops_the_running_allocator_and_waits_for_teardown(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    hold = tmp_path / "founder-hold.json"
    world["runner"]["dispatch_hold_path"] = hold
    running = set()
    stopped = []
    def starter(argv, *, log_path):
        running.add("00")
        hold.write_text('{}')
        return 12345678
    original_state = launch._shard_state
    def state(shard):
        if shard["index"] == 0 and "00" in running:
            return "running"
        return original_state(shard)
    def stopper(shard):
        stopped.append(shard["index"])
        running.clear()
    monkeypatch.setattr(launch, "_shard_state", state)
    world["runner"].update(allocator_starter=starter, allocator_stopper=stopper)
    _, summary = _staged_run_reported(world)
    assert summary["blocked"] == 1 and stopped == [0]
    assert world["client"].blocked == ["team_evaluation_founder_dispatch_hold"]
    assert not (world["job_dir"] / ".team-credential").exists()


def test_transient_monitor_outage_defers_allocation_then_recovers(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    healthy = world["runner"]["execution_monitor"]
    calls = []
    def monitor(control):
        calls.append(len(world["allocator_calls"]))
        if len(calls) == 1:
            raise OSError("synthetic 503")
        return healthy(control)
    world["runner"]["execution_monitor"] = monitor
    _, summary = _staged_run_reported(world)
    assert summary["completed"] == 1 and calls[:2] == [0, 0]


def test_allocator_stop_refuses_a_reused_process_identity(tmp_path, monkeypatch) -> None:
    path = tmp_path / "allocator_process.json"
    path.write_text(json.dumps({"pid": 1234, "start_ticks": "original"}))
    monkeypatch.setattr(launch, "_process_start_ticks", lambda _pid: "different")
    with pytest.raises(launch.TeamCandidateLaunchError, match="allocator_stop_identity_unproven"):
        launch._stop_allocator({"root": str(tmp_path)})


def test_an_unobservable_control_plane_stops_before_the_stale_watchdog(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    world["runner"]["monotonic"] = iter([0, launch.MONITOR_UNAVAILABLE_SECONDS]).__next__
    def unavailable(_control):
        raise OSError("synthetic outage")
    world["runner"]["execution_monitor"] = unavailable
    _, summary = _staged_run_reported(world)
    assert summary["blocked"] == 1 and world["allocator_calls"] == []
    assert world["client"].blocked == ["team_evaluation_control_unavailable_timeout"]


def test_a_spawn_without_a_pid_receipt_cannot_allocate_a_second_time(tmp_path) -> None:
    (tmp_path / "allocator_launch_intent.json").write_text('{}')
    assert launch._shard_state({"root": str(tmp_path)}) == "unknown"


def _unreconciled_spawn(tmp_path):
    request = {"job_id": "synthetic-crash-window"}
    root = tmp_path / "shards" / "00" / "stage-00"
    root.mkdir(parents=True)
    shard = {"index": 0, "root": str(root), "cell_count": 1}
    (root / launch.SHARD_FILENAME).write_text(json.dumps(shard))
    (root / "allocator_launch_intent.json").write_text('{}')
    for name, value in ((launch.REQUEST_FILENAME, request), (launch.INTENT_FILENAME, {"shard_count": 1}),
                        (launch.CONTROL_FILENAME, {"run_id": "synthetic-run"})):
        (tmp_path / name).write_text(json.dumps(value))
    private = tmp_path / ".team-credential"
    private.mkdir(mode=0o700)
    (private / "bearer").write_text("synthetic-private-bearer")
    (private / "bearer").chmod(0o600)
    return request, shard


def test_unknown_spawn_cannot_close_on_a_momentary_provider_zero(tmp_path) -> None:
    from blueprint_pipeline.team_candidate_quick10_closeout import (
        allocator_running,
        seal_blocked_terminal,
    )

    request, _shard = _unreconciled_spawn(tmp_path)
    assert allocator_running(tmp_path) is True
    seal_blocked_terminal(request=request, job_dir=tmp_path, error=ValueError("synthetic-parent-crash"),
        provider_zero=lambda: pytest.fail("zero snapshot cannot resolve a child that may allocate later"))
    assert not (tmp_path / TERMINAL_FILENAME).exists()
    assert (tmp_path / ".team-credential" / "bearer").is_file()


def test_cancelled_unknown_spawn_retains_secret_and_stays_pending_without_a_second_allocation(tmp_path) -> None:
    _unreconciled_spawn(tmp_path)
    (tmp_path / "stop_requested.json").write_text(json.dumps({"reason": "founder_stop"}))
    terminal = launch.run_staged_evaluation(job_dir=tmp_path, hooks={
        "monotonic": iter([0, 0, launch.STOP_TEARDOWN_GRACE_SECONDS]).__next__,
        "allocator_starter": lambda *_args, **_kwargs: pytest.fail("unreconciled spawn cannot allocate again"),
        "allocator_stopper": lambda *_args: pytest.fail("unknown process cannot be signalled"),
        "provider_zero": lambda: pytest.fail("unreconciled spawn cannot settle on global zero")})
    assert terminal is None and not (tmp_path / TERMINAL_FILENAME).exists()
    assert (tmp_path / ".team-credential" / "bearer").is_file()
    assert (tmp_path / "stop_teardown_pending.json").is_file()


def test_reused_pid_is_lost_without_waiting_for_or_signalling_the_unrelated_process(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.team_candidate_quick10_closeout import (
        allocator_running,
        seal_blocked_terminal,
    )

    request, shard = _unreconciled_spawn(tmp_path)
    (Path(shard["root"]) / "allocator_process.json").write_text(json.dumps({"pid": 1234, "start_ticks": "old"}))
    monkeypatch.setattr(launch, "_alive", lambda _pid: True)
    monkeypatch.setattr(launch, "_process_start_ticks", lambda _pid: "new")
    assert launch._shard_state(shard) == "lost"
    assert allocator_running(tmp_path) is False
    zeros = []
    seal_blocked_terminal(request=request, job_dir=tmp_path, error=ValueError("synthetic-original-process-exited"),
        provider_zero=lambda: zeros.append(True) or dict(ZERO))
    assert zeros == [True] and (tmp_path / TERMINAL_FILENAME).is_file()
    with pytest.raises(launch.TeamCandidateLaunchError, match="stop_identity_unproven"):
        launch._stop_allocator(shard)


def test_live_pid_without_start_identity_remains_unknown(tmp_path, monkeypatch) -> None:
    _request, shard = _unreconciled_spawn(tmp_path)
    (Path(shard["root"]) / "allocator_process.json").write_text(json.dumps({"pid": 1234, "start_ticks": None}))
    monkeypatch.setattr(launch, "_alive", lambda _pid: True)
    monkeypatch.setattr(launch, "_process_start_ticks", lambda _pid: "current")
    assert launch._shard_state(shard) == "unknown"


def test_unknown_spawn_pauses_other_shards_until_result_or_identity_reconciliation(tmp_path) -> None:
    from itertools import count

    _request, first = _unreconciled_spawn(tmp_path)
    first_root = Path(first["root"])
    (first_root / "allocator_result.json").write_text(json.dumps({"status": "completed"}))
    for index in (1, 2):
        root = tmp_path / "shards" / f"{index:02d}" / "stage-00"
        root.mkdir(parents=True)
        (root / launch.SHARD_FILENAME).write_text(json.dumps({"index": index, "root": str(root), "cell_count": 1}))
        if index == 1:
            (root / "allocator_launch_intent.json").write_text('{}')
    (tmp_path / launch.INTENT_FILENAME).write_text(json.dumps({"shard_count": 3}))
    monitored = []
    def monitor(control):
        monitored.append(True)
        return ({"ok": True, "run_id": control["run_id"], "execution_deadline_at_ms": 4_000_000_000_000}
                if len(monitored) == 1 else {"cancellation_requested": True})
    terminal = launch.run_staged_evaluation(job_dir=tmp_path, hooks={
        "execution_monitor": monitor, "monotonic": count(0, launch.STOP_TEARDOWN_GRACE_SECONDS).__next__,
        "sleep": lambda _seconds: None,
        "allocator_starter": lambda *_args, **_kwargs: pytest.fail("unknown shard must pause other allocations"),
        "provider_zero": lambda: pytest.fail("unknown shard cannot settle on global zero")})
    assert terminal is None and len(monitored) == 2
    assert (tmp_path / "allocator_reconciliation_pending.json").is_file()
    assert (tmp_path / ".team-credential" / "bearer").is_file()


def test_partial_allocator_result_does_not_resolve_an_unknown_spawn(tmp_path) -> None:
    from blueprint_pipeline.team_candidate_quick10_closeout import allocator_running

    _request, shard = _unreconciled_spawn(tmp_path)
    (Path(shard["root"]) / "allocator_result.json").write_text('{"status":')
    assert launch._shard_state(shard) == "unknown" and allocator_running(tmp_path) is True


def test_missing_planned_shard_artifacts_cannot_prove_zero_episodes(tmp_path) -> None:
    from blueprint_pipeline.team_candidate_quick10_closeout import blocked_execution_progress

    (tmp_path / launch.INTENT_FILENAME).write_text(json.dumps({"shard_count": 1}))
    progress = blocked_execution_progress(tmp_path)
    assert progress["observed_attempted_episode_count"] == 0
    assert progress["measurement_complete"] is False


def test_slow_stop_teardown_stays_pending_after_the_local_grace(tmp_path, monkeypatch) -> None:
    root = tmp_path / "shards" / "00" / "stage-00"
    root.mkdir(parents=True)
    (root / launch.SHARD_FILENAME).write_text(json.dumps({"index": 0, "root": str(root)}))
    for name, value in ((launch.REQUEST_FILENAME, {"job_id": "synthetic-job"}),
                        (launch.INTENT_FILENAME, {"shard_count": 1}), (launch.CONTROL_FILENAME, {}),
                        ("stop_requested.json", {"reason": "founder_stop"})):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(launch, "_shard_state", lambda _shard: "running")
    stopped = []
    terminal = launch.run_staged_evaluation(job_dir=tmp_path, hooks={
        "allocator_stopper": lambda shard: stopped.append(shard["index"]),
        "monotonic": iter([0, 0, launch.STOP_TEARDOWN_GRACE_SECONDS]).__next__,
        "sleep": lambda _seconds: None,
        "provider_zero": lambda: pytest.fail("running allocator cannot prove provider zero")})
    assert terminal is None and stopped == [0]
    assert (tmp_path / "stop_teardown_pending.json").is_file()
    assert not (tmp_path / TERMINAL_FILENAME).exists()


def test_closeout_waits_for_provider_zero_before_reporting(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    live = {**ZERO, "provider_zero_verified": False, "live_instance_count": 1, "blockers": ["live"]}
    observations = iter([dict(ZERO), live])  # zero before launch, then a live instance at closeout
    world["provider_zero"] = lambda: next(observations, dict(ZERO))
    _staged, pending = _staged_run_reported(world)
    assert pending["pending"] == 1 and world["client"].posted == [] and len(world["allocator_calls"]) == 8
    assert not (world["job_dir"] / TERMINAL_FILENAME).exists()
    launch.run_queue(world["queue"], hooks=world["runner"])
    second = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert second["completed"] == 1 and len(world["allocator_calls"]) == 8  # recovered, never re-allocated


def test_the_offer_lists_each_team_evaluation_only_while_eligible(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    capture = world["capture"]
    descriptor = {"scene_id": "site-req1", "capture_id": "cap-1", "site_submission_id": "req1",
                  "metadata": {"capture_entry_source": "website_task_capture"}}
    root = tmp_path / "partition" / "scenes" / "site-req1" / "captures" / "cap-1"
    root.mkdir(parents=True)
    (root / "capture_descriptor.json").write_text(json.dumps(descriptor))
    specs = root / "pipeline" / "simulation_automation" / "episode_specs.json"
    specs.parent.mkdir(parents=True)
    specs.write_text(json.dumps({"episode_count": 2, "episodes": [{"scenario_id": "nominal"}] * 2}))
    registry_path = Path(os.environ[REGISTRY_ENV])
    registry = json.loads(registry_path.read_text())
    registry["profiles"][0]["capture_root"] = str(root.resolve())
    registry_path.write_text(json.dumps(_sealed(registry, "registry_digest")))
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr("blueprint_pipeline.agent_execution_offer.is_website_entry_source", lambda _value: True)
    published = publish_agent_execution_offer(root, transport=lambda **kwargs: sent.append(kwargs["payload"]))
    assert published["status"] == "published" and capture.exists()
    offer = sent[-1]["offer"]
    assert offer["policy_execution_profiles"] == ["team_evaluation_scored_v1"]
    assert offer["episode_count"] == 200 and offer["scenario_id"] == "nominal"
    contract = offer["team_evaluation"]["contract"]
    assert contract["scored"]["episodes"] == 200 and contract["scored"]["cells_published"] is False
    assert "practice" not in contract and contract["instruction"]
    assert contract["task_success_contract"]["contract_digest"].startswith("sha256:")
    # A rigid scene whose check is undecided is unlisted until a separate probe passes.
    receipt_path = tmp_path / "operator" / "probe" / probe_gate.RECEIPT_FILENAME
    receipt_path.unlink()
    publish_agent_execution_offer(root, transport=lambda **kwargs: sent.append(kwargs["payload"]))
    assert sent[-1].get("offer") is None or "team_evaluation" not in sent[-1]["offer"]
    # A scene whose check failed is not listed at all, and says why.
    failed = {key: value for key, value in world["probe_receipt"].items() if key != "receipt_digest"}
    failed.update(status="failed", learned_policy_evaluation_admitted=False)
    receipt_path.write_text(json.dumps(_sealed(failed, "receipt_digest")))
    published = publish_agent_execution_offer(root, transport=lambda **kwargs: sent.append(kwargs["payload"]))
    assert sent[-1].get("offer") is None or "team_evaluation" not in sent[-1]["offer"], published


def test_the_terminal_reader_is_bound_to_its_request_and_silent_for_other_jobs(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    job = world["job_dir"]
    job.mkdir(parents=True)
    assert read_team_candidate_terminal(job_dir=job, expected_job_id="canonical-job-1",
                                        expected_canonical_request_digest="sha256:" + "0" * 64) is None
    _staged_run_reported(world)
    digest = canonical_digest(world["row"]["execution_admission"]["canonical_execution_request"])
    sealed = read_team_candidate_terminal(job_dir=job, expected_job_id="canonical-job-1",
                                          expected_canonical_request_digest=digest)
    assert sealed["status"] == "completed" and sealed["episodes_run"] == 200
    with pytest.raises(ValueError, match="team_candidate_terminal_binding_mismatch"):
        read_team_candidate_terminal(job_dir=job, expected_job_id="another-job",
                                     expected_canonical_request_digest=digest)
    tampered = {**sealed, "episodes_succeeded": 10}
    (job / TERMINAL_FILENAME).write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="team_candidate_terminal_binding_mismatch"):
        read_team_candidate_terminal(job_dir=job, expected_job_id="canonical-job-1",
                                     expected_canonical_request_digest=digest)


def test_a_team_brings_its_own_arm_and_cameras_to_its_evaluation(tmp_path, monkeypatch) -> None:
    from tests.test_team_embodiment import manifest, write_robot_usd

    model = write_robot_usd(tmp_path / "acme_arm.usda")
    world = _world(tmp_path, monkeypatch, robot={"manifest": manifest(model), "model": model})
    staged, summary = _staged_run_reported(world)
    assert staged["staged"] == 1 and summary["completed"] == 1, (staged, summary)
    job = world["job_dir"]
    admission = json.loads((job / "team_candidate_admission.json").read_text())
    assert (admission["policy_interface_id"], admission["embodiment_preset_id"]) == (
        "blueprint_policy_https.v1", "team_embodiment")
    spec = json.loads(next(job.glob("shards/00/stage-00/runtime_inputs/policy_execution_spec.*.json")).read_text())
    binding = spec["team_embodiment"]
    assert binding["manifest"]["embodiment_digest"] == admission["embodiment"]["embodiment_digest"]
    # The verified robot ships in the sealed bundle at the path the binding names; the worker's
    # static preflight (run by the stand-in GPU host) resolved every cell onto it.
    bundle = next(job.glob("shards/00/stage-00/bundle/team_candidate_quick10_session_bundle.zip"))
    with zipfile.ZipFile(bundle) as archive:
        shipped = archive.read("provider_runtime/native_task_packet/team_embodiment/robot.usd")
    assert "sha256:" + hashlib.sha256(shipped).hexdigest() == binding["model"]["sha256"]
    (_path, result), _settlement = world["client"].posted
    assert result["episodes_run"] == 200 and result["scoring"]["runtime"] == "team_evaluation_scored"


def test_a_team_arm_that_cannot_reach_the_task_is_refused_before_any_allocation(tmp_path, monkeypatch) -> None:
    from tests.test_team_embodiment import manifest, write_robot_usd

    model = write_robot_usd(tmp_path / "acme_arm.usda")
    world = _world(tmp_path, monkeypatch, robot={"manifest": manifest(model, reach_m=0.1), "model": model})
    summary = executor.poll_once(client=world["client"], capture_root=world["capture"],
                                 team_launch_hooks=world["hooks"])
    assert summary["blocked"] == 1 and world["allocator_calls"] == []
    assert world["client"].blocked == ["team_embodiment_station_out_of_reach"]
    assert _Broker.accessed == []


def test_a_scene_written_in_by_its_own_records_runs_without_a_hand_written_profile(
        tmp_path, monkeypatch) -> None:
    from blueprint_pipeline.team_candidate_quick10_registry import (
        load_registry,
        profile_for_capture,
        write_derived_profile,
    )

    world = _world(tmp_path, monkeypatch, derived=True)
    profile = profile_for_capture(world["capture"])
    # The scene's facts came from its row; who may fund it, at what rate, and rights from the operator.
    assert profile["allowed_founder_approvers"] == ["founder@blueprint"] and len(profile["scored_matrix_salt"]) == 64
    assert load_registry()["profiles"] == []
    staged, summary = _staged_run_reported(world)
    assert staged["staged"] == 1 and summary["completed"] == 1, (staged, summary)
    # A scene's row can never carry an operator decision.
    refused = write_derived_profile({"capture_root": str(world["capture"]), "allowed_founder_approvers": ["someone"]})
    assert refused["status"] == "derived_profile_refused" and "allowed_founder_approvers" in refused["unexpected"]


def test_a_prepared_scene_hosts_a_team_run_from_its_own_packet_with_no_prior_run(tmp_path, monkeypatch) -> None:
    """No robot placement, no frozen pair: the scene's own compiled packet is the whole scene half."""

    from blueprint_pipeline.team_candidate_quick10_registry import (
        profile_for_capture,
        write_derived_profile,
    )

    world = _world(tmp_path, monkeypatch, derived=True, scene_half="team_scene")
    profile = profile_for_capture(world["capture"])
    assert "pair_runtime_inputs_path" not in profile and profile["team_scene_inputs_path"]
    staged, summary = _staged_run_reported(world)
    assert staged["staged"] == 1 and summary["completed"] == 1, (staged, summary)
    # Every session ran on the packet itself as its construction lineage, under the team's own run kind.
    inputs = json.loads(next(world["job_dir"].glob("shards/00/stage-00/runtime_inputs/"
                                                   "policy_canary_runtime_inputs.json")).read_text())
    assert inputs["run_kind"] == "team_candidate_evaluation"
    assert inputs["construction_result"] == inputs["base_native_packet"]
    assert {cell["control_diagnostic"]["mode"] for cell in inputs["cells"]} == {"nonblocking_diagnostic_pending"}
    preflight = json.loads(next(world["job_dir"].glob(
        "shards/00/stage-00/bundle/policy_canary_static_startup_preflight.v1.json")).read_text())
    assert preflight["status"] == "passed"
    # A row names exactly one scene half.
    both = write_derived_profile({**{key: profile[key] for key in (
        "capture_root", "scene_id", "task_id", "scenario_id", "robot_free_probe_root", "scene_revision_digest",
        "team_scene_inputs_path")}, "pair_runtime_inputs_path": "/x", "pair_activation_path": "/y"})
    assert both["status"] == "derived_profile_refused" and both["missing"] == ["team_scene_inputs_path"]
