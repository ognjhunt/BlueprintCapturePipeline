"""Hermetic Quick-10 rehearsal with ONE team candidate in place of the frozen pair.

Beta journey step: L5 team policy evaluation. This drives the production
orchestration end to end, the same way the paired rehearsal does: the real
isolated-cell worker (:func:`_run_isolated_cell_processes`), the real session
contract, the real episode runner and scorer, and the real team HTTPS client
built by the production client factory over the production
``fetch_bounded_https`` call signature. Only Isaac (the rehearsal fake that
keeps its close/rebuild semantics) and the team's web service are faked. No
GPU, provider, network or spend.

The run binds a team admission record it does not re-derive; admission (rights,
lease, budget, founder approval, robot-free probe, interface) is pinned in
``tests/test_team_candidate_quick10_admission.py``. The rehearsal scene is a
rigid task, for which no robot-free probe exists yet, so a real admission would
refuse it; the runtime mechanics are the same for every task kind.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import blueprint_pipeline.droid_https_policy_client as https_client
import blueprint_pipeline.native_task_arena_policy_canary_worker as worker
from blueprint_pipeline.agent_run_executor import AgentRunWebAppClient
from blueprint_pipeline.agent_run_result_scoring import validate_scoring_statement
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PROVIDER_RESULT_FILENAME,
    TEAM_ADMISSION_SCHEMA_VERSION,
    PolicyCanarySessionError,
    build_session_authority,
    validate_runtime_input_manifest,
    validate_session_result,
)
from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
from blueprint_pipeline.team_candidate_quick10_admission import RUNTIME, execution_spec, runtime_binding
from blueprint_pipeline.team_candidate_quick10_result import TeamCandidateResultError, team_candidate_terminal
from tests import test_native_task_arena_policy_canary_lifecycle_rehearsal as paired

TEAM = "team-acme.pi05-ft-7"
ORIGIN = "https://policy.acme-robotics.example"
TOKEN_ENV = "BLUEPRINT_TEAM_POLICY_BEARER_FILE"
MODEL_DIGEST = "sha256:" + "7" * 64
TEAM_POLICY_ID = "acme-pi05-ft-7"


def _admission(contract: dict[str, Any]) -> dict[str, Any]:
    """A sealed admission record as admit_team_candidate emits it (fields only, see module doc)."""

    value = {
        "schema_version": TEAM_ADMISSION_SCHEMA_VERSION, "status": "admitted",
        "run_kind": "team_candidate_evaluation", "runtime": RUNTIME, "evaluation_purpose": "practice",
        "episodes": 10,
        "team_id": "team-acme", "checkpoint_id": "pi05-ft-7", "candidate_id": TEAM,
        "policy_interface_id": "openpi_droid_https.v1",
        "embodiment_preset_id": "droid_franka_panda_robotiq_2f85_v1",
        "delivery": {"mode": "https_policy", "endpoint_url": ORIGIN + "/v1/act",
                     "metadata_url": ORIGIN + "/v1/metadata", "action_chunk_rows": 10, "open_loop_horizon": 8,
                     "model_identity_digest": MODEL_DIGEST, "team_policy_id": TEAM_POLICY_ID},
        "credential": {"lease_id": "lease-1", "credential_ref_digest": "sha256:" + "6" * 64,
                       "credential_env": TOKEN_ENV, "origin": ORIGIN},
        "scene_revision_digest": "sha256:" + "9" * 64, "task_spec_digest": "sha256:" + "5" * 64,
        "task_success_contract_digest": contract["contract_digest"],
        "robot_free_probe_receipt_digest": "sha256:" + "4" * 64,
        "rights_digest": "sha256:" + "3" * 64, "founder_approval_digest": "sha256:" + "2" * 64,
        "budget": {"cap_usd": 20, "hard_ttl_seconds": 9000, "max_attempts": 1, "episodes": 10},
        "claim_ceiling": "diagnostic_policy_execution", "evidence_scope": "development_only",
        "adp_qualified_evidence_eligible": False, "physical_performance_claimed": False,
        "admission_digest": "",
    }
    value["admission_digest"] = canonical_digest(value, digest_field="admission_digest")
    return value


def _token_file(tmp_path: Path) -> str:
    """The private file the provider bootstrap writes; the runtime reads it, never argv or bundle bytes."""
    secrets = tmp_path / "runtime-secrets"
    secrets.mkdir(mode=0o700)
    token = secrets / "team-policy-bearer"
    token.write_text("team-secret-token\n", encoding="utf-8")
    token.chmod(0o600)
    return str(token)


def _stage_team_runtime(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Lay out a team-candidate provider bundle exactly as the worker reads it."""

    runtime, output = paired._stage_runtime_root(tmp_path, scene_plan=paired._destination_scene_plan())
    inputs = json.loads((runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json").read_text())
    activation_path = runtime / "runtime_inputs" / "task_evaluation_policy_campaign_activation.v1.json"
    activation = json.loads(activation_path.read_text())
    contract = inputs["task_success_contract"]
    admission = _admission(contract)
    for name in ("policy_execution_spec.pi05_droid.json", "policy_execution_spec.groot_n17_droid.json"):
        (runtime / "runtime_inputs" / name).unlink()
    activation.update(run_kind="team_candidate_evaluation", candidate_ids=[TEAM],
                      campaign_units=[{**unit, "candidate_ids": [TEAM]} for unit in activation["campaign_units"]])
    activation["activation_digest"] = canonical_digest(activation, digest_field="activation_digest")
    paired._write(activation_path, activation)
    inputs.update(run_kind="team_candidate_evaluation", candidate_ids=[TEAM],
                  activation_digest=activation["activation_digest"],
                  team_candidate_admission={**runtime_binding(admission),
                                            "scene_revision_digest": inputs["scene_revision_digest"]})
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    inputs_path = paired._write(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json", inputs)
    authority = build_session_authority(
        activation_manifest=activation, activation_record=paired._record(activation_path),
        runtime_inputs=inputs, runtime_input_record=paired._record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-0123456789abcdef", hard_cap_usd=20.0,
        hard_ttl_seconds=9_000)
    paired._write(runtime / "runtime_inputs" / "policy_canary_session_authority.json", authority)
    spec = execution_spec(admission, task_success_contract=contract, prompt="place the book in the tray",
                          max_policy_queries=1)
    paired._write(runtime / "runtime_inputs" / f"policy_execution_spec.{TEAM}.json", spec)
    manifest = json.loads((runtime / "adp_arena_provider_manifest.json").read_text())
    manifest.update(execution_mode="team_candidate_evaluation_session", run_kind="team_candidate_evaluation",
                    runtime_inputs_digest=inputs["runtime_inputs_digest"],
                    authority_digest=authority["authority_digest"],
                    execution_spec_digests={TEAM: spec["execution_spec_digest"]})
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    paired._write(runtime / "adp_arena_provider_manifest.json", manifest)
    return runtime, output, inputs, admission


class _TeamService:
    """The team's own HTTPS policy service, as the bounded transport sees it."""

    def __init__(self, *, malformed_post: int | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.malformed_post = malformed_post

    def __call__(self, url: str, **kwargs: Any) -> Any:
        assert kwargs["allowed_origins"] == (ORIGIN,) and kwargs["max_redirects"] == 0
        assert kwargs["allowed_content_types"] == ("application/json",)
        assert kwargs["headers"]["Authorization"] == "Bearer team-secret-token"
        self.requests.append({"url": url, "method": kwargs["method"],
                              "body": json.loads(kwargs["data"]) if kwargs["data"] else None})
        if url == ORIGIN + "/v1/metadata":
            body = {"schema_version": https_client.METADATA_SCHEMA, "policy_id": TEAM_POLICY_ID,
                    "policy_interface_id": "openpi_droid_https.v1", "action_space": "joint_position",
                    "action_chunk_rows": 10, "model_identity_digest": MODEL_DIGEST}
        else:
            assert url == ORIGIN + "/v1/act" and kwargs["method"] == "POST"
            actions = np.zeros((10, 8))
            actions[:, 0] = 0.25
            actions[:, 7] = 0.9
            body = {"actions": actions.tolist(), "policy_timing": {"infer_ms": 9.0}}
            if sum(row["method"] == "POST" for row in self.requests) == self.malformed_post:
                # A team may claim anything; the envelope admits actions and timing only.
                body["task_success"] = True
        return type("Response", (), {"body": json.dumps(body).encode(), "status": 200})()


def _team_runtime(isaac: Any) -> worker.CellRuntime:
    runtime = paired._rehearsal_runtime(isaac)
    # The production factory, not a test client: the team spec resolves to the HTTPS client.
    return worker.CellRuntime(**{**runtime.__dict__, "policy_client": canary_policy_client})


def _close_provider(result: dict[str, Any]) -> dict[str, Any]:
    """Stand in for the control-plane closeout that confirms teardown and provider zero."""

    closed = {**result, "status": "completed_unqualified", "provider_allocations_observed": 1,
              "session_closeout": {**result["session_closeout"], "status": "closed",
                                   "provider_allocations_observed": 1, "teardown_completed": True,
                                   "provider_zero_confirmed": True}}
    closed["result_digest"] = canonical_digest(closed, digest_field="result_digest")
    return closed


def test_quick10_runs_ten_simulator_scored_episodes_of_one_team_candidate(tmp_path, monkeypatch) -> None:
    runtime_root, provider_output, inputs, admission = _stage_team_runtime(tmp_path)
    service = _TeamService()
    monkeypatch.setattr(https_client, "fetch_bounded_https", service)
    monkeypatch.setenv(TOKEN_ENV, _token_file(tmp_path))
    isaacs: list[Any] = []

    exit_code = worker._run_isolated_cell_processes(
        runtime_root=runtime_root, output_root=provider_output,
        run_cell_process=lambda **kwargs: paired._run_cell_in_process(
            **kwargs, isaacs=isaacs, runtime_factory=_team_runtime))

    assert exit_code == 0
    result = paired._sealed_result(provider_output / PROVIDER_RESULT_FILENAME)
    assert result["run_kind"] == "team_candidate_evaluation"
    assert result["candidate_ids"] == [TEAM]
    assert result["paired_diagnostic_summary"] is None
    assert result["isolated_simulation_process_count"] == 10
    assert len(result["episodes"]) == 10 and len(isaacs) == 10
    assert {(row["cell_id"], row["seed"]) for row in result["episodes"]} == {
        (f"cell-{index}", 3100 + index) for index in range(10)}
    for row in result["episodes"]:
        assert row["candidate_id"] == TEAM and row["status"] == "completed"
        assert row["candidate_policy_queried"] is True
        assert row["scoring_authority"] == "deterministic_simulator_state"
        assert row["episode"]["score"]["status"] == "scored"
        assert isinstance(row["episode"]["score"]["task_succeeded"], bool)
        # Identity stays the team's; only the observation contract is pi0.5-DROID's.
        assert row["episode"]["candidate_id"] == TEAM
        assert row["episode"]["observation_conversion"]["candidate_id"] == "pi05_droid"
        assert row["checkpoint_digest"] == MODEL_DIGEST
    # The team saw exactly the pi0.5-DROID observation: two lossless 224x224 views and state.
    posts = [row for row in service.requests if row["method"] == "POST"]
    assert len(posts) == 10
    for row in posts:
        observation = row["body"]["observation"]
        assert set(observation) == {"observation/exterior_image_1_left", "observation/wrist_image_left",
                                    "observation/joint_position", "observation/gripper_position", "prompt"}
        for key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
            image = Image.open(io.BytesIO(base64.b64decode(observation[key]["data"])))
            assert image.format == "PNG" and image.size == (224, 224)
        assert len(observation["observation/joint_position"]) == 7
    # The bearer credential never reaches retained evidence.
    assert "team-secret-token" not in json.dumps(result)
    assert not any("team-secret-token" in path.read_text(errors="ignore")
                   for path in provider_output.rglob("*.json"))

    closed = _close_provider(result)
    assert validate_session_result(closed)["run_kind"] == "team_candidate_evaluation"
    terminal = team_candidate_terminal(session_results=[closed], purpose="practice", runtime_binding=inputs["team_candidate_admission"])
    succeeded = sum(row["episode"]["score"]["task_succeeded"] for row in result["episodes"])
    assert terminal["episodes_run"] == 10 and terminal["episodes_succeeded"] == succeeded
    assert terminal["scoring"] == validate_scoring_statement(terminal["scoring"], episodes_run=10)
    assert terminal["scoring"] == {
        "schema_version": "blueprint.agent_run_result_scoring.v1",
        "runtime": "team_evaluation_practice", "success_source": "simulator_scored",
        "scored_episode_count": 10, "claim_ceiling": "diagnostic_policy_execution",
        "success_contract_digest": inputs["task_success_contract_digest"],
        "scene_revision_digest": inputs["scene_revision_digest"],
        "robot_free_probe_receipt_digest": admission["robot_free_probe_receipt_digest"],
        "team_candidate_admission_digest": admission["admission_digest"],
        "session_result_digest": closed["result_digest"]}

    # And it reaches the WebApp as one signed result carrying that statement verbatim.
    sent: list[tuple[str, dict[str, Any]]] = []

    class Client(AgentRunWebAppClient):
        def __init__(self) -> None:
            pass

        def _json(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None):
            sent.append((path, payload or {}))
            return {"ok": True}

    row = {"reservation_id": "free_team", "team_id": "team-acme", "quoted_episodes": 10, "quoted_usd": 0,
           "execution_admission_digest": "sha256:" + "a" * 64}
    Client().report_completed(row, "agent-attempt-1", terminal)
    path, payload = sent[0]
    assert path == "/api/internal/pipeline/agent-run-results"
    assert payload["episodes_run"] == 10 and payload["scoring"] == terminal["scoring"]


def test_a_refused_team_response_keeps_the_run_out_of_recommendation_evidence(tmp_path, monkeypatch) -> None:
    runtime_root, provider_output, inputs, _admission_record = _stage_team_runtime(tmp_path)
    # On the fourth episode the team's service adds a field of its own: the client
    # refuses the envelope after the query, so no action reaches the robot.
    monkeypatch.setattr(https_client, "fetch_bounded_https", _TeamService(malformed_post=4))
    monkeypatch.setenv(TOKEN_ENV, _token_file(tmp_path))
    isaacs: list[Any] = []
    assert worker._run_isolated_cell_processes(
        runtime_root=runtime_root, output_root=provider_output,
        run_cell_process=lambda **kwargs: paired._run_cell_in_process(
            **kwargs, isaacs=isaacs, runtime_factory=_team_runtime)) == 0
    result = paired._sealed_result(provider_output / PROVIDER_RESULT_FILENAME)
    refused = [row for row in result["episodes"] if row["status"] != "completed"]
    assert len(result["episodes"]) == 10 and len(refused) == 1
    assert refused[0]["candidate_policy_queried"] is True and refused[0]["actions_reached_robot"] is False
    blocked = _close_provider(result)
    blocked["status"] = "blocked"
    blocked["result_digest"] = canonical_digest(blocked, digest_field="result_digest")
    terminal = team_candidate_terminal(session_results=[blocked], purpose="practice", runtime_binding=inputs["team_candidate_admission"])
    # The team's service caused it, so it is the team's failure, not a void episode.
    refused_row = next(row for row in terminal["episode_outcomes"] if row["cell_id"] == refused[0]["cell_id"])
    assert refused_row["outcome"] == "failure" and refused_row["reason"].startswith("policy_")
    assert terminal["episodes_run"] == 10 and terminal["scoring"]["scored_episode_count"] == 10


def test_runtime_refuses_a_frozen_id_a_second_candidate_or_an_unbound_admission(tmp_path) -> None:
    _runtime, _output, inputs, admission = _stage_team_runtime(tmp_path)
    reseal = lambda value: {**value, "runtime_inputs_digest": canonical_digest(  # noqa: E731
        value, digest_field="runtime_inputs_digest")}
    assert validate_runtime_input_manifest(inputs)["candidate_ids"] == [TEAM]
    for change, code in (
        ({"candidate_ids": ["pi05_droid"]}, "policy_canary_team_candidate_invalid"),
        ({"candidate_ids": [TEAM, "other-team"]}, "policy_canary_team_candidate_invalid"),
        ({"team_candidate_admission": None}, "policy_canary_team_admission_binding_invalid"),
        ({"team_candidate_admission": {**inputs["team_candidate_admission"],
                                       "task_success_contract_digest": "sha256:" + "0" * 64}},
         "policy_canary_team_admission_binding_invalid"),
        ({"run_kind": "internal_policy_canary"}, "policy_canary_candidate_pair_invalid"),
    ):
        with pytest.raises(PolicyCanarySessionError, match=code):
            validate_runtime_input_manifest(reseal({**inputs, **change}))
    with pytest.raises(TeamCandidateResultError, match="binding_mismatch"):
        team_candidate_terminal(session_results=[_fake_closed_team_result(inputs)], purpose="practice",
                                runtime_binding={**inputs["team_candidate_admission"], "candidate_id": "x"})
    assert admission["adp_qualified_evidence_eligible"] is False


def _fake_closed_team_result(inputs: dict[str, Any]) -> dict[str, Any]:
    """A minimal valid closed team session result (ten completed, scored episodes)."""

    episodes = []
    for cell in inputs["cells"]:
        row = {key: cell[key] for key in ("cell_id", "seed", "cell_spec_digest", "family",
                                          "resolved_scenario", "resolved_scenario_digest")}
        row.update(candidate_id=TEAM, status="completed", run_kind="team_candidate_evaluation",
                   claim_ceiling="diagnostic_policy_execution", ranking_eligible=False,
                   candidate_policy_queried=True, actions_reached_robot=True, arm_moved=True,
                   scoring_authority="deterministic_simulator_state",
                   episode={"score": {"status": "scored", "task_succeeded": True}},
                   **{field: "sha256:" + "e" * 64 for field in (
                       "checkpoint_digest", "runtime_identity_digest", "lossless_frame_manifest_digest",
                       "review_video_digest", "returned_action_sequence_digest", "action_delivery_readback_digest",
                       "state_trace_digest", "contact_force_digest", "task_object_trajectory_digest",
                       "deterministic_score_digest")})
        episodes.append(row)
    value = {"schema_version": "native_task_arena_policy_canary_session_result.v1",
             "status": "completed_unqualified", "run_kind": "team_candidate_evaluation",
             "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": [TEAM],
             "task_success_contract": inputs["task_success_contract"],
             "task_success_contract_digest": inputs["task_success_contract_digest"],
             "episodes_per_policy": 10, "learned_policy_rollout_count": 10, "provider_allocations_observed": 1,
             "retry_cap": 0, "warm_session_open_count": 1, "episodes": episodes,
             "session_closeout": {"provider_allocations_observed": 1, "teardown_completed": True,
                                  "provider_zero_confirmed": True},
             "scene_promotion_performed": False, "official_ranking_performed": False, "result_digest": ""}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_projection_counts_only_simulator_verdicts(tmp_path) -> None:
    _runtime, _output, inputs, _admission_record = _stage_team_runtime(tmp_path)
    closed = _fake_closed_team_result(inputs)
    binding = inputs["team_candidate_admission"]
    assert team_candidate_terminal(session_results=[closed], purpose="practice", runtime_binding=binding)["episodes_succeeded"] == 10
    # A self-graded row is refused by the session contract itself.
    self_graded = json.loads(json.dumps(closed))
    self_graded["episodes"][0]["scoring_authority"] = "candidate_policy"
    self_graded["result_digest"] = canonical_digest(self_graded, digest_field="result_digest")
    with pytest.raises(TeamCandidateResultError, match="session_result_invalid"):
        team_candidate_terminal(session_results=[self_graded], purpose="practice", runtime_binding=binding)
    # An undetermined score is the simulator's failure: void, neither a success nor counted.
    undetermined = json.loads(json.dumps(closed))
    undetermined["episodes"][1]["episode"] = {"score": {"status": "undetermined", "task_succeeded": None}}
    undetermined["result_digest"] = canonical_digest(undetermined, digest_field="result_digest")
    terminal = team_candidate_terminal(session_results=[undetermined], purpose="practice", runtime_binding=binding)
    assert terminal["scoring"]["scored_episode_count"] == 9 and terminal["episodes_succeeded"] == 9
