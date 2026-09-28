from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_interpretation import (
    DeterministicFixtureInterpreter,
    EpisodeInterpretationError,
    EpisodeInterpreterOutput,
    InterpreterIdentity,
    OpenAIMultimodalEpisodeInterpreter,
    build_episode_interpretation_request,
    interpret_episode,
    materialize_episode_interpretation_rights,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _criteria(*, no_drop: bool) -> dict:
    return {
        "destination_containment": {
            "mode": "required",
            "position_bounds_world_m": {
                "minimum": [1.1, 1.9, 0.79],
                "maximum": [1.2, 2.1, 0.81],
            },
        },
        "orientation": {
            "mode": "ignored",
            "reference_xyzw": [0.0, 0.0, 0.0, 1.0],
            "tolerance_rad": 0.2,
        },
        "support": {
            "height_mode": "required",
            "height_interval_m": [0.79, 0.81],
            "contact_mode": "required",
        },
        "terminal_task_contact": {"mode": "cleared"},
        "gripper_state": {"mode": "ignored", "threshold_m": None},
        "settling": {
            "mode": "required",
            "window_samples": 2,
            "position_tolerance_m": 0.01,
            "orientation_tolerance_rad": 0.1,
        },
        "safety": {"mode": "required"},
        "motion": {
            "movement_epsilon_m": 0.001,
            "minimum_translation_m": 0.1,
            "minimum_lift_m": None,
        },
        "temporal_invariants": {
            "schema_version": "rigid_task_event_ledger_expectation.v1",
            "no_drop": {
                "mode": "required" if no_drop else "ignored",
                "minimum_fall_m": 0.02,
            },
            "maximum_task_contact_force_n": None,
            "forbidden_contact_classes": [],
            "containment_excursions": "forbidden",
            "workspace_excursions": "ignored",
            "maximum_retries": None,
            "maximum_regrasps": None,
        },
    }


def _episode_root(tmp_path: Path, *, no_drop: bool, deterministic_success: bool) -> dict:
    root = tmp_path / "run"
    root.mkdir()
    task_spec = {
        "manipulation_strategy": "planar_push",
        "subject_asset_id": "cup",
    }
    contract = seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id="scene839873",
        task_id="move_cup_to_green_target",
        author_source="site_robot_team",
        author_id="robot-team:task-owners",
        confirmation_status="confirmed",
        confirmed_by_team_id="robot-team:task-owners",
        criteria=_criteria(no_drop=no_drop),
    )
    contract_path = root / "task_success_contract.json"
    _write(contract_path, contract)

    drop = {
        "contact_lost_step": 2,
        "unsupported_started_step": 2,
        "reference_height_m": 0.86,
        "minimum_height_m": 0.8,
        "minimum_height_step": 3,
        "support_recontact_step": 3,
        "task_contact_recovered_step": 4,
        "fall_m": 0.06,
        "destination_inside_at_recontact": True,
    }
    score = {
        "schema_version": "adp_rigid_task_scoring.v2",
        "status": "scored",
        "task_kind": "rigid_pick_place",
        "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
        "task_succeeded": deterministic_success,
        "outcome": ("pushed_and_settled" if deterministic_success else "temporal_invariant_failed"),
        "criteria_satisfied": {"no_drop": not no_drop},
        "failed_criteria": [] if deterministic_success else ["no_drop"],
        "event_ledger": {
            "schema_version": "rigid_task_event_ledger.v1",
            "drop_events": [drop],
            "peak_task_contact_force_n": 8.0,
            "required_readback_gaps": [],
            "derived_only_from_episode_samples": True,
        },
        "learned_judge_consulted": False,
        "candidate_policy_queried_by_scorer": False,
        "report_digest": "",
    }
    score["report_digest"] = canonical_digest(score, digest_field="report_digest")
    score_path = root / "score.json"
    _write(score_path, score)

    task_samples = [
        {
            "step_index": step,
            "simulation_time_s": step * 0.1,
            "task_object_pose_world": pose,
        }
        for step, pose in enumerate(
            (
                [1.0, 2.0, 0.8, 0, 0, 0, 1],
                [1.0, 2.0, 0.86, 0, 0, 0, 1],
                [1.13, 2.0, 0.86, 0, 0, 0, 1],
                [1.15, 2.0, 0.8, 0, 0, 0, 1],
                [1.15, 2.0, 0.8, 0, 0, 0, 1],
            )
        )
    ]
    state = {
        "schema_version": "policy_episode_state_trace.v1",
        "joint_states": [],
        "task_state_samples": task_samples,
        "trace_digest": "",
    }
    state["trace_digest"] = canonical_digest(state, digest_field="trace_digest")
    state_path = root / "state.json"
    _write(state_path, state)
    contact = {
        "schema_version": "policy_episode_contact_force_trace.v1",
        "samples": [
            {
                "step_index": 2,
                "task_contact_active": False,
                "support_contact_active": False,
            },
            {
                "step_index": 3,
                "task_contact_active": False,
                "support_contact_active": True,
            },
            {
                "step_index": 4,
                "task_contact_active": True,
                "support_contact_active": True,
            },
        ],
        "typed_gap": None,
        "trace_digest": "",
    }
    contact["trace_digest"] = canonical_digest(contact, digest_field="trace_digest")
    contact_path = root / "contact.json"
    _write(contact_path, contact)

    frame_paths: list[Path] = []
    frame_rows = []
    for index in range(5):
        path = root / "media" / f"frame-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png-fixture-" + bytes([index]))
        frame_paths.append(path)
        frame_rows.append(
            {
                "frame_index": index,
                "relative_path": path.relative_to(root).as_posix(),
                "png_sha256": _sha(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "adp_observation_frame_manifest.v1",
        "episode_id": "episode-1",
        "policy_input_frames": frame_rows[:-1],
        "terminal_observation": frame_rows[-1],
        "frame_manifest_digest": "",
    }
    manifest["frame_manifest_digest"] = canonical_digest(
        manifest, digest_field="frame_manifest_digest"
    )
    manifest_path = root / "frame_manifest.json"
    _write(manifest_path, manifest)
    video_path = root / "media" / "episode.mp4"
    video_path.write_bytes(b"derived-review-video-fixture")
    return {
        "root": root,
        "contract": contract,
        "contract_path": contract_path,
        "score": score,
        "score_path": score_path,
        "state_path": state_path,
        "contact_path": contact_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "video_path": video_path,
    }


def _request(data: dict, *, videos: bool = True):
    return build_episode_interpretation_request(
        episode_id="episode-1",
        candidate_policy_id="groot_n17_droid",
        evidence_root=data["root"],
        task_success_contract_path=data["contract_path"],
        deterministic_score_path=data["score_path"],
        state_trace_path=data["state_path"],
        contact_force_trace_path=data["contact_path"],
        frame_manifest_path=data["manifest_path"],
        review_video_paths=[data["video_path"]] if videos else [],
    )


def _output(data: dict) -> EpisodeInterpreterOutput:
    score_digest = data["score"]["report_digest"]
    state_digest = json.loads(data["state_path"].read_text())["trace_digest"]
    contact_digest = json.loads(data["contact_path"].read_text())["trace_digest"]
    return EpisodeInterpreterOutput.model_validate(
        {
            "episode_outcome": "appears_complete",
            "summary": (
                "The cup was released early, fell, regained support, was recovered, "
                "and ultimately remained on the green destination."
            ),
            "events": [
                {
                    "event_type": "drop",
                    "start_step": 2,
                    "end_step": 3,
                    "start_time_seconds": 0.2,
                    "end_time_seconds": 0.3,
                    "description": "The cup lost robot/support contact and fell 6 cm.",
                    "confidence": 0.99,
                    "evidence_refs": [
                        {
                            "artifact_role": "deterministic_score",
                            "artifact_digest": score_digest,
                            "step_index": 2,
                        },
                        {
                            "artifact_role": "contact_force_trace",
                            "artifact_digest": contact_digest,
                            "step_index": 2,
                        },
                    ],
                },
                {
                    "event_type": "recovery_and_terminal_placement",
                    "start_step": 3,
                    "end_step": 4,
                    "start_time_seconds": 0.3,
                    "end_time_seconds": 0.4,
                    "description": "The cup regained support in the target and was recovered.",
                    "confidence": 0.96,
                    "evidence_refs": [
                        {
                            "artifact_role": "state_trace",
                            "artifact_digest": state_digest,
                            "step_index": 4,
                        }
                    ],
                },
            ],
            "possible_missed_events": [],
            "contract_considerations": [
                "Terminal placement and the earlier drop are separate facts."
            ],
            "confidence": 0.97,
        }
    )


def _rights(data: dict, request, interpreter, tmp_path: Path) -> Path:
    path = tmp_path / "rights.json"
    materialize_episode_interpretation_rights(
        episode_id=request.episode_id,
        input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=interpreter.identity,
        allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
        external_disclosure_authorized=False,
        accepted_by="robot-team:task-owner",
        accepted_on="2026-09-03T12:00:00Z",
        authority_reference="task-owner-approval-42",
        source_rights_admission_digest="sha256:" + "a" * 64,
        output_path=path,
    )
    return path


def test_drop_then_recovery_then_terminal_success_is_narrated(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    receipt = interpret_episode(
        request=request,
        interpreter=interpreter,
        rights_attestation_path=_rights(data, request, interpreter, tmp_path),
        output_path=tmp_path / "interpretation.json",
    )

    assert [event["event_type"] for event in receipt["learned_interpretation"]["events"]] == [
        "drop",
        "recovery_and_terminal_placement",
    ]
    assert receipt["deterministic_agreement"] == "agrees"
    assert receipt["authoritative_deterministic_result"]["task_succeeded"] is True
    assert receipt["proof_boundary"]["learned_interpretation_only"] is True
    assert receipt["proof_boundary"]["ranking_or_promotion_effect"] == "none"
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_no_drop_contract_disagreement_cannot_overwrite_failure(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    receipt = interpret_episode(
        request=request,
        interpreter=interpreter,
        rights_attestation_path=_rights(data, request, interpreter, tmp_path),
        output_path=tmp_path / "interpretation.json",
    )

    assert receipt["learned_interpretation"]["episode_outcome"] == "appears_complete"
    assert receipt["deterministic_agreement"] == "disagrees"
    assert receipt["authoritative_deterministic_result"]["task_succeeded"] is False
    assert receipt["authoritative_deterministic_result"]["failed_criteria"] == ["no_drop"]
    assert receipt["proof_boundary"]["authoritative_task_success_unchanged"] is True


def test_missing_review_video_abstains_without_calling_interpreter(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data, videos=False)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    receipt = interpret_episode(
        request=request,
        interpreter=interpreter,
        rights_attestation_path=None,
        output_path=tmp_path / "interpretation.json",
    )

    assert interpreter.call_count == 0
    assert receipt["status"] == "abstained"
    assert receipt["deterministic_agreement"] == "abstains"
    assert receipt["provider_called"] is False
    assert receipt["learned_interpretation"]["possible_missed_events"][0]["reason"] == (
        "required_review_video_missing"
    )


def test_candidate_policy_cannot_be_its_own_interpreter(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data)
    interpreter = DeterministicFixtureInterpreter(
        _output(data),
        interpreter_id="groot_n17_droid",
        principal_kind="candidate_policy",
    )

    with pytest.raises(EpisodeInterpretationError, match="candidate_policy_self_grading_forbidden"):
        interpret_episode(
            request=request,
            interpreter=interpreter,
            rights_attestation_path=None,
            output_path=tmp_path / "interpretation.json",
        )
    assert interpreter.call_count == 0


def test_rights_are_bound_to_exact_input_digest(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    rights_path = _rights(data, request, interpreter, tmp_path)
    rights = json.loads(rights_path.read_text())
    rights["input_bundle_digest"] = "sha256:" + "b" * 64
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    _write(rights_path, rights)

    with pytest.raises(EpisodeInterpretationError, match="episode_interpretation_rights_invalid"):
        interpret_episode(
            request=request,
            interpreter=interpreter,
            rights_attestation_path=rights_path,
            output_path=tmp_path / "interpretation.json",
        )
    assert interpreter.call_count == 0


def test_output_cannot_claim_authoritative_success() -> None:
    value = EpisodeInterpreterOutput(
        episode_outcome="unclear",
        summary="Insufficient evidence.",
        confidence=0.1,
    ).model_dump()
    value["task_succeeded"] = True
    with pytest.raises(ValueError):
        EpisodeInterpreterOutput.model_validate(value)


class _ExternalFixtureInterpreter(DeterministicFixtureInterpreter):
    @property
    def identity(self) -> InterpreterIdentity:
        base = super().identity
        return InterpreterIdentity(
            interpreter_id=base.interpreter_id,
            principal_kind=base.principal_kind,
            provider_id="example-provider",
            execution_site="external_provider",
            runtime="example-runtime",
            model="example-model",
            model_version="2026-09-03",
        )


def test_external_provider_requires_explicit_disclosure_authority(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data)
    interpreter = _ExternalFixtureInterpreter(_output(data))
    rights_path = tmp_path / "rights.json"
    materialize_episode_interpretation_rights(
        episode_id=request.episode_id,
        input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=interpreter.identity,
        allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
        external_disclosure_authorized=False,
        accepted_by="robot-team:task-owner",
        accepted_on="2026-09-03T12:00:00Z",
        authority_reference="task-owner-approval-42",
        source_rights_admission_digest="sha256:" + "a" * 64,
        output_path=rights_path,
    )

    with pytest.raises(EpisodeInterpretationError, match="episode_interpretation_rights_invalid"):
        interpret_episode(
            request=request,
            interpreter=interpreter,
            rights_attestation_path=rights_path,
            output_path=tmp_path / "interpretation.json",
        )


class _FakeAgentsSDKInvoker:
    def __init__(self, output: EpisodeInterpreterOutput) -> None:
        self.output = output
        self.calls = []

    def invoke(self, spec, input_value):
        self.calls.append((spec, input_value))
        return AgentsSDKInvocationResult(
            output=self.output,
            provider="openai",
            model=spec.model,
            sdk_version="0.19.1",
            latency_seconds=0.01,
            usage={},
            cost_usd=0.0,
            cost_status="hermetic_fake",
        )


def test_openai_adapter_uses_agents_sdk_and_discloses_frame_sampling_gap(
    tmp_path: Path,
) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data)
    invoker = _FakeAgentsSDKInvoker(_output(data))
    interpreter = OpenAIMultimodalEpisodeInterpreter(
        invoker=invoker,
        model="gpt-6-sol",
        model_version="gpt-6-sol-2026-09-03",
        max_frames=2,
        run_id="quick10-interpretation-batch",
    )
    rights_path = tmp_path / "rights.json"
    materialize_episode_interpretation_rights(
        episode_id=request.episode_id,
        input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=interpreter.identity,
        allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
        external_disclosure_authorized=True,
        accepted_by="robot-team:task-owner",
        accepted_on="2026-09-03T12:00:00Z",
        authority_reference="task-owner-approval-42",
        source_rights_admission_digest="sha256:" + "a" * 64,
        output_path=rights_path,
    )

    receipt = interpret_episode(
        request=request,
        interpreter=interpreter,
        rights_attestation_path=rights_path,
        output_path=tmp_path / "interpretation.json",
    )

    assert len(invoker.calls) == 1
    spec, provider_input = invoker.calls[0]
    assert spec.output_type is EpisodeInterpreterOutput
    assert spec.run_id == "quick10-interpretation-batch"
    images = [
        item
        for message in provider_input
        for item in message["content"]
        if item["type"] == "input_image"
    ]
    assert len(images) == 2
    assert all(item["detail"] == "low" for item in images)
    compact = json.loads(provider_input[0]["content"][1]["text"])
    assert "policy_input_observations" not in compact["frame_manifest"]
    assert "review_observations" not in compact["frame_manifest"]
    assert set(compact["state_trace"]) == {
        "schema_version",
        "trace_digest",
        "task_state_samples",
        "joint_states",
        "sampling",
    }
    assert receipt["provider_called"] is True
    assert (
        receipt["learned_interpretation"]["possible_missed_events"][-1]["reason"]
        == "OpenAI adapter sampled 2 of 5 lossless frames."
    )


def test_openai_adapter_preserves_first_terminal_and_event_nearby_camera_groups(
    tmp_path: Path,
) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    request = _request(data)
    score = copy.deepcopy(request.deterministic_score)
    score["event_ledger"] = {
        **score["event_ledger"],
        "drop_events": [{"step_index": 50}],
    }
    state = {
        **request.state_trace,
        "task_state_samples": [
            {"step_index": 0},
            {"step_index": 100},
        ],
    }
    request = replace(request, deterministic_score=score, state_trace=state)
    frame_rows = [
        {"simulation_time_s": float(time), "camera_id": camera}
        for time in range(5)
        for camera in ("external", "overview", "wrist")
    ]
    interpreter = OpenAIMultimodalEpisodeInterpreter(
        invoker=_FakeAgentsSDKInvoker(_output(data)),
        model="gpt-6-luna",
        model_version="gpt-6-luna",
        max_frames=9,
    )

    selected = interpreter._selected_frame_indices(request, frame_rows)

    assert selected == [0, 1, 2, 6, 7, 8, 12, 13, 14]


def test_openai_adapter_bounds_trace_rows_while_preserving_event_steps() -> None:
    rows = [
        {"step_index": index, "task_object_pose_world": [float(index), 0.0, 0.0]}
        for index in range(261)
    ]

    selected = OpenAIMultimodalEpisodeInterpreter._selected_trace_rows(
        rows,
        event_steps={17, 129, 244},
    )

    assert len(selected) == 96
    assert {17, 129, 244}.issubset({row["step_index"] for row in selected})
    assert selected[0]["step_index"] == 0
    assert selected[-1]["step_index"] == 260


def test_tampered_score_fails_before_interpreter(tmp_path: Path) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    score = copy.deepcopy(data["score"])
    score["task_succeeded"] = False
    _write(data["score_path"], score)

    with pytest.raises(EpisodeInterpretationError, match="score_digest_invalid"):
        _request(data)


# -- Streamed episodes: frames and the review video stay in the archive ------


def _streamed(data: dict, tmp_path: Path, monkeypatch):
    from tests.provider_output_fixtures import serve_member_views, stream_evidence_tree

    streamed = stream_evidence_tree(data["root"], tmp_path / "streamed" / "attempt_001")
    serve_member_views(monkeypatch, streamed.store)
    return streamed


def _streamed_request(data: dict, streamed, *, videos: bool = True):
    def relative(path: Path) -> str:
        return path.relative_to(data["root"]).as_posix()

    return build_episode_interpretation_request(
        episode_id="episode-1",
        candidate_policy_id="groot_n17_droid",
        evidence_root=streamed.evidence,
        task_success_contract_path=relative(data["contract_path"]),
        deterministic_score_path=relative(data["score_path"]),
        state_trace_path=relative(data["state_path"]),
        contact_force_trace_path=relative(data["contact_path"]),
        frame_manifest_path=relative(data["manifest_path"]),
        review_video_paths=[relative(data["video_path"])] if videos else [],
    )


def test_streamed_request_has_the_download_mode_input_bundle_digest(tmp_path: Path, monkeypatch) -> None:
    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    downloaded = _request(data)
    streamed = _streamed(data, tmp_path, monkeypatch)
    assert streamed.remote() == ["media/episode.mp4", *[f"media/frame-{index}.png" for index in range(5)]]

    request = _streamed_request(data, streamed)

    assert request.input_receipt == downloaded.input_receipt
    assert request.input_receipt["all_source_bytes_rehashed"] is True
    assert downloaded.frame_reader is None and downloaded.source_digest_basis is None
    archive = streamed.index["archive"]
    assert request.source_digest_basis == {
        "schema_version": "episode_interpretation_source_digest_basis.v1",
        "basis": "provider_output_member_index",
        "member_index_digest": streamed.index["index_digest"],
        "archive_sha256": archive["sha256"],
        "durable_reference_uri": archive["durable_reference"]["uri"],
        "full_byte_readback_passed": True,
        "indexed_lossless_frames": [0, 1, 2, 3, 4],
        "indexed_review_videos": [0],
    }
    assert streamed.data_ranges() == []  # every binding came from the index
    # A frame the manifest expects but the index does not vouch for keeps download mode's code.
    manifest = json.loads((streamed.evidence / "frame_manifest.json").read_text())
    manifest["policy_input_frames"][1]["png_sha256"] = "sha256:" + "0" * 64
    manifest["frame_manifest_digest"] = canonical_digest(manifest, digest_field="frame_manifest_digest")
    for path in (streamed.evidence / "frame_manifest.json", data["manifest_path"]):
        path.chmod(0o640)
        _write(path, manifest)
    for build in (lambda: _request(data), lambda: _streamed_request(data, streamed)):
        with pytest.raises(EpisodeInterpretationError, match="^episode_interpretation_frame_digest_mismatch:1$"):
            build()


def test_v1_interpreter_reads_only_its_selected_frames_by_range(tmp_path: Path, monkeypatch) -> None:
    import base64

    data = _episode_root(tmp_path, no_drop=False, deterministic_success=True)
    streamed = _streamed(data, tmp_path, monkeypatch)
    receipts = {}
    for mode, request in (("download", _request(data)), ("stream", _streamed_request(data, streamed))):
        invoker = _FakeAgentsSDKInvoker(_output(data))
        interpreter = OpenAIMultimodalEpisodeInterpreter(
            invoker=invoker, model="gpt-6-sol", model_version="gpt-6-sol-2026-09-03", max_frames=2,
            run_id="quick10-interpretation-batch")
        rights_path = tmp_path / f"{mode}-rights.json"
        materialize_episode_interpretation_rights(
            episode_id=request.episode_id, input_bundle_digest=request.input_receipt["input_bundle_digest"],
            identity=interpreter.identity, allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
            external_disclosure_authorized=True, accepted_by="robot-team:task-owner",
            accepted_on="2026-09-03T12:00:00Z", authority_reference="task-owner-approval-42",
            source_rights_admission_digest="sha256:" + "a" * 64, output_path=rights_path)
        before = len(streamed.data_ranges())
        receipts[mode] = interpret_episode(request=request, interpreter=interpreter,
                                           rights_attestation_path=rights_path,
                                           output_path=tmp_path / f"{mode}-interpretation.json")
        frames = request.input_receipt["artifacts"]["lossless_frames"]
        selected = interpreter._selected_frame_indices(request, frames)
        images = [item["image_url"] for message in invoker.calls[0][1] for item in message["content"]
                  if item["type"] == "input_image"]
        assert images == ["data:image/png;base64," + base64.b64encode(
            (data["root"] / frames[index]["relative_path"]).read_bytes()).decode() for index in selected]
        if mode == "stream":
            rows = [streamed.rows[frames[index]["relative_path"]] for index in selected]
            assert streamed.data_ranges()[before:] == [
                (row["data_offset"], row["data_offset"] + row["compressed_size"] - 1) for row in rows]
    # The same receipt, save the recorded basis of the index-bound digests.
    streamed_receipt = dict(receipts["stream"])
    assert streamed_receipt.pop("source_digest_basis")["basis"] == "provider_output_member_index"
    assert {key: value for key, value in streamed_receipt.items() if key != "receipt_digest"} == {
        key: value for key, value in receipts["download"].items() if key != "receipt_digest"}
    assert not any(path.suffix in {".png", ".mp4"} for path in streamed.evidence.rglob("*"))
