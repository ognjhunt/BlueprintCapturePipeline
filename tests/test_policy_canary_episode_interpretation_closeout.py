from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.episode_interpretation import (
    DeterministicFixtureInterpreter,
    OpenAIMultimodalEpisodeInterpreter,
    build_episode_interpretation_request,
    interpret_episode,
    materialize_episode_interpretation_rights,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)
from blueprint_pipeline.policy_canary_episode_interpretation_closeout import (
    materialize_policy_canary_episode_interpretations,
)
from tests.test_episode_interpretation import _episode_root, _output


def _record(path: Path, root: Path, role: str) -> dict[str, object]:
    import hashlib

    return {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "media_type": "video/mp4" if path.suffix == ".mp4" else "application/json",
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _session(tmp_path: Path, *, deterministic_success: bool = True) -> tuple[dict, dict]:
    data = _episode_root(
        tmp_path, no_drop=not deterministic_success, deterministic_success=deterministic_success
    )
    root = data["root"]
    artifacts = {
        "score_receipt": _record(data["score_path"], root, "score_receipt"),
        "state_trace": _record(data["state_path"], root, "state_trace"),
        "contact_force_trace": _record(
            data["contact_path"], root, "contact_force_trace"
        ),
        "frame_manifest": _record(
            data["manifest_path"], root, "lossless_frame_manifest"
        ),
        "review_video": _record(data["video_path"], root, "review_video"),
    }
    inventory = list(artifacts.values())
    for path in sorted((root / "media").glob("*.png")):
        inventory.append(_record(path, root, "lossless_frame"))
    episodes = []
    for index in range(20):
        candidate = "pi05_droid" if index < 10 else "groot_n17_droid"
        episodes.append(
            {
                "candidate_id": candidate,
                "cell_id": f"quick-cell-{index % 10}",
                "seed": 3100 + index % 10,
                "status": "completed",
                "episode": {"episode_id": f"episode-{index}", "score": data["score"]},
                "evidence_artifacts": dict(artifacts),
            }
        )
    result = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "run_id": "scene839873-quick10",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "task_success_contract": data["contract"],
        "task_success_contract_digest": data["contract"]["contract_digest"],
        "episodes": episodes,
        "artifact_inventory": inventory,
        "artifact_inventory_digest": canonical_digest({"value": inventory}),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return data, result


class _FixtureRunner:
    def __init__(self, interpreter: DeterministicFixtureInterpreter) -> None:
        self.interpreter = interpreter

    @property
    def identity(self):
        return self.interpreter.identity

    def __call__(self, *, request, rights_attestation_path, output_path):
        return interpret_episode(
            request=request,
            interpreter=self.interpreter,
            rights_attestation_path=rights_attestation_path,
            output_path=output_path,
        )


def _rights_for_all(data: dict, result: dict, rights_root: Path, interpreter) -> None:
    evidence = data["root"]
    contract = evidence / "episode_interpretation_sources/task_success_contract.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(canonical_json(result["task_success_contract"]) + "\n", encoding="utf-8")
    for row in result["episodes"]:
        artifacts = row["evidence_artifacts"]
        request = build_episode_interpretation_request(
            episode_id=row["episode"]["episode_id"],
            candidate_policy_id=row["candidate_id"],
            evidence_root=evidence,
            task_success_contract_path=contract,
            deterministic_score_path=artifacts["score_receipt"]["relative_path"],
            state_trace_path=artifacts["state_trace"]["relative_path"],
            contact_force_trace_path=artifacts["contact_force_trace"]["relative_path"],
            frame_manifest_path=artifacts["frame_manifest"]["relative_path"],
            review_video_paths=[artifacts["review_video"]["relative_path"]],
        )
        token = request.input_receipt["input_bundle_digest"].removeprefix("sha256:")
        materialize_episode_interpretation_rights(
            episode_id=request.episode_id,
            input_bundle_digest=request.input_receipt["input_bundle_digest"],
            identity=interpreter.identity,
            allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
            external_disclosure_authorized=False,
            accepted_by="robot-team:task-owner",
            accepted_on="2026-09-03T12:00:00Z",
            authority_reference="quick10-interpretation-approval",
            source_rights_admission_digest="sha256:" + "a" * 64,
            output_path=rights_root / f"{token}.json",
        )


def test_closeout_materializes_twenty_receipts_and_replay_does_not_reinvoke(
    tmp_path: Path,
) -> None:
    data, result = _session(tmp_path)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    runner = _FixtureRunner(interpreter)
    rights_root = tmp_path / "rights"
    _rights_for_all(data, result, rights_root, interpreter)

    first = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=result,
        runner=runner,
        rights_root=rights_root,
        environment={},
    )
    second = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=first,
        runner=runner,
        rights_root=rights_root,
        environment={},
    )

    assert first["episode_interpretation"]["receipt_count"] == 20
    assert first["episode_interpretation"]["completed_count"] == 20
    assert interpreter.call_count == 20
    assert second["episode_interpretation"]["reused_receipt_count"] == 20
    assert second["episode_interpretation"]["provider_call_count"] == 0
    assert len(list((data["root"] / "episode_interpretation/receipts").glob("*.json"))) == 20
    assert all(
        row["episode"]["score"]["task_succeeded"] is True for row in second["episodes"]
    )


def test_prior_attempt_marker_prevents_duplicate_inference(tmp_path: Path) -> None:
    data, result = _session(tmp_path)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    runner = _FixtureRunner(interpreter)
    rights_root = tmp_path / "rights"
    _rights_for_all(data, result, rights_root, interpreter)
    first_token = sorted(path.stem for path in rights_root.glob("*.json"))[0]
    marker = data["root"] / "episode_interpretation/attempted" / f"{first_token}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")

    interpreted = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=result,
        runner=runner,
        rights_root=rights_root,
        environment={},
    )

    assert interpreter.call_count == 19
    reasons = [
        row["abstention_reason"]
        for row in interpreted["episode_interpretation"]["receipts"]
    ]
    assert reasons.count("prior_interpretation_execution_ambiguous") == 1


def test_unavailable_configuration_yields_twenty_nonblocking_abstentions(
    tmp_path: Path,
) -> None:
    data, result = _session(tmp_path)

    interpreted = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=result,
        environment={},
    )

    summary = interpreted["episode_interpretation"]
    assert summary["status"] == "abstained"
    assert summary["receipt_count"] == summary["abstained_count"] == 20
    assert summary["provider_call_count"] == 0
    assert summary["score_overwrite_performed"] is False


def test_interpreter_disagreement_is_retained_without_score_overwrite(tmp_path: Path) -> None:
    data, result = _session(tmp_path, deterministic_success=False)
    interpreter = DeterministicFixtureInterpreter(_output(data))
    runner = _FixtureRunner(interpreter)
    rights_root = tmp_path / "rights"
    _rights_for_all(data, result, rights_root, interpreter)

    interpreted = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=result,
        runner=runner,
        rights_root=rights_root,
        environment={},
    )

    assert interpreted["episode_interpretation"]["disagreement_count"] == 20
    assert all(
        row["episode"]["score"]["task_succeeded"] is False
        for row in interpreted["episodes"]
    )
    assert interpreted["episode_interpretation"]["ranking_or_promotion_effect"] == "none"


def _production_environment(tmp_path: Path, *, profile: dict) -> dict[str, str]:
    files = {}
    for name in ("key", "admin", "scope"):
        path = tmp_path / name
        path.write_text("configured", encoding="utf-8")
        files[name] = path
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(canonical_json(profile) + "\n", encoding="utf-8")
    return {
        "BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE": str(profile_path),
        "BLUEPRINT_LIVE_AGENTS_SDK": "1",
        "OPENAI_API_KEY_FILE": str(files["key"]),
        "OPENAI_ADMIN_API_KEY_FILE": str(files["admin"]),
        "OPENAI_PROJECT_ID": "project-episode-interpretation",
        "BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_API_KEY_ID": "key-scope",
        "BLUEPRINT_OPENAI_EPISODE_INTERPRETATION_COST_SCOPE_ATTESTATION_FILE": str(
            files["scope"]
        ),
    }


def _profile() -> dict:
    value = {
        "schema_version": "policy_canary_episode_interpreter_profile.v1",
        "status": "configured",
        "interpreter_id": "openai_multimodal_episode_interpreter_v1",
        "provider_id": "openai",
        "runtime": "openai_agents_sdk",
        "model": "gpt-5.6-terra",
        "model_version": "gpt-5.6-terra-2026-09-03",
        "max_frames": 64,
        "max_input_tokens": 240_000,
        "max_output_tokens": 8_000,
        # One cap covers the whole twenty-call batch; it is not reset per episode.
        "max_cost_usd": 1.5,
        "profile_digest": "",
    }
    value["profile_digest"] = canonical_digest(value, digest_field="profile_digest")
    return value


def test_production_route_uses_one_cost_gate_and_one_aggregate_sdk_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.policy_canary_episode_interpretation_closeout as closeout

    data, result = _session(tmp_path)
    profile = _profile()
    environment = _production_environment(tmp_path, profile=profile)
    rights_root = tmp_path / "rights"

    class RightsInvoker:
        def invoke(self, *_args, **_kwargs):  # pragma: no cover - rights construction only
            raise AssertionError

    rights_interpreter = OpenAIMultimodalEpisodeInterpreter(
        invoker=RightsInvoker(),
        model=profile["model"],
        model_version=profile["model_version"],
    )
    # Build the same per-input human attestations the production resolver consumes.
    evidence = data["root"]
    contract = evidence / "episode_interpretation_sources/task_success_contract.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(canonical_json(result["task_success_contract"]) + "\n", encoding="utf-8")
    for row in result["episodes"]:
        artifacts = row["evidence_artifacts"]
        request = build_episode_interpretation_request(
            episode_id=row["episode"]["episode_id"],
            candidate_policy_id=row["candidate_id"],
            evidence_root=evidence,
            task_success_contract_path=contract,
            deterministic_score_path=artifacts["score_receipt"]["relative_path"],
            state_trace_path=artifacts["state_trace"]["relative_path"],
            contact_force_trace_path=artifacts["contact_force_trace"]["relative_path"],
            frame_manifest_path=artifacts["frame_manifest"]["relative_path"],
            review_video_paths=[artifacts["review_video"]["relative_path"]],
        )
        materialize_episode_interpretation_rights(
            episode_id=request.episode_id,
            input_bundle_digest=request.input_receipt["input_bundle_digest"],
            identity=rights_interpreter.identity,
            allowed_artifact_roles=rights_interpreter.disclosed_artifact_roles(request),
            external_disclosure_authorized=True,
            accepted_by="robot-team:task-owner",
            accepted_on="2026-09-03T12:00:00Z",
            authority_reference="quick10-openai-approval",
            source_rights_admission_digest="sha256:" + "b" * 64,
            output_path=(
                rights_root
                / f"{request.input_receipt['input_bundle_digest'].removeprefix('sha256:')}.json"
            ),
        )

    gate_calls = {"build": 0, "reserve": 0, "complete": 0}

    class FakeGate:
        def reserve(self):
            gate_calls["reserve"] += 1

        def complete(self, **_kwargs):
            gate_calls["complete"] += 1

    def build_gate(**_kwargs):
        gate_calls["build"] += 1
        return FakeGate()

    invokers = []

    class FakeInvoker:
        def __init__(self, config):
            self.config = config
            self.calls = 0
            invokers.append(self)

        def configure_reservation_audit(self, **_kwargs):
            return None

        def invoke(self, spec, _input_value):
            self.calls += 1
            return AgentsSDKInvocationResult(
                output=_output(data),
                provider="openai",
                model=spec.model,
                sdk_version="hermetic",
                latency_seconds=0.0,
                usage={},
                cost_usd=0.0,
                cost_status="hermetic",
            )

    monkeypatch.setattr(closeout, "build_openai_official_cost_run_gate", build_gate)
    monkeypatch.setattr(closeout, "OpenAIAgentsSDKInvoker", FakeInvoker)

    interpreted = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=result,
        rights_root=rights_root,
        environment=environment,
    )

    assert gate_calls == {"build": 1, "reserve": 1, "complete": 1}
    assert len(invokers) == 1 and invokers[0].calls == 20
    assert invokers[0].config.max_inference_cost_usd == profile["max_cost_usd"]
    assert interpreted["episode_interpretation"]["provider_call_count"] == 20
    assert interpreted["episode_interpretation"]["provider_invocation_attempt_count"] == 20


def test_production_route_does_not_reserve_cost_without_eligible_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.policy_canary_episode_interpretation_closeout as closeout

    data, result = _session(tmp_path)
    environment = _production_environment(tmp_path, profile=_profile())
    rights_root = tmp_path / "empty-rights"
    rights_root.mkdir()
    monkeypatch.setattr(
        closeout,
        "build_openai_official_cost_run_gate",
        lambda **_kwargs: pytest.fail("cost gate constructed without eligible rights"),
    )

    interpreted = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path,
        evidence_root=data["root"],
        session_result=result,
        rights_root=rights_root,
        environment=environment,
    )

    assert interpreted["episode_interpretation"]["provider_call_count"] == 0
    assert interpreted["episode_interpretation"]["abstained_count"] == 20
