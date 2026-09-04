"""Best-effort learned interpretation during policy-canary closeout.

Deterministic episode truth is already sealed when this module is called.  The
module may add explanation receipts, but every configuration, rights, provider,
or model failure becomes an explicit abstention and is never raised into the
authoritative billing/teardown/delivery path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV
from .decision_evidence_contracts import canonical_digest, canonical_json
from .episode_interpretation import (
    OPENAI_ADAPTER_ID,
    RECEIPT_SCHEMA_VERSION,
    EpisodeInterpretationRequest,
    InterpreterIdentity,
    OpenAIMultimodalEpisodeInterpreter,
    build_episode_interpretation_request,
    interpret_episode,
    materialize_episode_interpretation_abstention,
    validate_episode_interpretation_rights,
)
from .episode_interpretation_batch_authority import (
    derive_episode_interpretation_rights,
    validate_episode_interpretation_batch_authority,
)
from .openai_official_cost_gate import build_openai_official_cost_run_gate
from .task_evaluation_supervisor.agents_sdk import (
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


SCHEMA_VERSION = "policy_canary_episode_interpretation_closeout.v1"
PLAN_SCHEMA_VERSION = "policy_canary_episode_interpretation_plan.v1"
PROFILE_SCHEMA_VERSION = "policy_canary_episode_interpreter_profile.v1"
BATCH_AUTHORITY_ENV = (
    "BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_BATCH_AUTHORITY_FILE"
)


class EpisodeInterpretationRunner(Protocol):
    """Injectable runner used by hermetic closeout tests."""

    @property
    def identity(self) -> InterpreterIdentity: ...

    def __call__(
        self,
        *,
        request: EpisodeInterpretationRequest,
        rights_attestation_path: Path,
        output_path: Path,
    ) -> Mapping[str, Any]: ...


class _RightsOnlyInterpreter:
    """Validate OpenAI disclosure rights before any cost reservation."""

    def __init__(self, identity: InterpreterIdentity) -> None:
        self.identity = identity

    def disclosed_artifact_roles(
        self, request: EpisodeInterpretationRequest
    ) -> tuple[str, ...]:
        del request
        return (
            "task_success_contract",
            "deterministic_score",
            "state_trace",
            "contact_force_trace",
            "frame_manifest",
            "lossless_frame",
        )

    def interpret(self, request: EpisodeInterpretationRequest) -> Any:
        del request
        raise AssertionError("rights-only interpreter cannot execute")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("episode_interpretation_file_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("episode_interpretation_file_invalid")
    return dict(value)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    content = canonical_json(dict(value)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("episode_interpretation_symlink_forbidden")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("episode_interpretation_immutable_conflict")


def _artifact(path: Path, *, root: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("episode_interpretation_artifact_invalid")
    return {
        "role": role,
        "media_type": "application/json",
        "relative_path": path.resolve().relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_path(root: Path, record: Any) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError("episode_interpretation_source_record_missing")
    relative = str(record.get("relative_path") or "")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("episode_interpretation_source_outside_root")
    if path.is_symlink() or not path.is_file():
        raise ValueError("episode_interpretation_source_missing")
    if _sha256(path) != record.get("sha256"):
        raise ValueError("episode_interpretation_source_digest_mismatch")
    return path


def _manifest_relative_frames(manifest: Mapping[str, Any]) -> list[str]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(manifest.get("policy_input_frames"), list):
        rows.extend(row for row in manifest["policy_input_frames"] if isinstance(row, Mapping))
    for field in ("policy_input_observations", "review_observations"):
        for observation in manifest.get(field) or []:
            views = observation.get("views") if isinstance(observation, Mapping) else None
            if isinstance(views, Mapping):
                rows.extend(row for row in views.values() if isinstance(row, Mapping))
    terminal = manifest.get("terminal_observation")
    if isinstance(terminal, Mapping):
        views = terminal.get("views")
        if isinstance(views, Mapping):
            rows.extend(row for row in views.values() if isinstance(row, Mapping))
        else:
            rows.append(terminal)
    return [
        str(row.get("relative_path") or row.get("path") or "")
        for row in rows
        if str(row.get("relative_path") or row.get("path") or "")
    ]


def _episode_evidence_root(*, run_evidence_root: Path, manifest_path: Path) -> Path:
    manifest = _read(manifest_path)
    relative_frames = _manifest_relative_frames(manifest)
    if not relative_frames:
        raise ValueError("episode_interpretation_frame_inventory_missing")
    candidate = manifest_path.parent
    while run_evidence_root == candidate or run_evidence_root in candidate.parents:
        if all((candidate / relative).is_file() for relative in relative_frames):
            return candidate
        if candidate == run_evidence_root:
            break
        candidate = candidate.parent
    raise ValueError("episode_interpretation_frame_root_unresolved")


def _validated_receipt(path: Path, *, input_bundle_digest: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        value.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or value.get("input_bundle_digest") != input_bundle_digest
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("proof_boundary", {}).get("ranking_or_promotion_effect") != "none"
        or value.get("proof_boundary", {}).get("authoritative_task_success_unchanged")
        is not True
    ):
        return None
    return value


def _validated_rights_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        value = _read(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    digest = str(value.get("rights_digest") or "")
    return (
        digest
        if digest == canonical_digest(value, digest_field="rights_digest")
        else None
    )


def _load_profile(environment: Mapping[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    raw = str(
        environment.get("BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE") or ""
    ).strip()
    if not raw:
        return None, "interpreter_profile_unavailable"
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return None, "interpreter_profile_invalid"
        value = _read(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None, "interpreter_profile_invalid"
    numbers = ("max_frames", "max_input_tokens", "max_output_tokens")
    if (
        value.get("schema_version") != PROFILE_SCHEMA_VERSION
        or value.get("status") != "configured"
        or value.get("interpreter_id") != OPENAI_ADAPTER_ID
        or value.get("provider_id") != "openai"
        or value.get("runtime") != "openai_agents_sdk"
        or not str(value.get("model") or "").strip()
        or not str(value.get("model_version") or "").strip()
        or any(not isinstance(value.get(field), int) for field in numbers)
        or not isinstance(value.get("max_cost_usd"), (int, float))
        or isinstance(value.get("max_cost_usd"), bool)
        or float(value.get("max_cost_usd") or 0) <= 0
        or value.get("profile_digest")
        != canonical_digest(value, digest_field="profile_digest")
    ):
        return None, "interpreter_profile_invalid"
    return value, None


def _production_prerequisite_reason(
    environment: Mapping[str, str], *, rights_root: Path | None
) -> str | None:
    if rights_root is None or not rights_root.is_dir() or rights_root.is_symlink():
        return "rights_attestation_unavailable"
    required = {
        "OPENAI_API_KEY_FILE": "interpreter_secret_unavailable",
        "OPENAI_ADMIN_API_KEY_FILE": "official_cost_gate_unavailable",
        "OPENAI_PROJECT_ID": "official_cost_gate_unavailable",
        "BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_API_KEY_ID": (
            "official_cost_gate_unavailable"
        ),
        "BLUEPRINT_OPENAI_EPISODE_INTERPRETATION_COST_SCOPE_ATTESTATION_FILE": (
            "official_cost_gate_unavailable"
        ),
    }
    for name, reason in required.items():
        value = str(environment.get(name) or "").strip()
        if not value:
            return reason
        if name.endswith("_FILE"):
            path = Path(value).expanduser()
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                return reason
    if str(environment.get(LIVE_AGENTS_SDK_ENV) or "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "provider_invocation_not_authorized"
    return None


def _abstain(
    request: EpisodeInterpretationRequest,
    *,
    reason: str,
    path: Path,
    identity: InterpreterIdentity | None,
) -> dict[str, Any]:
    return materialize_episode_interpretation_abstention(
        request=request,
        reason=reason,
        output_path=path,
        interpreter_identity=identity,
    )


def materialize_policy_canary_episode_interpretations(
    *,
    run_root: str | Path,
    evidence_root: str | Path,
    session_result: Mapping[str, Any],
    runner: EpisodeInterpretationRunner | None = None,
    rights_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    batch_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add optional receipts and return a cloned result with a closeout summary."""

    env = dict(os.environ if environment is None else environment)
    root = Path(run_root).expanduser().resolve()
    evidence = Path(evidence_root).expanduser().resolve()
    result = json.loads(json.dumps(dict(session_result), allow_nan=False))
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("policy_canary_interpretation_episode_inventory_invalid")
    interpretation_root = evidence / "episode_interpretation"

    profile: dict[str, Any] | None = None
    unavailable_reason: str | None = None
    selected_runner = runner
    rights = (
        Path(rights_root).expanduser().resolve()
        if rights_root is not None
        else Path(str(env.get("BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_RIGHTS_ROOT"))).expanduser().resolve()
        if env.get("BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_RIGHTS_ROOT")
        else None
    )
    if selected_runner is None:
        profile, unavailable_reason = _load_profile(env)
        unavailable_reason = unavailable_reason or _production_prerequisite_reason(
            env, rights_root=rights
        )

    requests: list[tuple[dict[str, Any], EpisodeInterpretationRequest, Path, Path, Path]] = []
    receipts: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for row in episodes:
        source = row.get("evidence_artifacts")
        source = dict(source) if isinstance(source, Mapping) else {}
        raw_episode = row.get("episode")
        raw_episode = dict(raw_episode) if isinstance(raw_episode, Mapping) else {}
        episode_id = str(
            raw_episode.get("episode_id")
            or f"{result.get('run_id', 'policy-canary')}--{row.get('cell_id')}--{row.get('candidate_id')}"
        )
        try:
            score_path = _source_path(evidence, source.get("score_receipt"))
            state_path = _source_path(evidence, source.get("state_trace"))
            contact_path = _source_path(evidence, source.get("contact_force_trace"))
            manifest_path = _source_path(evidence, source.get("frame_manifest"))
            video_paths = (
                [_source_path(evidence, source.get("review_video"))]
                if source.get("review_video") is not None
                else []
            )
            episode_evidence = _episode_evidence_root(
                run_evidence_root=evidence,
                manifest_path=manifest_path,
            )
            contract_path = (
                episode_evidence
                / "episode_interpretation_sources"
                / "task_success_contract.json"
            )
            _write_once(contract_path, dict(result.get("task_success_contract") or {}))
            contract_artifact = _artifact(
                contract_path,
                root=evidence,
                role="episode_interpretation_task_success_contract",
            )
            if contract_artifact["relative_path"] not in {
                item["relative_path"] for item in artifacts
            }:
                artifacts.append(contract_artifact)
            request = build_episode_interpretation_request(
                episode_id=episode_id,
                candidate_policy_id=str(row.get("candidate_id") or ""),
                evidence_root=episode_evidence,
                task_success_contract_path=contract_path,
                deterministic_score_path=score_path,
                state_trace_path=state_path,
                contact_force_trace_path=contact_path,
                frame_manifest_path=manifest_path,
                review_video_paths=video_paths,
            )
        except Exception:
            # A deterministic evidence gap is already represented by the provider
            # result.  It cannot be converted into an inferred input bundle.
            receipts.append(
                {
                    "episode_id": episode_id,
                    "status": "abstained",
                    "abstention_reason": "interpretation_input_bundle_unavailable",
                    "input_bundle_digest": None,
                    "receipt": None,
                }
            )
            continue
        token = request.input_receipt["input_bundle_digest"].removeprefix("sha256:")
        plan_path = interpretation_root / "plans" / f"{token}.json"
        receipt_path = interpretation_root / "receipts" / f"{token}.json"
        marker_path = interpretation_root / "attempted" / f"{token}.json"
        requests.append((row, request, plan_path, receipt_path, marker_path))

    identity: InterpreterIdentity | None = getattr(selected_runner, "identity", None)
    interpreter: OpenAIMultimodalEpisodeInterpreter | None = None
    cost_gate = None
    audit: InferenceReservationAudit | None = None
    cost_completion_error: str | None = None
    invalid_rights_inputs: set[str] = set()
    if selected_runner is None and profile is not None and unavailable_reason is None and requests:
        identity = InterpreterIdentity(
            interpreter_id=str(profile["interpreter_id"]),
            principal_kind="independent_interpreter",
            provider_id="openai",
            execution_site="external_provider",
            runtime="openai_agents_sdk",
            model=str(profile["model"]),
            model_version=str(profile["model_version"]),
        )
        eligible = []
        rights_interpreter = _RightsOnlyInterpreter(identity)
        resolved_batch_authority = dict(batch_authority) if batch_authority else None
        batch_authority_path = str(env.get(BATCH_AUTHORITY_ENV) or "").strip()
        if resolved_batch_authority is None and batch_authority_path:
            try:
                resolved_batch_authority = validate_episode_interpretation_batch_authority(
                    _read(Path(batch_authority_path).expanduser().resolve()),
                    run_id=str(result.get("run_id") or root.name),
                    interpreter=rights_interpreter,
                    interpreter_profile_digest=str(profile["profile_digest"]),
                    maximum_cost_usd=float(profile["max_cost_usd"]),
                )
            except Exception:
                unavailable_reason = "batch_rights_authority_invalid"
        elif resolved_batch_authority is not None:
            try:
                resolved_batch_authority = validate_episode_interpretation_batch_authority(
                    resolved_batch_authority,
                    run_id=str(result.get("run_id") or root.name),
                    interpreter=rights_interpreter,
                    interpreter_profile_digest=str(profile["profile_digest"]),
                    maximum_cost_usd=float(profile["max_cost_usd"]),
                )
            except Exception:
                resolved_batch_authority = None
                unavailable_reason = "batch_rights_authority_invalid"
        for _, request, _, receipt_path, marker_path in requests:
            rights_path = rights / f"{request.input_receipt['input_bundle_digest'].removeprefix('sha256:')}.json"  # type: ignore[operator]
            if (
                resolved_batch_authority is not None
                and not rights_path.exists()
                and not receipt_path.exists()
                and not marker_path.exists()
            ):
                try:
                    derive_episode_interpretation_rights(
                        authority=resolved_batch_authority,
                        request=request,
                        interpreter=rights_interpreter,
                        output_path=rights_path,
                    )
                except Exception:
                    invalid_rights_inputs.add(
                        str(request.input_receipt["input_bundle_digest"])
                    )
            if rights_path.is_file() and not receipt_path.exists() and not marker_path.exists():
                try:
                    validate_episode_interpretation_rights(
                        rights_path=rights_path,
                        request=request,
                        interpreter=rights_interpreter,
                    )
                except Exception:
                    invalid_rights_inputs.add(
                        str(request.input_receipt["input_bundle_digest"])
                    )
                    continue
                eligible.append((request, rights_path))
        if not eligible:
            unavailable_reason = "rights_attestation_unavailable"
        try:
            if not eligible:
                raise ValueError("episode_interpretation_no_eligible_inputs")
            aggregate_request_digest = canonical_digest(
                {
                    "input_bundle_digests": [
                        request.input_receipt["input_bundle_digest"]
                        for request, _ in eligible
                    ],
                    "profile_digest": profile["profile_digest"],
                }
            )
            authorization_digest = canonical_digest(
                {"rights_digests": [_read(path).get("rights_digest") for _, path in eligible]}
            )
            audit = InferenceReservationAudit(
                run_root=interpretation_root,
                run_id=str(result.get("run_id") or root.name),
            )
            invoker = OpenAIAgentsSDKInvoker(
                OpenAIAgentsSDKConfig(
                    model=str(profile["model"]),
                    max_turns=1,
                    max_output_tokens=int(profile["max_output_tokens"]),
                    max_input_tokens=int(profile["max_input_tokens"]),
                    allow_live_invocation=True,
                    tracing_disabled=True,
                    max_inference_cost_usd=float(profile["max_cost_usd"]),
                )
            )
            invoker.configure_reservation_audit(
                record_reservation=audit.record_reservation,
                record_completion=audit.record_completion,
                restored_reserved_cost_usd=0.0,
            )
            interpreter = OpenAIMultimodalEpisodeInterpreter(
                invoker=invoker,
                model=str(profile["model"]),
                model_version=str(profile["model_version"]),
                max_frames=int(profile["max_frames"]),
                max_input_tokens=int(profile["max_input_tokens"]),
                max_output_tokens=int(profile["max_output_tokens"]),
                run_id=str(result.get("run_id") or root.name),
            )
            candidate_gate = build_openai_official_cost_run_gate(
                scope_attestation_path=env[
                    "BLUEPRINT_OPENAI_EPISODE_INTERPRETATION_COST_SCOPE_ATTESTATION_FILE"
                ],
                admin_api_key_file=env["OPENAI_ADMIN_API_KEY_FILE"],
                project_id=env["OPENAI_PROJECT_ID"],
                api_key_id=env[
                    "BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_API_KEY_ID"
                ],
                lane_id="policy_canary_episode_interpretation",
                run_id=str(result.get("run_id") or root.name),
                request_digest=aggregate_request_digest,
                candidate_digest=str(profile["profile_digest"]),
                authorization_receipt_digest=authorization_digest,
                max_cost_usd=float(profile["max_cost_usd"]),
                output_root=interpretation_root / "official_openai_cost",
                provider_id="openai",
                paid_resource_class="policy_canary_episode_interpretation",
                require_zero_baseline=False,
            )
            candidate_gate.reserve()
            cost_gate = candidate_gate
        except Exception:
            unavailable_reason = "official_cost_gate_unavailable"

    provider_call_count = 0
    provider_invocation_attempt_count = 0
    provider_invocation_error_type: str | None = None
    reused_count = 0
    for row, request, plan_path, receipt_path, marker_path in requests:
        input_digest = str(request.input_receipt["input_bundle_digest"])
        existing = _validated_receipt(receipt_path, input_bundle_digest=input_digest)
        if existing is not None:
            receipt = existing
            reused_count += 1
            historical_provider_call = int(receipt.get("provider_called") is True)
            provider_call_count += historical_provider_call
            provider_invocation_attempt_count += historical_provider_call
            if plan_path.is_file():
                artifacts.append(
                    _artifact(
                        plan_path,
                        root=evidence,
                        role="episode_interpretation_plan",
                    )
                )
            if marker_path.is_file():
                artifacts.append(
                    _artifact(
                        marker_path,
                        root=evidence,
                        role="episode_interpretation_attempt",
                    )
                )
        else:
            rights_path = (
                rights / f"{input_digest.removeprefix('sha256:')}.json"
                if rights is not None
                else None
            )
            reason = unavailable_reason
            if input_digest in invalid_rights_inputs:
                reason = "rights_attestation_invalid"
            if (
                identity is not None
                and (
                    identity.principal_kind != "independent_interpreter"
                    or identity.interpreter_id == request.candidate_policy_id
                    or identity.model == request.candidate_policy_id
                )
            ):
                reason = "candidate_policy_self_grading_forbidden"
            if marker_path.exists():
                reason = "prior_interpretation_execution_ambiguous"
            elif rights_path is None or not rights_path.is_file():
                reason = "rights_attestation_unavailable"
            plan = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "episode_id": request.episode_id,
                "candidate_policy_id": request.candidate_policy_id,
                "input_bundle_digest": input_digest,
                "interpreter": identity.__dict__ if identity is not None else None,
                "interpreter_identity_digest": (
                    canonical_digest(identity.__dict__) if identity is not None else None
                ),
                "interpreter_profile_digest": profile.get("profile_digest") if profile else None,
                "rights_attestation_digest": (
                    _validated_rights_digest(rights_path)
                ),
                "execution_status": "eligible" if reason is None else "abstain",
                "abstention_reason": reason,
                "score_or_ranking_effect": "none",
                "plan_digest": "",
            }
            plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
            if plan_path.exists():
                artifacts.append(
                    _artifact(
                        plan_path,
                        root=evidence,
                        role="episode_interpretation_plan",
                    )
                )
            else:
                _write_once(plan_path, plan)
                artifacts.append(
                    _artifact(
                        plan_path,
                        root=evidence,
                        role="episode_interpretation_plan",
                    )
                )
            if reason is not None:
                receipt = _abstain(request, reason=reason, path=receipt_path, identity=identity)
            else:
                marker = {
                    "schema_version": "policy_canary_episode_interpretation_attempt.v1",
                    "input_bundle_digest": input_digest,
                    "attempt_permitted_once": True,
                    "attempt_digest": "",
                }
                marker["attempt_digest"] = canonical_digest(marker, digest_field="attempt_digest")
                _write_once(marker_path, marker)
                artifacts.append(
                    _artifact(
                        marker_path,
                        root=evidence,
                        role="episode_interpretation_attempt",
                    )
                )
                try:
                    if selected_runner is not None:
                        receipt = dict(
                            selected_runner(
                                request=request,
                                rights_attestation_path=rights_path,  # type: ignore[arg-type]
                                output_path=receipt_path,
                            )
                        )
                    else:
                        provider_invocation_attempt_count += 1
                        receipt = interpret_episode(
                            request=request,
                            interpreter=interpreter,  # type: ignore[arg-type]
                            rights_attestation_path=rights_path,
                            output_path=receipt_path,
                        )
                    provider_call_count += int(receipt.get("provider_called") is True)
                except Exception as exc:
                    if selected_runner is None:
                        provider_invocation_error_type = type(exc).__name__
                    receipt = _abstain(
                        request,
                        reason="interpreter_execution_unavailable",
                        path=receipt_path,
                        identity=identity,
                    )
        receipt_record = _artifact(
            receipt_path, root=evidence, role="episode_interpretation_receipt"
        )
        artifacts.append(receipt_record)
        row.setdefault("evidence_artifacts", {})["episode_interpretation"] = receipt_record
        receipts.append(
            {
                "episode_id": request.episode_id,
                "status": receipt["status"],
                "abstention_reason": receipt.get("abstention_reason"),
                "input_bundle_digest": input_digest,
                "receipt_digest": receipt["receipt_digest"],
                "deterministic_agreement": receipt["deterministic_agreement"],
                "receipt": receipt_record,
            }
        )

    if cost_gate is not None:
        try:
            cost_gate.complete(
                provider_call_performed=provider_invocation_attempt_count > 0,
                runtime_result_digest=canonical_digest(
                    {"receipt_digests": [row.get("receipt_digest") for row in receipts]}
                ),
                runtime_exception_type=provider_invocation_error_type,
            )
        except Exception as exc:
            cost_completion_error = type(exc).__name__
    if audit is not None:
        try:
            audit.write_manifest()
        except Exception:
            pass
    for path in sorted((interpretation_root / "official_openai_cost").glob("*.json")):
        artifacts.append(
            _artifact(path, root=evidence, role="episode_interpretation_cost_evidence")
        )
    for path in sorted((interpretation_root / "inference_reservations").rglob("*.json")):
        artifacts.append(
            _artifact(path, root=evidence, role="episode_interpretation_inference_budget")
        )
    inventory = result.setdefault("artifact_inventory", [])
    known = {str(row.get("relative_path")) for row in inventory if isinstance(row, Mapping)}
    inventory.extend(row for row in artifacts if row["relative_path"] not in known)
    completed = sum(row.get("status") == "completed" for row in receipts)
    abstained = sum(row.get("status") == "abstained" for row in receipts)
    disagreed = sum(row.get("deterministic_agreement") == "disagrees" for row in receipts)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if completed and not abstained else "abstained" if abstained == len(receipts) else "partial",
        "episode_count": len(episodes),
        "receipt_count": sum(row.get("receipt") is not None for row in receipts),
        "completed_count": completed,
        "abstained_count": abstained,
        "disagreement_count": disagreed,
        "reused_receipt_count": reused_count,
        "provider_call_count": provider_call_count,
        "provider_invocation_attempt_count": provider_invocation_attempt_count,
        "input_bundle_unavailable_count": sum(
            row.get("input_bundle_digest") is None for row in receipts
        ),
        "interpreter": identity.__dict__ if identity is not None else None,
        "interpreter_profile_digest": profile.get("profile_digest") if profile else None,
        "official_cost_completion_error_type": cost_completion_error,
        "authoritative_deterministic_result_unchanged": True,
        "score_overwrite_performed": False,
        "ranking_or_promotion_effect": "none",
        "receipts": receipts,
        "summary_digest": "",
    }
    summary["summary_digest"] = canonical_digest(summary, digest_field="summary_digest")
    result["episode_interpretation"] = summary
    result["artifact_inventory_digest"] = canonical_digest(
        {"value": result["artifact_inventory"]}
    )
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def best_effort_policy_canary_episode_interpretations(
    *,
    run_root: str | Path,
    evidence_root: str | Path,
    session_result: Mapping[str, Any],
    runner: EpisodeInterpretationRunner | None = None,
    rights_root: str | Path | None = None,
    batch_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Never let the optional interpretation layer block terminal closeout."""

    try:
        return materialize_policy_canary_episode_interpretations(
            run_root=run_root,
            evidence_root=evidence_root,
            session_result=session_result,
            runner=runner,
            rights_root=rights_root,
            batch_authority=batch_authority,
        )
    except Exception as exc:
        result = json.loads(json.dumps(dict(session_result), allow_nan=False))
        count = len(result.get("episodes") or [])
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "abstained",
            "episode_count": count,
            "receipt_count": 0,
            "completed_count": 0,
            "abstained_count": count,
            "disagreement_count": 0,
            "reused_receipt_count": 0,
            "provider_call_count": 0,
            "provider_invocation_attempt_count": 0,
            "input_bundle_unavailable_count": count,
            "interpreter": None,
            "interpreter_profile_digest": None,
            "official_cost_completion_error_type": None,
            "closeout_error_type": type(exc).__name__,
            "authoritative_deterministic_result_unchanged": True,
            "score_overwrite_performed": False,
            "ranking_or_promotion_effect": "none",
            "receipts": [],
            "summary_digest": "",
        }
        summary["summary_digest"] = canonical_digest(
            summary, digest_field="summary_digest"
        )
        result["episode_interpretation"] = summary
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        return result


def main() -> int:
    """Run one digest-bound closeout/backfill without mutating source evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--session-result", type=Path, required=True)
    parser.add_argument("--rights-root", type=Path)
    parser.add_argument("--batch-authority", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_policy_canary_episode_interpretations(
        run_root=args.run_root,
        evidence_root=args.evidence_root,
        session_result=_read(args.session_result.expanduser().resolve()),
        rights_root=args.rights_root,
        batch_authority=(
            _read(args.batch_authority.expanduser().resolve())
            if args.batch_authority is not None
            else None
        ),
    )
    _write_once(args.output.expanduser().resolve(), result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BATCH_AUTHORITY_ENV",
    "EpisodeInterpretationRunner",
    "PLAN_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "best_effort_policy_canary_episode_interpretations",
    "materialize_policy_canary_episode_interpretations",
]
