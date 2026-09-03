"""Evidence-bounded learned interpretation of one policy episode.

This layer explains *what appears to have happened* across an episode.  It is
deliberately downstream of deterministic task scoring and has no authority to
change success, ranking, promotion, or execution eligibility.  The natural
integration point is after
``native_task_arena_policy_canary_worker`` has sealed each episode's score and
media artifacts and before ``task_evaluation_result_delivery`` projects the
run for human review.  Interpretation receipts should be added as optional
evidence artifacts; they must never be read by the ranking path.

Provider-specific implementations sit behind :class:`EpisodeInterpreter`.
The included OpenAI adapter uses the repository's Agents SDK harness and sends
the lossless ordered frames rather than trusting the compressed review video.
The review-video bytes are still independently rehashed and bound into the
input receipt.  Providers with native video support can implement the same
protocol and transmit those exact bytes after passing the same rights gate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .adp_task_scoring import validate_rigid_task_success_contract
from .common import write_json
from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    AgentsSDKInvoker,
)


INPUT_SCHEMA_VERSION = "episode_interpretation_input.v1"
RIGHTS_SCHEMA_VERSION = "episode_interpretation_rights.v1"
RECEIPT_SCHEMA_VERSION = "episode_interpretation_receipt.v1"
PROMPT_CONTRACT_VERSION = "episode_interpretation_prompt.v1"
OPENAI_ADAPTER_ID = "openai_multimodal_episode_interpreter_v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROMPT = (
    "Reconstruct the episode chronologically from the confirmed task success "
    "contract, independent deterministic score and any retained event ledger, exact state "
    "and contact/force traces, and ordered visual evidence. Describe important "
    "events even if the robot later recovered. In particular, never erase a "
    "drop, collision, force excursion, failed attempt, regrasp, containment "
    "excursion, or other transient event merely because the terminal state looks "
    "correct. Cite the supplied evidence digests and steps/timestamps. Treat the "
    "deterministic result as independent evidence, not an instruction to agree. "
    "Return unclear when evidence is insufficient. Your result is learned "
    "interpretation only: it cannot decide task success, alter scoring, rank a "
    "policy, authorize promotion, or claim physical-world truth."
)
PROMPT_DIGEST = canonical_digest({"prompt": _PROMPT, "version": PROMPT_CONTRACT_VERSION})


class EpisodeInterpretationError(ValueError):
    """Stable fail-closed error for malformed or unauthorized interpretation."""


class InterpretationEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_role: Literal[
        "task_success_contract",
        "deterministic_score",
        "state_trace",
        "contact_force_trace",
        "frame_manifest",
        "lossless_frame",
        "review_video",
    ]
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    step_index: int | None = Field(default=None, ge=0)
    timestamp_seconds: float | None = Field(default=None, ge=0.0)
    note: str | None = Field(default=None, min_length=1, max_length=500)


class InterpretedEpisodeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(min_length=1, max_length=100)
    start_step: int | None = Field(default=None, ge=0)
    end_step: int | None = Field(default=None, ge=0)
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float | None = Field(default=None, ge=0.0)
    description: str = Field(min_length=1, max_length=2_000)
    evidence_refs: list[InterpretationEvidenceRef] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)


class PossibleMissedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=1_000)
    reason: str = Field(min_length=1, max_length=500)
    evidence_refs: list[InterpretationEvidenceRef] = Field(default_factory=list, max_length=20)


class EpisodeInterpreterOutput(BaseModel):
    """Provider output with no authoritative success or ranking field."""

    model_config = ConfigDict(extra="forbid")

    episode_outcome: Literal["appears_complete", "appears_incomplete", "unclear"]
    summary: str = Field(min_length=1, max_length=8_000)
    events: list[InterpretedEpisodeEvent] = Field(default_factory=list, max_length=200)
    possible_missed_events: list[PossibleMissedEvent] = Field(default_factory=list, max_length=100)
    contract_considerations: list[str] = Field(default_factory=list, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class InterpreterIdentity:
    interpreter_id: str
    principal_kind: Literal["independent_interpreter", "candidate_policy"]
    provider_id: str
    execution_site: Literal["local", "external_provider"]
    runtime: str
    model: str
    model_version: str


@dataclass(frozen=True)
class EpisodeInterpretationRequest:
    episode_id: str
    candidate_policy_id: str
    evidence_root: Path
    task_success_contract: Mapping[str, Any]
    deterministic_score: Mapping[str, Any]
    state_trace: Mapping[str, Any]
    contact_force_trace: Mapping[str, Any]
    frame_manifest: Mapping[str, Any]
    review_video_paths: tuple[Path, ...]
    ordered_frame_paths: tuple[Path, ...]
    input_receipt: Mapping[str, Any]


class EpisodeInterpreter(Protocol):
    @property
    def identity(self) -> InterpreterIdentity: ...

    def disclosed_artifact_roles(self, request: EpisodeInterpretationRequest) -> Sequence[str]: ...

    def interpret(self, request: EpisodeInterpretationRequest) -> EpisodeInterpreterOutput: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_mapping(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EpisodeInterpretationError(error) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise EpisodeInterpretationError(error)
    return dict(value)


def _inside(root: Path, candidate: Path, *, error: str) -> Path:
    resolved = candidate.expanduser().resolve()
    if root != resolved and root not in resolved.parents:
        raise EpisodeInterpretationError(error)
    if candidate.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise EpisodeInterpretationError(error)
    return resolved


def _validate_intrinsic_digest(value: Mapping[str, Any], *, digest_field: str, error: str) -> str:
    supplied = str(value.get(digest_field) or "")
    if not _SHA256.fullmatch(supplied) or supplied != canonical_digest(
        value, digest_field=digest_field
    ):
        raise EpisodeInterpretationError(error)
    return supplied


def _frame_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(manifest.get("policy_input_frames"), list):
        records.extend(row for row in manifest["policy_input_frames"] if isinstance(row, Mapping))
        terminal = manifest.get("terminal_observation")
        if isinstance(terminal, Mapping):
            records.append(terminal)
        return records
    observations: list[Mapping[str, Any]] = []
    for field in ("policy_input_observations", "review_observations"):
        raw = manifest.get(field)
        if isinstance(raw, list):
            observations.extend(row for row in raw if isinstance(row, Mapping))
    terminal = manifest.get("terminal_observation")
    if isinstance(terminal, Mapping):
        observations.append(terminal)
    observations.sort(key=lambda row: int(row.get("observation_index", -1)))
    for observation in observations:
        views = observation.get("views")
        if isinstance(views, Mapping):
            records.extend(row for _, row in sorted(views.items()) if isinstance(row, Mapping))
    return records


def _verified_frames(
    *, root: Path, manifest: Mapping[str, Any]
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    bindings: list[dict[str, Any]] = []
    for index, row in enumerate(_frame_records(manifest)):
        relative = str(row.get("relative_path") or row.get("path") or "").strip()
        expected = str(row.get("png_sha256") or row.get("sha256") or "")
        if not relative or not _SHA256.fullmatch(expected):
            raise EpisodeInterpretationError(
                f"episode_interpretation_frame_binding_invalid:{index}"
            )
        unresolved = Path(relative)
        path = _inside(
            root,
            unresolved if unresolved.is_absolute() else root / unresolved,
            error=f"episode_interpretation_frame_file_invalid:{index}",
        )
        if _file_sha256(path) != expected:
            raise EpisodeInterpretationError(
                f"episode_interpretation_frame_digest_mismatch:{index}"
            )
        size = row.get("size_bytes")
        if size is not None and size != path.stat().st_size:
            raise EpisodeInterpretationError(f"episode_interpretation_frame_size_mismatch:{index}")
        paths.append(path)
        bindings.append(
            {
                "frame_index": index,
                "source_frame_index": row.get("frame_index"),
                "camera_id": row.get("camera_id"),
                "kind": row.get("kind"),
                "simulation_time_s": row.get("simulation_time_s"),
                "timestamp_ns": row.get("timestamp_ns"),
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": expected,
                "size_bytes": path.stat().st_size,
            }
        )
    if not paths:
        raise EpisodeInterpretationError("episode_interpretation_frame_inventory_missing")
    return paths, bindings


def _artifact_record(path: Path, *, root: Path, logical_digest: str) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "logical_digest": logical_digest,
    }


def build_episode_interpretation_request(
    *,
    episode_id: str,
    candidate_policy_id: str,
    evidence_root: str | Path,
    task_success_contract_path: str | Path,
    deterministic_score_path: str | Path,
    state_trace_path: str | Path,
    contact_force_trace_path: str | Path,
    frame_manifest_path: str | Path,
    review_video_paths: Sequence[str | Path],
) -> EpisodeInterpretationRequest:
    """Rehash and bind every source artifact before any interpreter can run."""

    if not episode_id.strip() or not candidate_policy_id.strip():
        raise EpisodeInterpretationError("episode_interpretation_identity_missing")
    root = Path(evidence_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise EpisodeInterpretationError("episode_interpretation_evidence_root_invalid")

    def source(raw: str | Path, code: str) -> Path:
        unresolved = Path(raw).expanduser()
        return _inside(
            root,
            unresolved if unresolved.is_absolute() else root / unresolved,
            error=code,
        )

    contract_path = source(
        task_success_contract_path, "episode_interpretation_task_contract_file_invalid"
    )
    score_path = source(deterministic_score_path, "episode_interpretation_score_file_invalid")
    state_path = source(state_trace_path, "episode_interpretation_state_trace_file_invalid")
    contact_path = source(
        contact_force_trace_path, "episode_interpretation_contact_trace_file_invalid"
    )
    manifest_path = source(
        frame_manifest_path, "episode_interpretation_frame_manifest_file_invalid"
    )
    contract = _read_mapping(contract_path, error="episode_interpretation_task_contract_invalid")
    try:
        contract = validate_rigid_task_success_contract(contract, require_confirmed=True)
    except ValueError as exc:
        raise EpisodeInterpretationError(
            "episode_interpretation_task_contract_unconfirmed_or_invalid"
        ) from exc
    contract_digest = str(contract["contract_digest"])
    score = _read_mapping(score_path, error="episode_interpretation_score_invalid")
    score_digest = _validate_intrinsic_digest(
        score,
        digest_field="report_digest",
        error="episode_interpretation_score_digest_invalid",
    )
    embedded_contract = score.get("task_success_contract")
    event_ledger = score.get("event_ledger")
    current_provenance = (
        score.get("task_success_contract_digest") == contract_digest
        and isinstance(embedded_contract, Mapping)
        and embedded_contract.get("contract_digest") == contract_digest
        and isinstance(event_ledger, Mapping)
        and event_ledger.get("derived_only_from_episode_samples") is True
        and score.get("candidate_policy_queried_by_scorer") is False
    )
    retained_legacy_provenance = (
        score.get("schema_version") == "adp009d_task_scoring.v1"
        and score.get("judgement_source") == "deterministic_simulator_object_state"
        and score.get("rendered_image_consulted") is False
        and score.get("caller_asserted_success_accepted") is False
        and score.get("candidate_policy_queried") is False
        and score.get("failure_modes_fully_determined") is True
    )
    if score.get("learned_judge_consulted") is not False or not (
        current_provenance or retained_legacy_provenance
    ):
        raise EpisodeInterpretationError(
            "episode_interpretation_deterministic_score_provenance_invalid"
        )
    state = _read_mapping(state_path, error="episode_interpretation_state_trace_invalid")
    state_digest = _validate_intrinsic_digest(
        state,
        digest_field="trace_digest",
        error="episode_interpretation_state_trace_digest_invalid",
    )
    contact = _read_mapping(contact_path, error="episode_interpretation_contact_trace_invalid")
    contact_digest = _validate_intrinsic_digest(
        contact,
        digest_field="trace_digest",
        error="episode_interpretation_contact_trace_digest_invalid",
    )
    manifest = _read_mapping(manifest_path, error="episode_interpretation_frame_manifest_invalid")
    manifest_digest = _validate_intrinsic_digest(
        manifest,
        digest_field="frame_manifest_digest",
        error="episode_interpretation_frame_manifest_digest_invalid",
    )
    frames, frame_bindings = _verified_frames(root=root, manifest=manifest)
    videos: list[Path] = []
    video_bindings: list[dict[str, Any]] = []
    for index, raw in enumerate(review_video_paths):
        path = source(raw, f"episode_interpretation_review_video_invalid:{index}")
        if path.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv"}:
            raise EpisodeInterpretationError(
                f"episode_interpretation_review_video_type_invalid:{index}"
            )
        videos.append(path)
        video_bindings.append(
            {
                "video_index": index,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    artifacts = {
        "task_success_contract": _artifact_record(
            contract_path, root=root, logical_digest=contract_digest
        ),
        "deterministic_score": _artifact_record(score_path, root=root, logical_digest=score_digest),
        "state_trace": _artifact_record(state_path, root=root, logical_digest=state_digest),
        "contact_force_trace": _artifact_record(
            contact_path, root=root, logical_digest=contact_digest
        ),
        "frame_manifest": _artifact_record(
            manifest_path, root=root, logical_digest=manifest_digest
        ),
        "lossless_frames": frame_bindings,
        "review_videos": video_bindings,
    }
    receipt: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "episode_id": episode_id,
        "candidate_policy_id": candidate_policy_id,
        "task_success_contract_digest": contract_digest,
        "deterministic_score_digest": score_digest,
        "artifacts": artifacts,
        "review_video_count": len(videos),
        "lossless_frame_count": len(frames),
        "all_source_bytes_rehashed": True,
        "input_bundle_digest": "",
    }
    receipt["input_bundle_digest"] = canonical_digest(receipt, digest_field="input_bundle_digest")
    return EpisodeInterpretationRequest(
        episode_id=episode_id,
        candidate_policy_id=candidate_policy_id,
        evidence_root=root,
        task_success_contract=contract,
        deterministic_score=score,
        state_trace=state,
        contact_force_trace=contact,
        frame_manifest=manifest,
        review_video_paths=tuple(videos),
        ordered_frame_paths=tuple(frames),
        input_receipt=receipt,
    )


def materialize_episode_interpretation_rights(
    *,
    episode_id: str,
    input_bundle_digest: str,
    identity: InterpreterIdentity,
    allowed_artifact_roles: Sequence[str],
    external_disclosure_authorized: bool,
    accepted_by: str,
    accepted_on: str,
    authority_reference: str,
    source_rights_admission_digest: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal human authority for one exact interpreter input bundle."""

    roles = sorted(set(str(role) for role in allowed_artifact_roles))
    if (
        not episode_id.strip()
        or not _SHA256.fullmatch(input_bundle_digest)
        or not _SHA256.fullmatch(source_rights_admission_digest)
        or not roles
        or not accepted_by.strip()
        or not accepted_on.strip()
        or not authority_reference.strip()
        or identity.principal_kind != "independent_interpreter"
    ):
        raise EpisodeInterpretationError("episode_interpretation_rights_request_invalid")
    value: dict[str, Any] = {
        "schema_version": RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_episode_interpretation",
        "episode_id": episode_id,
        "input_bundle_digest": input_bundle_digest,
        "interpreter": identity.__dict__,
        "allowed_artifact_roles": roles,
        "external_disclosure_authorized": external_disclosure_authorized,
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "source_rights_admission_digest": source_rights_admission_digest,
        "accepted_by": accepted_by.strip(),
        "accepted_on": accepted_on.strip(),
        "authority_reference": authority_reference.strip(),
        "learned_interpretation_only": True,
        "rights_digest": "",
    }
    value["rights_digest"] = canonical_digest(value, digest_field="rights_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise EpisodeInterpretationError("episode_interpretation_rights_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return value


def _validate_rights(
    *,
    rights_path: str | Path,
    request: EpisodeInterpretationRequest,
    interpreter: EpisodeInterpreter,
) -> tuple[Path, dict[str, Any]]:
    path = Path(rights_path).expanduser().resolve()
    rights = _read_mapping(path, error="episode_interpretation_rights_invalid")
    identity = interpreter.identity
    disclosed = sorted(set(interpreter.disclosed_artifact_roles(request)))
    allowed = rights.get("allowed_artifact_roles")
    external = identity.execution_site == "external_provider"
    if (
        rights.get("schema_version") != RIGHTS_SCHEMA_VERSION
        or rights.get("status") != "accepted_for_episode_interpretation"
        or rights.get("episode_id") != request.episode_id
        or rights.get("input_bundle_digest") != request.input_receipt["input_bundle_digest"]
        or rights.get("interpreter") != identity.__dict__
        or not isinstance(allowed, list)
        or any(role not in allowed for role in disclosed)
        or rights.get("external_disclosure_authorized") is not external
        or rights.get("provider_training_authorized") is not False
        or rights.get("public_redistribution_authorized") is not False
        or not _SHA256.fullmatch(str(rights.get("source_rights_admission_digest") or ""))
        or not str(rights.get("accepted_by") or "").strip()
        or rights.get("learned_interpretation_only") is not True
        or rights.get("rights_digest") != canonical_digest(rights, digest_field="rights_digest")
    ):
        raise EpisodeInterpretationError("episode_interpretation_rights_invalid")
    return path, rights


def validate_episode_interpretation_rights(
    *,
    rights_path: str | Path,
    request: EpisodeInterpretationRequest,
    interpreter: EpisodeInterpreter,
) -> dict[str, Any]:
    """Public preflight used by closeout before reserving provider spend."""

    _, rights = _validate_rights(
        rights_path=rights_path,
        request=request,
        interpreter=interpreter,
    )
    return rights


def _evidence_digest_inventory(request: EpisodeInterpretationRequest) -> dict[str, set[str]]:
    artifacts = request.input_receipt["artifacts"]
    inventory = {
        role: {str(artifacts[role]["logical_digest"])}
        for role in (
            "task_success_contract",
            "deterministic_score",
            "state_trace",
            "contact_force_trace",
            "frame_manifest",
        )
    }
    inventory["lossless_frame"] = {str(row["sha256"]) for row in artifacts["lossless_frames"]}
    inventory["review_video"] = {str(row["sha256"]) for row in artifacts["review_videos"]}
    return inventory


def _validate_output_evidence(
    output: EpisodeInterpreterOutput, request: EpisodeInterpretationRequest
) -> None:
    inventory = _evidence_digest_inventory(request)
    if output.episode_outcome != "unclear" and not output.events:
        raise EpisodeInterpretationError("episode_interpretation_output_event_narrative_missing")
    refs = [ref for event in output.events for ref in event.evidence_refs]
    refs.extend(ref for missed in output.possible_missed_events for ref in missed.evidence_refs)
    for ref in refs:
        if ref.artifact_digest not in inventory[ref.artifact_role]:
            raise EpisodeInterpretationError(
                "episode_interpretation_output_evidence_reference_invalid"
            )
    for event in output.events:
        if (
            event.start_step is not None
            and event.end_step is not None
            and event.end_step < event.start_step
        ) or (
            event.start_time_seconds is not None
            and event.end_time_seconds is not None
            and event.end_time_seconds < event.start_time_seconds
        ):
            raise EpisodeInterpretationError("episode_interpretation_output_event_interval_invalid")


def _agreement(
    *, output: EpisodeInterpreterOutput, deterministic_score: Mapping[str, Any]
) -> Literal["agrees", "disagrees", "abstains"]:
    if output.episode_outcome == "unclear":
        return "abstains"
    deterministic = deterministic_score.get("task_succeeded")
    if deterministic_score.get("status") != "scored" or not isinstance(deterministic, bool):
        return "abstains"
    interpreted = output.episode_outcome == "appears_complete"
    return "agrees" if interpreted is deterministic else "disagrees"


def _abstention_output(
    request: EpisodeInterpretationRequest, *, reason: str
) -> EpisodeInterpreterOutput:
    manifest_digest = request.input_receipt["artifacts"]["frame_manifest"]["logical_digest"]
    return EpisodeInterpreterOutput(
        episode_outcome="unclear",
        summary=f"Episode interpretation abstained: {reason}.",
        events=[],
        possible_missed_events=[
            PossibleMissedEvent(
                description="One or more visually observable transient events may be absent.",
                reason=reason,
                evidence_refs=[
                    InterpretationEvidenceRef(
                        artifact_role="frame_manifest",
                        artifact_digest=manifest_digest,
                        note="The optional learned interpretation lane did not run.",
                    )
                ],
            )
        ],
        contract_considerations=[],
        confidence=0.0,
    )


def materialize_episode_interpretation_abstention(
    *,
    request: EpisodeInterpretationRequest,
    reason: str,
    output_path: str | Path,
    interpreter_identity: InterpreterIdentity | None = None,
) -> dict[str, Any]:
    """Seal a typed, idempotency-friendly abstention without provider inference."""

    if not re.fullmatch(r"[a-z][a-z0-9_]{2,127}", reason):
        raise EpisodeInterpretationError("episode_interpretation_abstention_reason_invalid")
    identity = interpreter_identity or InterpreterIdentity(
        interpreter_id="unavailable_independent_episode_interpreter",
        principal_kind="independent_interpreter",
        provider_id="unavailable",
        execution_site="local",
        runtime="not_invoked",
        model="not_configured",
        model_version="not_configured",
    )
    if (
        identity.principal_kind != "independent_interpreter"
        or identity.interpreter_id == request.candidate_policy_id
        or identity.model == request.candidate_policy_id
    ):
        raise EpisodeInterpretationError("candidate_policy_self_grading_forbidden")
    destination = Path(output_path).expanduser().resolve()
    output = _abstention_output(request, reason=reason)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "abstained",
        "abstention_reason": reason,
        "episode_id": request.episode_id,
        "candidate_policy_id": request.candidate_policy_id,
        "input_bundle_digest": request.input_receipt["input_bundle_digest"],
        "input_receipt": dict(request.input_receipt),
        "interpreter": identity.__dict__,
        "interpreter_identity_digest": canonical_digest(identity.__dict__),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_digest": PROMPT_DIGEST,
        "rights_attestation": None,
        "provider_called": False,
        "learned_interpretation": output.model_dump(mode="json"),
        "deterministic_agreement": "abstains",
        "authoritative_deterministic_result": {
            "score_digest": request.input_receipt["deterministic_score_digest"],
            "status": request.deterministic_score.get("status"),
            "task_succeeded": request.deterministic_score.get("task_succeeded"),
            "outcome": request.deterministic_score.get("outcome"),
            "failed_criteria": request.deterministic_score.get("failed_criteria"),
        },
        "proof_boundary": {
            "learned_interpretation_only": True,
            "authoritative_task_success_unchanged": True,
            "deterministic_score_overwrite_forbidden": True,
            "candidate_policy_self_grading_forbidden": True,
            "ranking_or_promotion_effect": "none",
            "review_video_is_derived_visual_evidence": True,
            "simulation_is_not_physical_truth": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = _read_mapping(
            destination, error="episode_interpretation_existing_receipt_invalid"
        )
        if existing != receipt:
            raise EpisodeInterpretationError("episode_interpretation_output_conflict")
        return existing
    write_json(destination, receipt)
    return receipt


def interpret_episode(
    *,
    request: EpisodeInterpretationRequest,
    interpreter: EpisodeInterpreter,
    rights_attestation_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run one independent interpreter and seal a non-authoritative receipt."""

    identity = interpreter.identity
    if (
        identity.principal_kind != "independent_interpreter"
        or identity.interpreter_id == request.candidate_policy_id
        or identity.model == request.candidate_policy_id
    ):
        raise EpisodeInterpretationError("candidate_policy_self_grading_forbidden")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise EpisodeInterpretationError("episode_interpretation_output_exists")

    rights: dict[str, Any] | None = None
    rights_record: dict[str, Any] | None = None
    provider_called = False
    if not request.review_video_paths:
        output = _abstention_output(request, reason="required_review_video_missing")
        status = "abstained"
    else:
        if rights_attestation_path is None:
            raise EpisodeInterpretationError("episode_interpretation_rights_missing")
        rights_path, rights = _validate_rights(
            rights_path=rights_attestation_path,
            request=request,
            interpreter=interpreter,
        )
        rights_record = {
            "sha256": _file_sha256(rights_path),
            "rights_digest": rights["rights_digest"],
        }
        output = interpreter.interpret(request)
        if not isinstance(output, EpisodeInterpreterOutput):
            output = EpisodeInterpreterOutput.model_validate(output)
        _validate_output_evidence(output, request)
        provider_called = identity.execution_site == "external_provider"
        status = "abstained" if output.episode_outcome == "unclear" else "completed"

    agreement = _agreement(output=output, deterministic_score=request.deterministic_score)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status,
        "episode_id": request.episode_id,
        "candidate_policy_id": request.candidate_policy_id,
        "input_bundle_digest": request.input_receipt["input_bundle_digest"],
        "input_receipt": dict(request.input_receipt),
        "interpreter": identity.__dict__,
        "interpreter_identity_digest": canonical_digest(identity.__dict__),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_digest": PROMPT_DIGEST,
        "rights_attestation": rights_record,
        "provider_called": provider_called,
        "learned_interpretation": output.model_dump(mode="json"),
        "deterministic_agreement": agreement,
        "authoritative_deterministic_result": {
            "score_digest": request.input_receipt["deterministic_score_digest"],
            "status": request.deterministic_score.get("status"),
            "task_succeeded": request.deterministic_score.get("task_succeeded"),
            "outcome": request.deterministic_score.get("outcome"),
            "failed_criteria": request.deterministic_score.get("failed_criteria"),
        },
        "proof_boundary": {
            "learned_interpretation_only": True,
            "authoritative_task_success_unchanged": True,
            "deterministic_score_overwrite_forbidden": True,
            "candidate_policy_self_grading_forbidden": True,
            "ranking_or_promotion_effect": "none",
            "review_video_is_derived_visual_evidence": True,
            "simulation_is_not_physical_truth": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, receipt)
    return receipt


class DeterministicFixtureInterpreter:
    """Hermetic adapter for contract tests; it performs no learned inference."""

    def __init__(
        self,
        output: EpisodeInterpreterOutput | Mapping[str, Any],
        *,
        interpreter_id: str = "deterministic_fixture_episode_interpreter",
        principal_kind: Literal[
            "independent_interpreter", "candidate_policy"
        ] = "independent_interpreter",
    ) -> None:
        self._output = EpisodeInterpreterOutput.model_validate(output)
        self._identity = InterpreterIdentity(
            interpreter_id=interpreter_id,
            principal_kind=principal_kind,
            provider_id="local",
            execution_site="local",
            runtime="hermetic_fixture",
            model="deterministic_fixture",
            model_version="v1",
        )
        self.call_count = 0

    @property
    def identity(self) -> InterpreterIdentity:
        return self._identity

    def disclosed_artifact_roles(self, request: EpisodeInterpretationRequest) -> Sequence[str]:
        del request
        return (
            "task_success_contract",
            "deterministic_score",
            "state_trace",
            "contact_force_trace",
            "frame_manifest",
            "lossless_frame",
            "review_video",
        )

    def interpret(self, request: EpisodeInterpretationRequest) -> EpisodeInterpreterOutput:
        del request
        self.call_count += 1
        return self._output.model_copy(deep=True)


class OpenAIMultimodalEpisodeInterpreter:
    """OpenAI Agents SDK adapter over lossless ordered episode frames.

    Direct video transport is not assumed by this repository's SDK seam.  The
    adapter instead sends the ordered lossless source frames from which the
    digest-bound review video was derived.  If a frame cap causes sampling, a
    mandatory possible-missed-event note is appended to the model output.
    """

    def __init__(
        self,
        *,
        invoker: AgentsSDKInvoker,
        model: str,
        model_version: str,
        max_frames: int = 12,
        max_input_tokens: int = 240_000,
        max_output_tokens: int = 8_000,
    ) -> None:
        if not model.strip() or not model_version.strip() or max_frames < 2:
            raise EpisodeInterpretationError("openai_episode_interpreter_config_invalid")
        self._invoker = invoker
        self._max_frames = max_frames
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._identity = InterpreterIdentity(
            interpreter_id=OPENAI_ADAPTER_ID,
            principal_kind="independent_interpreter",
            provider_id="openai",
            execution_site="external_provider",
            runtime="openai_agents_sdk",
            model=model,
            model_version=model_version,
        )

    @property
    def identity(self) -> InterpreterIdentity:
        return self._identity

    def disclosed_artifact_roles(self, request: EpisodeInterpretationRequest) -> Sequence[str]:
        del request
        return (
            "task_success_contract",
            "deterministic_score",
            "state_trace",
            "contact_force_trace",
            "frame_manifest",
            "lossless_frame",
        )

    @staticmethod
    def _selected_trace_rows(
        value: Any,
        *,
        event_steps: set[int],
        maximum_rows: int = 96,
    ) -> list[Mapping[str, Any]]:
        rows = [row for row in value or [] if isinstance(row, Mapping)]
        if len(rows) <= maximum_rows:
            return rows
        step_positions = [
            (index, int(row["step_index"]))
            for index, row in enumerate(rows)
            if isinstance(row.get("step_index"), int)
            and not isinstance(row.get("step_index"), bool)
        ]
        event_positions = sorted(
            {
                min(step_positions, key=lambda item: abs(item[1] - step))[0]
                for step in event_steps
            }
        )
        available = maximum_rows - 2
        if len(event_positions) > available:
            event_positions = [
                event_positions[
                    round(index * (len(event_positions) - 1) / (available - 1))
                ]
                for index in range(available)
            ]
        chosen = {0, len(rows) - 1, *event_positions}
        for slot in range(maximum_rows):
            if len(chosen) >= maximum_rows:
                break
            chosen.add(round(slot * (len(rows) - 1) / (maximum_rows - 1)))
        return [rows[index] for index in sorted(chosen)[:maximum_rows]]

    @classmethod
    def _compact_state_trace(
        cls, value: Mapping[str, Any], *, event_steps: set[int]
    ) -> dict[str, Any]:
        task_fields = (
            "step_index",
            "task_object_pose_world",
            "task_scoring_pose_world",
            "grasp_frame_position_world_m",
            "gripper_width_m",
            "finger_separation_m",
            "task_contact_active",
            "support_contact_active",
            "containment_violation",
            "locked_joint_containment_violation",
            "robot_collision_failure",
            "scene_collision_failure",
            "forbidden_robot_task_collision_failure",
            "task_robot_contact_peak_force_n",
            "task_scene_collision_peak_force_n",
            "task_support_contact_peak_force_n",
            "robot_scene_contact_peak_force_n",
            "robot_task_forbidden_collision_peak_force_n",
        )
        joint_fields = ("step_index", "joint_positions_rad")
        task_rows = cls._selected_trace_rows(
            value.get("task_state_samples"), event_steps=event_steps
        )
        joint_rows = cls._selected_trace_rows(
            value.get("joint_states"), event_steps=event_steps
        )
        return {
            "schema_version": value.get("schema_version"),
            "trace_digest": value.get("trace_digest"),
            "task_state_samples": [
                {key: row[key] for key in task_fields if key in row}
                for row in task_rows
            ],
            "joint_states": [
                {key: row[key] for key in joint_fields if key in row}
                for row in joint_rows
            ],
            "sampling": {
                "source_task_state_sample_count": len(
                    value.get("task_state_samples") or []
                ),
                "selected_task_state_sample_count": len(task_rows),
                "source_joint_state_count": len(value.get("joint_states") or []),
                "selected_joint_state_count": len(joint_rows),
                "event_steps_preserved_when_capacity_allows": sorted(event_steps),
            },
        }

    @classmethod
    def _compact_contact_trace(
        cls, value: Mapping[str, Any], *, event_steps: set[int]
    ) -> dict[str, Any]:
        fields = (
            "step_index",
            "gripper_width_m",
            "task_contact_active",
            "robot_collision_failure",
            "scene_collision_failure",
            "task_robot_contact_peak_force_n",
            "task_scene_collision_peak_force_n",
            "task_support_contact_peak_force_n",
        )
        rows = cls._selected_trace_rows(value.get("samples"), event_steps=event_steps)
        return {
            "schema_version": value.get("schema_version"),
            "trace_digest": value.get("trace_digest"),
            "typed_gap": value.get("typed_gap"),
            "samples": [
                {key: row[key] for key in fields if key in row}
                for row in rows
            ],
            "sampling": {
                "source_sample_count": len(value.get("samples") or []),
                "selected_sample_count": len(rows),
                "event_steps_preserved_when_capacity_allows": sorted(event_steps),
            },
        }

    @staticmethod
    def _compact_frame_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
        observations = []
        for field in ("policy_input_observations", "review_observations"):
            for row in value.get(field) or []:
                if not isinstance(row, Mapping):
                    continue
                observations.append(
                    {
                        "kind": row.get("kind"),
                        "observation_index": row.get("observation_index"),
                        "simulation_time_s": row.get("simulation_time_s"),
                        "timestamp_ns": row.get("timestamp_ns"),
                        "camera_ids": row.get("camera_ids"),
                    }
                )
        terminal = value.get("terminal_observation")
        if isinstance(terminal, Mapping):
            observations.append(
                {
                    "kind": terminal.get("kind"),
                    "observation_index": terminal.get("observation_index"),
                    "simulation_time_s": terminal.get("simulation_time_s"),
                    "timestamp_ns": terminal.get("timestamp_ns"),
                    "camera_ids": terminal.get("camera_ids"),
                }
            )
        return {
            "schema_version": value.get("schema_version"),
            "frame_manifest_digest": value.get("frame_manifest_digest"),
            "episode_id": value.get("episode_id"),
            "required_camera_ids": value.get("required_camera_ids"),
            "review_only_camera_ids": value.get("review_only_camera_ids"),
            "policy_input_observation_count": value.get(
                "policy_input_observation_count"
            ),
            "review_observation_count": value.get("review_observation_count"),
            "observations": observations,
        }

    @staticmethod
    def _event_steps(score: Mapping[str, Any]) -> set[int]:
        ledger = score.get("event_ledger")
        if not isinstance(ledger, Mapping):
            return set()
        steps: set[int] = set()
        for key, value in ledger.items():
            if key.endswith("_steps") and isinstance(value, list):
                steps.update(
                    item
                    for item in value
                    if isinstance(item, int) and not isinstance(item, bool) and item >= 0
                )
            if key not in {"drop_events", "safety_events"} or not isinstance(
                value, list
            ):
                continue
            for event in value:
                if not isinstance(event, Mapping):
                    continue
                for field in ("step_index", "step", "start_step", "end_step"):
                    item = event.get(field)
                    if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                        steps.add(item)
        return steps

    def _selected_frame_indices(
        self,
        request: EpisodeInterpretationRequest,
        frame_rows: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        count = len(frame_rows)
        if count <= self._max_frames:
            return list(range(count))
        groups: dict[float, list[int]] = {}
        for index, row in enumerate(frame_rows):
            time_value = row.get("simulation_time_s")
            if isinstance(time_value, (int, float)) and not isinstance(time_value, bool):
                groups.setdefault(float(time_value), []).append(index)
        if len(groups) < 2:
            return sorted(
                {
                    round(index * (count - 1) / (self._max_frames - 1))
                    for index in range(self._max_frames)
                }
            )
        times = sorted(groups)
        group_width = max(len(indices) for indices in groups.values())
        group_limit = max(2, self._max_frames // group_width)
        event_positions: list[int] = []
        samples = request.state_trace.get("task_state_samples") or []
        step_values = [
            row.get("step_index")
            for row in samples
            if isinstance(row, Mapping)
            and isinstance(row.get("step_index"), int)
            and not isinstance(row.get("step_index"), bool)
        ]
        if step_values and max(step_values) > 0:
            for step in self._event_steps(request.deterministic_score):
                target_time = times[-1] * step / max(step_values)
                position = min(
                    range(len(times)), key=lambda index: abs(times[index] - target_time)
                )
                if position not in event_positions:
                    event_positions.append(position)
        mandatory = [0, len(times) - 1, *event_positions]
        chosen_positions: set[int] = set()
        for position in mandatory:
            if len(chosen_positions) >= group_limit:
                break
            chosen_positions.add(position)
        for slot in range(group_limit):
            if len(chosen_positions) >= group_limit:
                break
            chosen_positions.add(round(slot * (len(times) - 1) / (group_limit - 1)))
        chosen = []
        for position in sorted(chosen_positions):
            chosen.extend(groups[times[position]])
        return sorted(chosen[: self._max_frames])

    def interpret(self, request: EpisodeInterpretationRequest) -> EpisodeInterpreterOutput:
        event_steps = self._event_steps(request.deterministic_score)
        compact = {
            "input_bundle_digest": request.input_receipt["input_bundle_digest"],
            "task_success_contract": request.task_success_contract,
            "deterministic_score": request.deterministic_score,
            "state_trace": self._compact_state_trace(
                request.state_trace, event_steps=event_steps
            ),
            "contact_force_trace": self._compact_contact_trace(
                request.contact_force_trace, event_steps=event_steps
            ),
            "frame_manifest": self._compact_frame_manifest(request.frame_manifest),
            "review_video_bindings": request.input_receipt["artifacts"]["review_videos"],
        }
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": _PROMPT},
            {"type": "input_text", "text": canonical_json(compact)},
        ]
        frame_rows = request.input_receipt["artifacts"]["lossless_frames"]
        selected = self._selected_frame_indices(request, frame_rows)
        for index in selected:
            path = request.ordered_frame_paths[index]
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": canonical_json(frame_rows[index]),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:{mime};base64,"
                            + base64.b64encode(path.read_bytes()).decode("ascii")
                        ),
                        "detail": "low",
                    },
                ]
            )
        spec = AgentsSDKAgentSpec(
            run_id=f"episode-interpretation-{request.episode_id}",
            capability="episode_interpretation",
            name="Blueprint Independent Episode Interpreter",
            instructions=_PROMPT,
            model=self._identity.model,
            max_turns=1,
            max_output_tokens=self._max_output_tokens,
            max_input_tokens=self._max_input_tokens,
            output_type=EpisodeInterpreterOutput,
            stable_developer_prefix=_PROMPT,
            prompt_contract_version=PROMPT_CONTRACT_VERSION,
            dynamic_suffix_fields=("episode_evidence",),
        )
        invocation = self._invoker.invoke(spec, [{"role": "user", "content": content}])
        if invocation.provider != "openai" or invocation.model != self._identity.model:
            raise EpisodeInterpretationError("openai_episode_interpreter_provider_identity_invalid")
        output = EpisodeInterpreterOutput.model_validate(invocation.output)
        if len(selected) < len(request.ordered_frame_paths):
            output.possible_missed_events.append(
                PossibleMissedEvent(
                    description="A transient visual event may occur between sampled frames.",
                    reason=(
                        f"OpenAI adapter sampled {len(selected)} of "
                        f"{len(request.ordered_frame_paths)} lossless frames."
                    ),
                    evidence_refs=[
                        InterpretationEvidenceRef(
                            artifact_role="frame_manifest",
                            artifact_digest=request.input_receipt["artifacts"]["frame_manifest"][
                                "logical_digest"
                            ],
                        )
                    ],
                )
            )
        return output


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal a non-authoritative episode interpretation receipt."
    )
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--candidate-policy-id", required=True)
    parser.add_argument("--task-success-contract", required=True, type=Path)
    parser.add_argument("--deterministic-score", required=True, type=Path)
    parser.add_argument("--state-trace", required=True, type=Path)
    parser.add_argument("--contact-force-trace", required=True, type=Path)
    parser.add_argument("--frame-manifest", required=True, type=Path)
    parser.add_argument("--review-video", action="append", default=[], type=Path)
    parser.add_argument("--fixture-output", required=True, type=Path)
    parser.add_argument("--rights-attestation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    request = build_episode_interpretation_request(
        episode_id=args.episode_id,
        candidate_policy_id=args.candidate_policy_id,
        evidence_root=args.evidence_root,
        task_success_contract_path=args.task_success_contract,
        deterministic_score_path=args.deterministic_score,
        state_trace_path=args.state_trace,
        contact_force_trace_path=args.contact_force_trace,
        frame_manifest_path=args.frame_manifest,
        review_video_paths=args.review_video,
    )
    fixture = _read_mapping(
        args.fixture_output.expanduser().resolve(),
        error="episode_interpretation_fixture_output_invalid",
    )
    receipt = interpret_episode(
        request=request,
        interpreter=DeterministicFixtureInterpreter(fixture),
        rights_attestation_path=args.rights_attestation,
        output_path=args.output,
    )
    print(canonical_json(receipt))
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DeterministicFixtureInterpreter",
    "EpisodeInterpretationError",
    "EpisodeInterpretationRequest",
    "EpisodeInterpreter",
    "EpisodeInterpreterOutput",
    "InterpretedEpisodeEvent",
    "InterpreterIdentity",
    "OpenAIMultimodalEpisodeInterpreter",
    "PossibleMissedEvent",
    "PROMPT_DIGEST",
    "RECEIPT_SCHEMA_VERSION",
    "build_episode_interpretation_request",
    "interpret_episode",
    "main",
    "materialize_episode_interpretation_abstention",
    "materialize_episode_interpretation_rights",
    "validate_episode_interpretation_rights",
]
