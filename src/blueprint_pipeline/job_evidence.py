"""Bind a commercial job to the existing scientific control plane.

This is a small projection, not a replacement scientific contract. An intake
question can exist without capture or a run. Scientific work still needs the
real immutable testbed/request and the Pipeline-owned method catalog. This
surface can execute only the existing read-only local adapters; paid policy
execution keeps its existing configured-scene preparation and allocator.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import (
    DecisionEnvelope,
    DecisionEvidenceRequest,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    canonical_json,
    cross_runtime_canonical_digest,
)
from .local_evidence_adapters import authorized_local_evidence_adapter_registry
from .task_evaluation_method_catalog import (
    TaskEvaluationMethodCatalogError,
    load_task_evaluation_method_catalog,
)
from .task_evaluation_run_control_plane import (
    authorize_task_evaluation_run,
    execute_and_aggregate_task_evaluation_run,
    prepare_task_evaluation_run,
)


REQUEST_SCHEMA = "job_evidence_request.v1"
PUBLICATION_SCHEMA = "job_evidence_publication.v1"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_REQUEST_FIELDS = {
    "schema_version", "request_id", "job_id", "brief_digest", "question",
    "decision_evidence_request", "testbed", "capture_session_id", "intake_id",
    "requested_by", "idempotency_key", "request_digest", "approved_task_digest",
    "context_limitations",
    "testbed_canonical_json", "decision_request_canonical_json",
    "compiled_card_canonical_json",
}


class JobEvidenceError(ValueError):
    pass


def _id(value: Any, field: str) -> str:
    return strict_identifier(value, field=field, max_length=192)


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise JobEvidenceError(f"{field}:invalid_digest")
    return value


def canonical_artifact_body(value: Mapping[str, Any], digest_field: str) -> str:
    """Opaque producer bytes keep Python-native historical digests unchanged."""
    return canonical_json({key: child for key, child in value.items() if key != digest_field})


def _restore_artifact(value: Mapping[str, Any], body: Any, digest_field: str) -> dict[str, Any]:
    if not isinstance(body, str):
        raise JobEvidenceError("job_evidence_producer_bytes_missing")
    parsed = json.loads(body)
    if not isinstance(parsed, dict) or digest_field in parsed or canonical_json(parsed) != body:
        raise JobEvidenceError("job_evidence_producer_bytes_invalid")
    digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    restored = {**parsed, digest_field: digest}
    if value.get(digest_field) != digest or cross_runtime_canonical_digest(restored) != cross_runtime_canonical_digest(value):
        raise JobEvidenceError("job_evidence_producer_bytes_binding_mismatch")
    return restored


def _validate_card_bodies(testbed: Mapping[str, Any], bodies: Any) -> None:
    cards = testbed.get("compiled_cards") or {}
    values = [cards["site_card"]] if cards.get("site_card") else []
    for field in ("task_cards", "scenario_cards", "eval_cards"):
        values.extend(cards.get(field, []))
    expected = {row["card_digest"]: row for row in values}
    if not expected and bodies in (None, {}):
        return
    if not isinstance(bodies, Mapping) or set(bodies) != set(expected):
        raise JobEvidenceError("job_evidence_compiled_card_bytes_missing")
    for digest, value in expected.items():
        _restore_artifact(value, bodies[digest], "card_digest")


def validate_job_evidence_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = json.loads(json.dumps(dict(value), allow_nan=False))
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise JobEvidenceError("job_evidence_schema_invalid")
    if set(request) - _REQUEST_FIELDS:
        raise JobEvidenceError("job_evidence_unknown_fields")
    for field in ("request_id", "job_id", "idempotency_key"):
        _id(request.get(field), field)
    _digest(request.get("brief_digest"), "brief_digest")
    _digest(request.get("request_digest"), "request_digest")
    if request["request_digest"] != cross_runtime_canonical_digest(
        request, digest_field="request_digest"
    ):
        raise JobEvidenceError("job_evidence_request_digest_mismatch")
    question = request.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise JobEvidenceError("job_evidence_question_invalid")
    actor = request.get("requested_by")
    if not isinstance(actor, dict) or set(actor) != {"uid"}:
        raise JobEvidenceError("job_evidence_actor_invalid")
    _id(actor["uid"], "requested_by.uid")
    limitations = request.get("context_limitations", [])
    if not isinstance(limitations, list) or len(limitations) > 20 or any(
        not isinstance(row, str) or len(row) > 1000 for row in limitations
    ):
        raise JobEvidenceError("job_evidence_context_limitations_invalid")
    scientific = request.get("decision_evidence_request")
    testbed = request.get("testbed")
    if scientific is None:
        if any(request.get(field) is not None for field in (
            "testbed", "capture_session_id", "intake_id", "approved_task_digest",
            "testbed_canonical_json", "decision_request_canonical_json",
            "compiled_card_canonical_json",
        )):
            raise JobEvidenceError("job_evidence_partial_scientific_binding")
        return request
    if not isinstance(scientific, Mapping) or not isinstance(testbed, Mapping):
        raise JobEvidenceError("job_evidence_scientific_context_missing")
    native_request = DecisionEvidenceRequest.from_mapping(_restore_artifact(
        scientific, request.get("decision_request_canonical_json"), "request_digest"
    )).to_mapping()
    native_testbed = MaintainedSiteTaskTestbed.from_mapping(_restore_artifact(
        testbed, request.get("testbed_canonical_json"), "testbed_digest"
    )).to_mapping()
    _validate_card_bodies(native_testbed, request.get("compiled_card_canonical_json"))
    for field in ("capture_session_id", "intake_id"):
        _id(request.get(field), field)
    if question != native_request["decision_question"]:
        raise JobEvidenceError("job_evidence_question_binding_mismatch")
    if any(native_request[key] != native_testbed[target] for key, target in (
        ("testbed_id", "testbed_id"), ("testbed_version", "version"),
        ("testbed_digest", "testbed_digest"),
    )):
        raise JobEvidenceError("job_evidence_testbed_binding_mismatch")
    task_digest = native_testbed.get("approved_task_definition", {}).get("digest")
    if request.get("approved_task_digest") != task_digest:
        raise JobEvidenceError("job_evidence_approved_task_binding_mismatch")
    if not task_digest:
        raise JobEvidenceError("job_evidence_approved_task_missing")
    sources = native_testbed.get("source_capture_bundles", [])
    if len([row for row in sources if row.get("bundle_id") == request["intake_id"]]) != 1:
        raise JobEvidenceError("job_evidence_intake_binding_mismatch")
    request["decision_evidence_request"] = native_request
    request["testbed"] = native_testbed
    return request


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise JobEvidenceError("job_evidence_not_found")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise JobEvidenceError("job_evidence_artifact_invalid")
    return value


def _immutable(path: Path, value: Mapping[str, Any]) -> None:
    body = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(body)
    except FileExistsError:
        if path.read_bytes() != body:
            raise JobEvidenceError("job_evidence_immutable_conflict")


class JobEvidenceStore:
    def __init__(self, root: str | Path, run_state_root: str | Path):
        self.root = Path(root).resolve()
        self.run_state_root = Path(run_state_root).resolve()

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _root(self, request_id: str) -> Path:
        return self.root / "requests" / _id(request_id, "request_id")

    def _current(self, request_id: str) -> dict[str, Any]:
        paths = sorted((self._root(request_id) / "publications").glob("*.json"))
        if not paths:
            raise JobEvidenceError("job_evidence_not_found")
        return _read(paths[-1])

    def _publish(
        self, request: Mapping[str, Any], *, state: str, summary: str,
        next_action: str, limitations: list[str], preparation: Mapping[str, Any] | None = None,
        envelope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = self._root(request["request_id"])
        plan = dict(preparation["evidence_plan"]) if preparation else None
        publication = {
            "schema_version": PUBLICATION_SCHEMA,
            "request_id": request["request_id"], "job_id": request["job_id"],
            "brief_digest": request["brief_digest"], "request_digest": request["request_digest"],
            "question": request["question"], "state": state,
            "summary": summary, "next_action": next_action, "limitations": limitations,
            "run_id": preparation["run_id"] if preparation else None,
            "approved_task_digest": request.get("approved_task_digest"),
            "decision_request_digest": (request.get("decision_evidence_request") or {}).get("request_digest"),
            "testbed_digest": (request.get("testbed") or {}).get("testbed_digest"),
            "plan_digest": plan["plan_digest"] if plan else None,
            "evidence_plan": plan, "decision_envelope": dict(envelope) if envelope else None,
            "evidence_plan_canonical_json": canonical_artifact_body(plan, "plan_digest") if plan else None,
            "decision_envelope_canonical_json": canonical_artifact_body(envelope, "decision_envelope_digest") if envelope else None,
            "authorization_candidates": preparation.get("authorization_candidates", []) if preparation else [],
        }
        paths = sorted((root / "publications").glob("*.json"))
        if paths:
            previous = _read(paths[-1])
            if publication == {key: row for key, row in previous.items() if key not in {
                "sequence", "publication_digest"
            }}:
                return previous
        publication["sequence"] = len(paths) + 1
        publication["publication_digest"] = cross_runtime_canonical_digest(
            publication, digest_field="publication_digest"
        )
        _immutable(root / "publications" / f"{publication['sequence']:08d}.json", publication)
        return publication

    def submit(self, value: Mapping[str, Any]) -> dict[str, Any]:
        request = validate_job_evidence_request(value)
        with self._lock():
            root = self._root(request["request_id"])
            # A retry key cannot allocate a second run under another request ID.
            key = cross_runtime_canonical_digest({
                "job_id": request["job_id"], "idempotency_key": request["idempotency_key"]
            })[7:]
            _immutable(self.root / "keys" / f"{key}.json", {
                "request_id": request["request_id"], "request_digest": request["request_digest"]
            })
            _immutable(root / "request.json", request)
            if (root / "publications").is_dir():
                current = self._current(request["request_id"])
                if current["state"] != "unavailable":
                    return current
            limitations = list(request.get("context_limitations", []))
            if request.get("decision_evidence_request") is None:
                return self._publish(
                    request, state="information_required",
                    summary="The question is recorded; no evaluation has been created.",
                    next_action="Agree the decision-relevant task, candidate and acceptance criteria. Reuse permitted evidence where sufficient; collect only the missing observations needed for the question.",
                    limitations=[*limitations, "No scoped scientific request or qualified testbed is attached."],
                )
            try:
                catalog = load_task_evaluation_method_catalog()
            except TaskEvaluationMethodCatalogError as exc:
                return self._publish(
                    request, state="unavailable", summary="Evidence planning is unavailable.",
                    next_action="Restore the Pipeline method catalog for this task, then retry this same request.",
                    limitations=[*limitations, str(exc)],
                )
            run_id = "job-evidence-" + request["request_digest"][7:39]
            preparation = prepare_task_evaluation_run(
                state_root=self.run_state_root, run_id=run_id,
                capture_session_id=request["capture_session_id"], intake_id=request["intake_id"],
                request_value=request["decision_evidence_request"], testbed_value=request["testbed"],
                method_values=[], qualification_values=[], method_catalog_value=catalog,
                idempotency_key="job-" + request["request_digest"][7:39],
            )
            _immutable(root / "preparation.json", preparation)
            return self._publish(
                request, state="planned", summary="Pipeline has planned the evidence needed for this exact question.",
                next_action="Run available local evidence, or use the existing bounded evaluation preparation for new policy execution.",
                limitations=[*limitations, "Planning does not authorize provider spending or physical work."],
                preparation=preparation,
            )

    def inspect(self, request_id: str) -> dict[str, Any]:
        with self._lock():
            return self._current(request_id)

    def cancel(self, request_id: str, request_digest: str, cancelled_by: str) -> dict[str, Any]:
        _id(cancelled_by, "cancelled_by")
        with self._lock():
            request = _read(self._root(request_id) / "request.json")
            if request["request_digest"] != request_digest:
                raise JobEvidenceError("job_evidence_request_binding_mismatch")
            current = self._current(request_id)
            if current["state"] in {"decision", "partial_decision", "abstention", "cancelled"}:
                return current
            _immutable(self._root(request_id) / "cancellation.json", {
                "request_digest": request_digest, "cancelled_by": cancelled_by,
            })
            preparation_path = self._root(request_id) / "preparation.json"
            preparation = _read(preparation_path) if preparation_path.exists() else None
            if preparation:
                _immutable(
                    self.run_state_root / "runs" / preparation["run_id"] / "artifacts" / "job_evidence_cancellation.json",
                    {"job_request_id": request_id, "request_digest": request_digest, "cancelled_by": cancelled_by},
                )
            return self._publish(
                request, state="cancelled", summary="This evidence request was cancelled.",
                next_action="Use a new request if the decision is still needed.",
                limitations=["No new execution is permitted by this request. Any separately launched paid run keeps its own cancellation and teardown process."],
                preparation=preparation,
            )

    def execute(self, request_id: str, request_digest: str, authorized_by: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(authorized_by, Mapping):
            raise JobEvidenceError("job_evidence_execution_actor_invalid")
        actor = dict(authorized_by)
        if set(actor) != {"role", "identity"} or actor.get("role") != "human":
            raise JobEvidenceError("job_evidence_execution_actor_invalid")
        _id(actor.get("identity"), "authorized_by.identity")
        with self._lock():
            request = _read(self._root(request_id) / "request.json")
            if request["request_digest"] != request_digest:
                raise JobEvidenceError("job_evidence_request_binding_mismatch")
            current = self._current(request_id)
            if current["state"] in {"decision", "partial_decision", "abstention"}:
                return current
            if current["state"] == "cancelled":
                raise JobEvidenceError("job_evidence_cancelled")
            preparation = _read(self._root(request_id) / "preparation.json")
            candidates = preparation["authorization_candidates"]
            references = [row["adapter_reference"] for row in candidates]
            try:
                if any(float(row["expected_cost_usd"]) != 0 for row in candidates):
                    raise ValueError("job_evidence_paid_execution_requires_existing_allocator")
                authorized_local_evidence_adapter_registry(references)
            except ValueError as exc:
                return self._publish(
                    request, state="blocked", summary="This plan needs an existing authorized runtime or additional evidence.",
                    next_action="Use the configured-scene evaluation preparation for supported policies, or collect the evidence requested in the plan.",
                    limitations=[str(exc)], preparation=preparation,
                )
            authorize_task_evaluation_run(
                state_root=self.run_state_root, run_id=preparation["run_id"],
                plan_digest=preparation["evidence_plan"]["plan_digest"],
                authorized_adapter_references=references, actor=actor,
                idempotency_key="job-local-" + request_digest[7:39],
            )
            result = execute_and_aggregate_task_evaluation_run(
                state_root=self.run_state_root, run_id=preparation["run_id"],
            )
            envelope = DecisionEnvelope.from_mapping(result["decision_envelope"]).to_mapping()
            plan = EvidencePlan.from_mapping(preparation["evidence_plan"]).to_mapping()
            if any(envelope[field] != plan[field] for field in (
                "request_digest", "plan_digest", "testbed_digest"
            )):
                raise JobEvidenceError("job_evidence_result_binding_mismatch")
            verdicts = envelope["per_claim_verdicts"]
            summary = "; ".join(f"{row['claim_id']}: {row['verdict']}" for row in verdicts)
            next_experiment = envelope["next_cheapest_experiment"]
            next_action = (
                "No additional measurement is requested for this question. Blueprint can assess the available delivery proposals."
                if next_experiment == "none_required" else next_experiment.replace("_", " ")
            )
            return self._publish(
                request, state=envelope["overall_outcome"], summary=summary,
                next_action=next_action,
                limitations=[
                    "The result applies only to the attached task, configuration and validation envelope.",
                    "It does not select a delivery proposal or authorize physical work.",
                    *[str(row) for row in envelope["unsupported_conditions"]],
                ], preparation=preparation, envelope=envelope,
            )
