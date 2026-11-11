"""Scene progression's half of preparation adoption across a deploy (dark launch).

``adopt_previous_preparation`` decides, on the first tick under a new release,
whether the attempt's $0 preparation is kept instead of retired. It returns the
earlier release record the attempt stays bound to, or None, which keeps
today's behaviour (``_release_successor`` retires and rebuilds). Any check that
cannot be made, or fails, returns None: adoption only ever skips $0 work.
See ``release_preparation_adoption`` for the rule and the switch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import task_evaluation_scene_intake as intake
from .release_preparation_adoption import contract_identity, enabled, record_adoption, record_path
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, release_inputs

REFUSED_LAUNCH_STATUS = "adopted_activation_refused"


def _verified_prior_release(output: Path, commit: str) -> dict[str, Any]:
    """The attempt's own release snapshot, proven still retained: every receipt and publication verifies."""
    prior = read(output / "release_binding.json", digest_field="release_digest")
    if prior.get("source_commit") != commit:
        raise ValueError("release_adoption_snapshot_mismatch")
    paths = {key: checked_file(prior[key]["path"], prior[key])
             for key in ("deploy_receipt", "release_provenance", "release_environment")}
    release_inputs(deploy_path=paths["deploy_receipt"], provenance_path=paths["release_provenance"],
                   publication_root=Path(prior["runtime_publication_root"]), commit=commit,
                   release_admission_mode=prior["release_admission_mode"])
    return prior


def adopt_previous_preparation(*, directory: Path, intent: Mapping[str, Any], state: dict[str, Any],
                               config: Mapping[str, Any], release: Mapping[str, Any],
                               binding: Mapping[str, Any]) -> dict[str, Any] | None:
    if not enabled():
        return None
    current = state.get("release_adoption")
    if isinstance(current, Mapping) and current.get("refused"):
        return None  # Refused once on the adopting release: rebuild there, once.
    sticky = isinstance(current, Mapping) and current.get("adopting_commit") == release["source_commit"]
    # A new adoption only before any paid reservation exists for the attempt.
    if not sticky and state.get("activation_link"):
        return None
    identity = state.get("preparation_contract")
    if not state.get("factory") or not isinstance(identity, str) or identity != contract_identity():
        return None
    try:
        from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest
        from .task_evaluation_scene_preparation_identity import SCHEMA, preparation_attempt_path
        attempt_id, commit = state["attempt_id"], state["attempt_commit"]
        attempt = intake._read(preparation_attempt_path(directory, attempt_id), "attempt_digest")
        if not (attempt.get("schema_version") == SCHEMA and attempt.get("maximum_spend_usd") == 0
                and attempt.get("paid_authority_granted") is False
                and attempt.get("intent_digest") == intent["intent_digest"]
                and attempt.get("input_digest") == binding["binding_digest"]
                and attempt.get("source_commit") == commit):
            return None
        output = Path(config["factory_output_root"]) / intent["intent_id"] / attempt_id
        prior = _verified_prior_release(output, commit)
        factory = read(checked_file(state["factory"]["path"], state["factory"]), digest_field="factory_digest")
        if "submission_request" not in factory or factory.get("source_commit") != commit:
            return None
        request = read(checked_file(factory["submission_request"]["path"], factory["submission_request"]))
        digest = launch_preparation_request_digest(request)
        if request.get("expected_production_commit") != commit:
            return None
        queue_root = Path(config["preparation_queue_root"])
        value = record_adoption(queue_root=queue_root, request_digest=digest,
            preparation_id=request["preparation_id"], prepared_commit=commit,
            adopting_commit=release["source_commit"], identity=identity,
            intent_id=intent["intent_id"], attempt_id=attempt_id)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    state["release_adoption"] = {
        "prepared_commit": commit, "adopting_commit": release["source_commit"], "contract_identity": identity,
        "record": record(record_path(queue_root, digest, release["source_commit"])),
        "record_digest": value["record_digest"]}
    return prior


def refuse_adoption(state: dict[str, Any], blockers) -> None:
    """The adopted preparation was refused before any provider call: the next tick rebuilds it once."""
    adoption = state.get("release_adoption")
    if isinstance(adoption, dict):
        adoption["refused"] = [str(item)[:200] for item in (blockers or ["release_adoption_refused"])][:8]
