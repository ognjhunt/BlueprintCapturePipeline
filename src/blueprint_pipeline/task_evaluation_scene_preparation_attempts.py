"""Preparation identities are not paid reservations and cannot authorize a GPU."""
from __future__ import annotations


from . import task_evaluation_scene_intake as intake
from .task_evaluation_scene_owner_authority import reopen_scene_intent
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_progression_state import require, safe_path

SCHEMA = "task_evaluation_scene_preparation_attempt.v1"


def preparation_attempt_path(directory, attempt_id):
    require(intake._identifier(attempt_id), "preparation_attempt_id_invalid")
    directory = safe_path(directory)
    paths = [directory / name / (attempt_id + ".json") for name in ("preparation-attempts", "attempts")]
    found = [path for path in paths if path.exists()]
    require(len(found) <= 1, "preparation_attempt_identity_ambiguous")
    return found[0] if found else paths[0]


def create_preparation_attempt(*, directory, attempt_id, source_commit, runtime_digest, input_digest, now=None):
    directory = safe_path(directory)
    intent = reopen_scene_intent(record(directory / "intent.json"), now=now)
    require(intake._COMMIT.fullmatch(source_commit) is not None
            and intake._DIGEST.fullmatch(runtime_digest) is not None
            and intake._DIGEST.fullmatch(input_digest) is not None, "preparation_attempt_binding_invalid")
    value = {"schema_version": SCHEMA, "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
        "attempt_id": attempt_id, "source_commit": source_commit, "runtime_digest": runtime_digest,
        "input_digest": input_digest, "provider": "control_plane", "maximum_spend_usd": 0,
        "status": "preparation_only", "paid_authority_granted": False, "provider_allocation_permitted": False}
    value = intake._seal(value, "attempt_digest")
    path = preparation_attempt_path(directory, attempt_id)
    require(path.parent.name == "preparation-attempts", "preparation_attempt_paid_record_conflict")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if not path.exists():
        intake.write_exclusive(path, value)
    require(intake._read(path, "attempt_digest") == value, "preparation_attempt_immutable_conflict")
    return value
