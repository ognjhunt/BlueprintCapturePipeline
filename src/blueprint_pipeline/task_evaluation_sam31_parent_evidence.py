"""Read-only parent-envelope proof shared by execution and prefix adoption."""
import re
from pathlib import Path
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest, validate_launch_preparation_request
from .task_evaluation_launch_preparation_queue import QUEUE_STATES
from .task_evaluation_sam31_phase_queue import _read, _require

def _parent(job: dict, root: Path) -> tuple[dict, str, Path]:
    digest = job.get("parent_request_digest")
    identifier = job.get("parent_preparation_id")
    _require(isinstance(identifier, str) and identifier and "/" not in identifier
             and isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None,
             "parent_identity_invalid")
    filename = f"{identifier}-{digest.removeprefix('sha256:')}.json"
    matches = [(state, root / state / filename) for state in QUEUE_STATES if (root / state / filename).exists()]
    _require(len(matches) == 1, "parent_identity_ambiguous")
    state, path = matches[0]
    envelope = _read(path)
    request = validate_launch_preparation_request(envelope["request"])
    _require(envelope.get("envelope_digest") == canonical_digest(envelope, digest_field="envelope_digest")
             and envelope.get("request_digest") == digest
             and launch_preparation_request_digest(request) == digest
             and request["preparation_id"] == identifier, "parent_envelope_invalid")
    return request, state, path



