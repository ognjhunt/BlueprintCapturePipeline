"""Read-only parent-envelope proof shared by execution and prefix adoption."""
import re
import os
from pathlib import Path
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
    validate_retained_preparation_request,
)
from .task_evaluation_launch_preparation_queue import QUEUE_STATES
from .task_evaluation_sam31_phase_queue import _read, _require

def _parent(job: dict, root: Path) -> tuple[dict, str, Path]:
    """Resolve the parent of a child that may execute under current admission."""
    return _read_parent(job, root, validator=validate_launch_preparation_request)


def retained_parent(job: dict, root: Path) -> tuple[dict, str, Path]:
    """Resolve historical parent evidence solely for completed-prefix adoption."""
    return _read_parent(job, root, validator=validate_retained_preparation_request)


def _read_parent(job: dict, root: Path, *, validator) -> tuple[dict, str, Path]:
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
    request = validator(envelope["request"])
    _require(envelope.get("envelope_digest") == canonical_digest(envelope, digest_field="envelope_digest")
             and envelope.get("request_digest") == digest
             and canonical_digest(request) == digest
             and request["preparation_id"] == identifier, "parent_envelope_invalid")
    return request, state, path





def configured_parent_route(job: dict, root: Path, input_root: Path) -> tuple[Path, Path]:
    """Resolve one parent across installed legacy and owned preparation stores.

    Queue locations come only from operator configuration, never from a child
    payload. Ambiguous copies across stores or queue states remain refused.
    """
    routes = [(root, input_root)]
    configured = os.getenv("BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG")
    if configured:
        config = _read(Path(configured))
        _require(config.get("schema_version") == "task_evaluation_scene_progression_config.v1"
                 and config.get("config_digest") == canonical_digest(config, digest_field="config_digest"),
                 "parent_routing_config_invalid")
        owned = Path(config["preparation_queue_root"])
        inputs = Path(config["preparation_worker"]["input_root"])
        _require(all(p.is_absolute() and not any(x.is_symlink() for x in (p, *p.parents))
                     for p in (owned, inputs)), "parent_routing_path_invalid")
        if owned != root:
            routes.append((owned, inputs))
        else:
            routes[0] = (owned, inputs)
    identifier, digest = job.get("parent_preparation_id"), job.get("parent_request_digest")
    _require(isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identifier)
             and isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest),
             "parent_identity_invalid")
    filename = f"{identifier}-{digest[7:]}.json"
    matches = [(parent, inputs) for parent, inputs in routes for state in QUEUE_STATES
               if (parent / state / filename).exists()]
    _require(len(matches) == 1, "parent_identity_ambiguous")
    return matches[0]
