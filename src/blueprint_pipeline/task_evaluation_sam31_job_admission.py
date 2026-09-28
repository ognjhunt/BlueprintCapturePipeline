"""Provider-free admission of exact SAM phase jobs and their parent evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_sam31_parent_evidence import _parent, configured_parent_route, retained_parent
from .task_evaluation_sam31_preparation_queue import verify_evidence_reference
from .task_evaluation_scene_construction_recipe import validate_scene_construction_recipe
from .task_evaluation_sam31_phase_queue import JOB_SCHEMA, PHASES, _read, _ref, _require

def _collect_preparation_references(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect typed immutable references without importing the hot launch worker."""

    references: list[dict[str, Any]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"uri", "digest", "size_bytes"}:
                references.append(
                    {
                        "contract_path": ".".join(path),
                        "uri": str(node["uri"]),
                        "digest": str(node["digest"]),
                        "size_bytes": int(node["size_bytes"]),
                    }
                )
                return
            for key, child in node.items():
                visit(child, (*path, str(key)))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, child in enumerate(node):
                visit(child, (*path, str(index)))

    visit(value, ())
    identities: dict[str, tuple[str, int]] = {}
    for reference in references:
        identity = (reference["digest"], reference["size_bytes"])
        prior = identities.setdefault(reference["uri"], identity)
        _require(prior == identity, "reference_uri_identity_conflict")
    return references


def _validated_job(job: dict, *, parent_queue: Path, input_root: Path,
                   source_commit: str, approved_roots: Sequence[Path],
                   validation_purpose: Literal["new_execution", "retained_offline_replay"] = "new_execution") -> tuple[dict, dict]:
    _require(validation_purpose in {"new_execution", "retained_offline_replay"}, "validation_purpose_invalid")
    _require(set(job) == {"schema_version", "child_id", "parent_preparation_id",
             "parent_request_digest", "plan_digest", "phase", "inputs_digest",
             "expected_source_commit", "plan_ref", "inputs", "job_digest"}
             and job.get("schema_version") == JOB_SCHEMA and job.get("phase") in PHASES
             and job.get("job_digest") == canonical_digest(job, digest_field="job_digest"),
             "job_contract_invalid")
    identities = {name: {key: value[key] for key in ("sha256", "size_bytes")}
                  for name, value in job["inputs"].items()}
    key = {name: job[name] for name in ("parent_request_digest", "plan_digest", "phase", "inputs_digest")}
    _require(job["inputs_digest"] == canonical_digest(identities)
             and job["child_id"] == "sam31-" + canonical_digest(key).removeprefix("sha256:"),
             "job_identity_invalid")
    parent_queue, input_root = configured_parent_route(job, parent_queue, input_root)
    # This helper reads evidence; production execution always uses its default
    # current admission. Offline replay alone may interpret a closed historical
    # administrative contract, retaining all parent/child and byte joins.
    reader = retained_parent if validation_purpose == "retained_offline_replay" else _parent
    request, _, _ = reader(job, parent_queue)
    _require(request["expected_production_commit"] == source_commit == job["expected_source_commit"],
             "source_commit_mismatch")
    plan_ref = _ref(job["plan_ref"])
    plan_path = verify_evidence_reference(plan_ref, approved_roots)
    _require(plan_ref["sha256"] == job["plan_digest"], "plan_identity_mismatch")
    bound = [ref for ref in _collect_preparation_references(request)
             if ref["contract_path"].startswith("runtime.mounts.")
             and ref["digest"] == plan_ref["sha256"] and ref["size_bytes"] == plan_ref["size_bytes"]]
    _require(bool(bound), "plan_not_bound_to_parent")
    cas = input_root / "content-addressed" / "sha256"
    def read_cas(ref):
        path = cas / ref["digest"].removeprefix("sha256:")
        verify_evidence_reference({"path": str(path), "sha256": ref["digest"],
                                  "size_bytes": ref["size_bytes"]}, (input_root,))
        return _read(path)
    recipe = validate_scene_construction_recipe(read_cas(request["construction"]["recipe"]))
    stage_one = read_cas(recipe["stage_sequence"][0]["configuration"])
    declared = stage_one.get("sam31_preparation_plan", {})
    _require(stage_one.get("sam31_review_kind") == "ai"
             and declared.get("digest") == plan_ref["sha256"]
             and declared.get("size_bytes") == plan_ref["size_bytes"]
             and any(ref["uri"] == declared.get("uri") for ref in bound), "plan_stage_binding_invalid")
    for value in job["inputs"].values():
        verify_evidence_reference(_ref(value), approved_roots)
    return request, _read(plan_path)

