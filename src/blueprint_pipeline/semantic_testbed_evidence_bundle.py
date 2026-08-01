"""Immutable Pipeline-owned semantic evidence bundles for testbed compilation.

The browser never supplies these artifacts.  A trusted local semantic stage
writes one bundle beside an exact reconstruction execution, and the live
testbed compiler reloads and revalidates that binding before projecting the
evidence into a maintained Site-Task Testbed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_digest, canonical_json
from .site_task_testbed_compiler import (
    SiteTaskTestbedCompilerError,
    validate_semantic_evidence_artifacts,
)


BUNDLE_SCHEMA_VERSION = "semantic_testbed_evidence_bundle.v1"
BUNDLE_FILENAME = "semantic_testbed_evidence_bundle.json"

_SEMANTIC_RESULT_SCHEMAS = {
    "semantic_gaussian_lifting": "semantic_gaussian_lifting_result.v1",
    "semantic_oriented_boxes": "semantic_oriented_box_result.v1",
    "semantic_collision_validation": "semantic_collision_validation_result.v1",
    "semantic_geometry_benchmark": "semantic_geometry_benchmark_result.v1",
}
_TERMINAL_STATUSES = {"completed", "partially_completed", "abstained", "blocked"}
_MAX_RESULT_BYTES = 256 * 1024 * 1024


class SemanticTestbedEvidenceBundleError(ValueError):
    """Fail-closed bundle validation or immutable-storage error."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_object(path: Path, *, code: str, max_bytes: int = _MAX_RESULT_BYTES) -> dict[str, Any]:
    if path.is_symlink():
        raise SemanticTestbedEvidenceBundleError(f"{code}:symlink_forbidden")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SemanticTestbedEvidenceBundleError(f"{code}:missing") from exc
    if size <= 0 or size > max_bytes:
        raise SemanticTestbedEvidenceBundleError(f"{code}:size_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticTestbedEvidenceBundleError(f"{code}:invalid_json") from exc
    if not isinstance(value, Mapping):
        raise SemanticTestbedEvidenceBundleError(f"{code}:not_object")
    return dict(value)


def _plan_root(state_root: str | Path, plan_id: str) -> tuple[Path, str]:
    try:
        identifier = strict_identifier(plan_id, field="plan_id", max_length=192)
    except ValueError as exc:
        raise SemanticTestbedEvidenceBundleError(str(exc)) from exc
    return Path(state_root).expanduser().resolve() / "plans" / identifier, identifier


def _execution_binding(
    *, state_root: str | Path, plan_id: str, execution_result_digest: str
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    root, identifier = _plan_root(state_root, plan_id)
    context = _read_object(root / "artifacts" / "context.json", code="context")
    execution = _read_object(
        root / "artifacts" / "execution_result.json", code="execution_result"
    )
    supplied = str(execution_result_digest or "").strip()
    if execution.get("execution_result_digest") != supplied:
        raise SemanticTestbedEvidenceBundleError("execution_result_digest:mismatch")
    if canonical_digest(execution, digest_field="execution_result_digest") != supplied:
        raise SemanticTestbedEvidenceBundleError("execution_result_digest:invalid")
    if execution.get("plan_id") != identifier:
        raise SemanticTestbedEvidenceBundleError("execution_result:plan_mismatch")
    if execution.get("context_digest") != context.get("context_digest"):
        raise SemanticTestbedEvidenceBundleError("execution_result:context_mismatch")
    if execution.get("state") not in {"completed", "partial", "abstained"}:
        raise SemanticTestbedEvidenceBundleError("execution_result:not_terminal")
    return context, execution, root, identifier


def _validate_semantic_results(
    artifacts: Mapping[str, Any],
    *,
    capture_digest: str,
    reconstruction_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not artifacts:
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_artifacts:empty")
    unknown = sorted(set(artifacts) - set(_SEMANTIC_RESULT_SCHEMAS))
    if unknown:
        raise SemanticTestbedEvidenceBundleError(
            "semantic_evidence_artifacts:unsupported:" + ",".join(unknown)
        )
    reconstruction_digests = {
        str(row.get("reconstruction_result_digest") or "")
        for row in reconstruction_results
    }
    splat_digests = {
        str(reference.get("digest") or "")
        for row in reconstruction_results
        for reference in (
            row.get("asset_references", {}).values()
            if isinstance(row.get("asset_references"), Mapping)
            else []
        )
        if isinstance(reference, Mapping)
    }
    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(artifacts):
        raw = artifacts[name]
        if not isinstance(raw, Mapping):
            raise SemanticTestbedEvidenceBundleError(f"{name}:not_object")
        result = json.loads(canonical_json(dict(raw)))
        if result.get("schema_version") != _SEMANTIC_RESULT_SCHEMAS[name]:
            raise SemanticTestbedEvidenceBundleError(f"{name}:schema_version_mismatch")
        supplied_digest = str(result.get("result_digest") or "")
        if canonical_digest(result, digest_field="result_digest") != supplied_digest:
            raise SemanticTestbedEvidenceBundleError(f"{name}:result_digest_mismatch")
        if result.get("status") not in _TERMINAL_STATUSES:
            raise SemanticTestbedEvidenceBundleError(f"{name}:not_terminal")
        bindings = _mapping(result.get("bindings"))
        if bindings.get("capture_digest") != capture_digest:
            raise SemanticTestbedEvidenceBundleError(f"{name}:capture_digest_mismatch")
        if bindings.get("reconstruction_digest") not in reconstruction_digests:
            raise SemanticTestbedEvidenceBundleError(f"{name}:reconstruction_digest_stale")
        if bindings.get("analysis_splat_digest") not in splat_digests:
            raise SemanticTestbedEvidenceBundleError(f"{name}:analysis_splat_digest_stale")
        normalized[name] = result
    return normalized


def _write_immutable(path: Path, value: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_object(path, code="semantic_evidence_bundle")
        if canonical_json(existing) != canonical_json(normalized):
            raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:immutable_conflict")
        return existing, True
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _read_object(path, code="semantic_evidence_bundle")
            if canonical_json(existing) != canonical_json(normalized):
                raise SemanticTestbedEvidenceBundleError(
                    "semantic_evidence_bundle:immutable_conflict"
                )
            return existing, True
    finally:
        temporary.unlink(missing_ok=True)
    return normalized, False


def write_semantic_testbed_evidence_bundle(
    *,
    state_root: str | Path,
    plan_id: str,
    execution_result_digest: str,
    semantic_evidence_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist exact semantic results beside an immutable reconstruction execution."""

    context, execution, root, identifier = _execution_binding(
        state_root=state_root,
        plan_id=plan_id,
        execution_result_digest=execution_result_digest,
    )
    capture_digest = str(context.get("capture_digest") or "")
    reconstruction_results = [
        dict(row) for row in execution.get("results", []) if isinstance(row, Mapping)
    ]
    artifacts = _validate_semantic_results(
        semantic_evidence_artifacts,
        capture_digest=capture_digest,
        reconstruction_results=reconstruction_results,
    )
    execution_component = str(execution_result_digest).removeprefix("sha256:")
    references = {
        name: {
            "uri": f"semantic-evidence://{identifier}/{execution_component}/{name}.json",
            "digest": result["result_digest"],
        }
        for name, result in artifacts.items()
    }
    try:
        validate_semantic_evidence_artifacts(
            artifacts,
            capture_digest=capture_digest,
            reconstruction_results=reconstruction_results,
            artifact_references=references,
        )
    except SiteTaskTestbedCompilerError as exc:
        raise SemanticTestbedEvidenceBundleError(
            f"semantic_evidence_artifacts:invalid_chain:{exc}"
        ) from exc
    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "plan_id": identifier,
        "execution_result_digest": execution_result_digest,
        "context_digest": context.get("context_digest"),
        "capture_digest": capture_digest,
        "reconstruction_result_digests": sorted(
            str(row.get("reconstruction_result_digest") or "")
            for row in reconstruction_results
        ),
        "semantic_evidence_artifacts": artifacts,
        "artifact_references": references,
        "proof_boundary": {
            "pipeline_owned_scientific_inputs": True,
            "browser_authored_science": False,
            "semantic_geometry_is_collision_truth": False,
            "semantic_geometry_is_physics_truth": False,
            "physical_task_success_established": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    bundle["bundle_digest"] = canonical_digest(bundle, digest_field="bundle_digest")
    path = root / "artifacts" / BUNDLE_FILENAME
    stored, already_exists = _write_immutable(path, bundle)
    return {**stored, "already_exists": already_exists}


def load_semantic_testbed_evidence_bundle(
    *, state_root: str | Path, plan_id: str, execution_result_digest: str
) -> dict[str, Any] | None:
    """Load a trusted bundle or return ``None`` when no semantic pass was run."""

    root, _ = _plan_root(state_root, plan_id)
    path = root / "artifacts" / BUNDLE_FILENAME
    if not path.exists():
        return None
    context, execution, _, identifier = _execution_binding(
        state_root=state_root,
        plan_id=plan_id,
        execution_result_digest=execution_result_digest,
    )
    bundle = _read_object(path, code="semantic_evidence_bundle")
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:schema_mismatch")
    if canonical_digest(bundle, digest_field="bundle_digest") != bundle.get("bundle_digest"):
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:digest_mismatch")
    if bundle.get("plan_id") != identifier:
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:plan_mismatch")
    if bundle.get("execution_result_digest") != execution_result_digest:
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:execution_stale")
    if bundle.get("context_digest") != context.get("context_digest"):
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:context_stale")
    if bundle.get("capture_digest") != context.get("capture_digest"):
        raise SemanticTestbedEvidenceBundleError("semantic_evidence_bundle:capture_stale")
    reconstruction_results = [
        dict(row) for row in execution.get("results", []) if isinstance(row, Mapping)
    ]
    expected_reconstruction_digests = sorted(
        str(row.get("reconstruction_result_digest") or "") for row in reconstruction_results
    )
    if bundle.get("reconstruction_result_digests") != expected_reconstruction_digests:
        raise SemanticTestbedEvidenceBundleError(
            "semantic_evidence_bundle:reconstruction_results_stale"
        )
    artifacts = _validate_semantic_results(
        _mapping(bundle.get("semantic_evidence_artifacts")),
        capture_digest=str(context.get("capture_digest") or ""),
        reconstruction_results=reconstruction_results,
    )
    references = _mapping(bundle.get("artifact_references"))
    execution_component = str(execution_result_digest).removeprefix("sha256:")
    expected_references = {
        name: {
            "uri": f"semantic-evidence://{identifier}/{execution_component}/{name}.json",
            "digest": result["result_digest"],
        }
        for name, result in artifacts.items()
    }
    if references != expected_references:
        raise SemanticTestbedEvidenceBundleError(
            "semantic_evidence_bundle:artifact_references_mismatch"
        )
    try:
        validate_semantic_evidence_artifacts(
            artifacts,
            capture_digest=str(context.get("capture_digest") or ""),
            reconstruction_results=reconstruction_results,
            artifact_references=references,
        )
    except SiteTaskTestbedCompilerError as exc:
        raise SemanticTestbedEvidenceBundleError(
            f"semantic_evidence_artifacts:invalid_chain:{exc}"
        ) from exc
    for name, result in artifacts.items():
        if _mapping(references.get(name)).get("digest") != result.get("result_digest"):
            raise SemanticTestbedEvidenceBundleError(f"{name}:artifact_reference_mismatch")
    if bundle.get("proof_boundary") != {
        "pipeline_owned_scientific_inputs": True,
        "browser_authored_science": False,
        "semantic_geometry_is_collision_truth": False,
        "semantic_geometry_is_physics_truth": False,
        "physical_task_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }:
        raise SemanticTestbedEvidenceBundleError(
            "semantic_evidence_bundle:proof_boundary_mismatch"
        )
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Store Pipeline-owned semantic evidence for one reconstruction execution."
    )
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--execution-result-digest", required=True)
    for name in _SEMANTIC_RESULT_SCHEMAS:
        parser.add_argument(f"--{name.replace('_', '-')}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifacts: dict[str, Any] = {}
    for name in _SEMANTIC_RESULT_SCHEMAS:
        raw_path = getattr(args, name)
        if raw_path:
            artifacts[name] = _read_object(
                Path(raw_path), code=f"semantic_result:{name}"
            )
    result = write_semantic_testbed_evidence_bundle(
        state_root=args.state_root,
        plan_id=args.plan_id,
        execution_result_digest=args.execution_result_digest,
        semantic_evidence_artifacts=artifacts,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "SemanticTestbedEvidenceBundleError",
    "load_semantic_testbed_evidence_bundle",
    "write_semantic_testbed_evidence_bundle",
]
