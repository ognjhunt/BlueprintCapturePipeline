"""Production no-spend worker for whole-splat object discovery.

The signed HTTP service only stages immutable requests.  This worker owns the
expensive local work: full-byte materialization, exact whole-scene rendering,
replaceable analyzer execution, deterministic metric admission, immutable
artifact publication, and terminal queue sealing.  Provider-GPU requests stop
at an explicit activation boundary and never allocate from this worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .scene_object_discovery import (
    build_full_scene_camera_plan,
    compile_scene_object_discovery,
    materialize_scene_object_discovery_renders,
)
from .scene_object_discovery_contract import validate_scene_object_discovery_request
from .scene_object_discovery_queue import (
    claim_scene_object_discovery_request,
    ensure_scene_object_discovery_queue_root,
    seal_scene_object_discovery_blocked,
    seal_scene_object_discovery_result,
)
from .task_evaluation_launch_preparation_worker import (
    _materialize_reference_records,
    collect_preparation_references,
    default_reference_fetcher,
    running_worker_source_commit,
    validate_allowed_uri_prefixes,
)


QUEUE_ROOT_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT"
INPUT_ROOT_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_INPUT_ROOT"
OUTPUT_ROOT_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_OUTPUT_ROOT"
ALLOWED_URI_PREFIXES_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_ALLOWED_URI_PREFIXES_JSON"
SERVICE_ACCOUNT_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_SERVICE_ACCOUNT"
ANALYZER_COMMANDS_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_ANALYZER_COMMANDS_JSON"
PUBLICATION_PREFIX_ENV = "BLUEPRINT_SCENE_OBJECT_DISCOVERY_PUBLICATION_PREFIX"
MAX_ANALYZER_OUTPUT_BYTES = 8 * 1024 * 1024

ReferenceFetcher = Callable[[str, Path, int], None]
RenderMaterializer = Callable[..., dict[str, Any]]
AnalyzerExecutor = Callable[..., dict[str, Any]]
ArtifactPublisher = Callable[..., dict[str, Any]]


class SceneObjectDiscoveryWorkerError(RuntimeError):
    """A claimed request cannot safely advance."""


def _load_json_mapping(path: str | Path, *, blocker: str) -> dict[str, Any]:
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneObjectDiscoveryWorkerError(blocker) from exc
    if not isinstance(value, Mapping):
        raise SceneObjectDiscoveryWorkerError(blocker)
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _materialize_inputs(
    *,
    request: Mapping[str, Any],
    input_root: str | Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    source_commit: str,
    fetcher: ReferenceFetcher,
) -> dict[str, Any]:
    validated = validate_scene_object_discovery_request(request)
    if validated["expected_production_commit"] != source_commit:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_worker_source_commit_mismatch"
        )
    try:
        account = pwd.getpwnam(service_account)
    except KeyError as exc:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_service_account_missing"
        ) from exc
    if os.geteuid() != account.pw_uid:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_service_account_identity_mismatch"
        )
    root = Path(input_root).expanduser()
    if root.is_symlink():
        raise SceneObjectDiscoveryWorkerError("scene_object_discovery_input_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve(strict=True)
    rows, unique_count = _materialize_reference_records(
        references=collect_preparation_references(validated),
        input_root=root,
        allowed_uri_prefixes=validate_allowed_uri_prefixes(allowed_uri_prefixes),
        fetcher=fetcher,
    )
    return {
        "references": rows,
        "unique_object_count": unique_count,
        "service_account": service_account,
        "service_account_uid": account.pw_uid,
        "full_byte_service_account_readback_passed": all(
            row["full_byte_service_account_readback_passed"] for row in rows
        ),
    }


def _reference_paths(materialized: Mapping[str, Any]) -> dict[str, Path]:
    return {
        str(row["contract_path"]): Path(str(row["materialized_path"])).resolve()
        for row in materialized["references"]
        if isinstance(row, Mapping)
    }


def _publisher_candidates(
    *, scene_analysis: Mapping[str, Any], request: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = scene_analysis.get("publisher_objects")
    if not isinstance(rows, list):
        return []
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label") or raw.get("name") or "").strip()
        bounds_min = raw.get("bounds_min")
        bounds_max = raw.get("bounds_max")
        if not label or not isinstance(bounds_min, list) or not isinstance(bounds_max, list):
            continue
        identity = str(raw.get("publisher_instance_id") or raw.get("object_id") or index)
        evidence = {
            "scene_analysis_digest": request["scene"]["scene_analysis"]["digest"],
            "publisher_instance_id": identity,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
        }
        candidates.append(
            {
                "candidate_id": f"publisher-{index + 1:04d}",
                "publisher_instance_id": identity,
                "label": label,
                "confidence": float(raw.get("confidence", 1.0)),
                "supporting_view_ids": [],
                "task_relevance": dict(raw.get("task_relevance") or {}),
                "metric_geometry": {
                    "authority": "publisher_metric_label",
                    "validated": raw.get("metric_validated") is True,
                    "evidence_digest": canonical_digest(evidence),
                    "bounds_min": bounds_min,
                    "bounds_max": bounds_max,
                },
            }
        )
    return candidates


def _normalize_visual_candidate(
    *, backend: str, index: int, raw: Mapping[str, Any], view_ids: Sequence[str]
) -> dict[str, Any] | None:
    label = str(raw.get("label") or raw.get("name") or "").strip()
    if not label:
        return None
    candidate_id = str(
        raw.get("candidate_id")
        or raw.get("id")
        or raw.get("object_id")
        or f"{backend}-{index + 1:04d}"
    )
    confidence = raw.get("confidence", raw.get("mean_confidence", 0.0))
    box = raw.get("boundingBox") if isinstance(raw.get("boundingBox"), Mapping) else {}
    center = box.get("center")
    extents = box.get("extents")
    metric = raw.get("metric_geometry")
    if not isinstance(metric, Mapping) and isinstance(center, list) and isinstance(extents, list):
        try:
            metric = {
                "authority": "rough_splat_analyzer_box",
                "validated": False,
                "evidence_digest": canonical_digest(
                    {"candidate_id": candidate_id, "center": center, "extents": extents}
                ),
                "bounds_min": [float(center[i]) - float(extents[i]) / 2.0 for i in range(3)],
                "bounds_max": [float(center[i]) + float(extents[i]) / 2.0 for i in range(3)],
            }
        except (IndexError, TypeError, ValueError):
            metric = None
    support = raw.get("supporting_view_ids")
    if not isinstance(support, list) or not support:
        support = list(view_ids)
    candidate = {
        "candidate_id": candidate_id,
        "label": label,
        "confidence": float(confidence),
        "supporting_view_ids": support,
        "task_relevance": dict(raw.get("task_relevance") or {}),
    }
    if isinstance(metric, Mapping):
        candidate["metric_geometry"] = dict(metric)
    preview = raw.get("preview")
    if isinstance(preview, Mapping):
        candidate["preview"] = dict(preview)
    return candidate


def _command_registry() -> dict[str, Any]:
    try:
        value = json.loads(os.getenv(ANALYZER_COMMANDS_ENV, "{}"))
    except json.JSONDecodeError as exc:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_analyzer_registry_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise SceneObjectDiscoveryWorkerError("scene_object_discovery_analyzer_registry_invalid")
    return dict(value)


def _run_command_analyzer(
    *, backend: str, context: Mapping[str, Any], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    command = configuration.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise SceneObjectDiscoveryWorkerError(f"scene_object_discovery_{backend}_command_missing")
    executable = Path(command[0])
    expected_digest = configuration.get("executable_digest")
    if (
        not executable.is_absolute()
        or executable.is_symlink()
        or not executable.is_file()
        or _sha256_file(executable) != expected_digest
    ):
        raise SceneObjectDiscoveryWorkerError(
            f"scene_object_discovery_{backend}_executable_identity_mismatch"
        )
    try:
        completed = subprocess.run(
            command,
            input=canonical_json(dict(context)),
            text=True,
            capture_output=True,
            timeout=int(configuration.get("timeout_seconds", 900)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SceneObjectDiscoveryWorkerError(
            f"scene_object_discovery_{backend}_command_failed"
        ) from exc
    if completed.returncode != 0 or len(completed.stdout.encode()) > MAX_ANALYZER_OUTPUT_BYTES:
        raise SceneObjectDiscoveryWorkerError(f"scene_object_discovery_{backend}_command_failed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SceneObjectDiscoveryWorkerError(
            f"scene_object_discovery_{backend}_output_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise SceneObjectDiscoveryWorkerError(f"scene_object_discovery_{backend}_output_invalid")
    return dict(value)


def _run_builtin_splat_analyzer(context: Mapping[str, Any]) -> dict[str, Any]:
    """Invoke the repository-owned Splat Analyzer adapter without a shell."""

    repo_root = Path(__file__).resolve().parents[2]
    adapter = repo_root / "scripts" / "object_index_splat_analyzer_runner.py"
    if adapter.is_symlink() or not adapter.is_file():
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_splat_analyzer_adapter_missing"
        )
    output_root = Path(str(context["output_root"])) / "splat-analyzer"
    output_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    request_path = output_root / "request.json"
    result_path = output_root / "result.json"
    payload = {
        "capture_root": str(output_root),
        "splat_analyzer_asset_path": context["materialized_paths"]["scene.source_splat"],
        "splat_analyzer_prompt": ", ".join(context["request"]["analysis"]["prompts"]),
        "descriptor": {"task_text": context["request"]["task"]["task_statement"]},
    }
    request_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(adapter), str(request_path), str(result_path)],
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_splat_analyzer_command_failed"
        )
    output = _load_json_mapping(
        result_path, blocker="scene_object_discovery_splat_analyzer_output_invalid"
    )
    if output.get("backend_status") not in {"ok", "skipped"}:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_splat_analyzer_command_failed"
        )
    return output


def default_analyzer_executor(*, backend: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """Run an operator-owned backend and normalize it to the compiler seam."""

    if backend == "publisher_semantics":
        rows = _publisher_candidates(
            scene_analysis=context["scene_analysis"], request=context["request"]
        )
    else:
        configuration = _command_registry().get(backend)
        if backend == "splat_analyzer" and not isinstance(configuration, Mapping):
            output = _run_builtin_splat_analyzer(context)
        elif isinstance(configuration, Mapping):
            output = _run_command_analyzer(
                backend=backend, context=context, configuration=configuration
            )
        else:
            raise SceneObjectDiscoveryWorkerError(
                f"scene_object_discovery_{backend}_qualified_adapter_missing"
            )
        raw_rows = output.get("candidates", output.get("proposals"))
        if backend == "splat_analyzer" and raw_rows is None:
            raw_rows = output.get("objects")
        if not isinstance(raw_rows, list):
            raise SceneObjectDiscoveryWorkerError(
                f"scene_object_discovery_{backend}_candidates_invalid"
            )
        view_ids = [row["camera_id"] for row in context["camera_plan"]["cameras"]]
        rows = [
            candidate
            for index, raw in enumerate(raw_rows)
            if isinstance(raw, Mapping)
            and (
                candidate := _normalize_visual_candidate(
                    backend=backend,
                    index=index,
                    raw=raw,
                    view_ids=view_ids,
                )
            )
            is not None
        ]
        if backend == "sam31":
            assert isinstance(configuration, Mapping)
            qualification = configuration.get("metric_qualification")
            qualified = (
                isinstance(qualification, Mapping)
                and qualification.get("production_large_scene_ready") is True
                and qualification.get("independent_deterministic_validation_passed") is True
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(qualification.get("evidence_digest") or ""),
                )
            )
            for candidate in rows:
                geometry = candidate.get("metric_geometry")
                if isinstance(geometry, dict) and geometry.get("authority") == (
                    "production_semantic_gaussian_obb"
                ):
                    geometry["production_large_scene_ready"] = qualified
                    geometry["independent_deterministic_validation_passed"] = qualified
    run = {
        "backend": backend,
        "source_splat_digest": context["source_binding"]["source_splat_digest"],
        "render_manifest_digest": context["render_binding"]["render_manifest_digest"],
        "candidates": rows,
    }
    run["run_digest"] = canonical_digest(run, digest_field="run_digest")
    return run


def default_artifact_publisher(
    *, value: Mapping[str, Any], relative_name: str, publication_prefix: str
) -> dict[str, Any]:
    """Publish a small source-object JSON to a governed S3 or GCS prefix."""

    body = (canonical_json(dict(value)) + "\n").encode()
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    uri = publication_prefix.rstrip("/") + "/" + relative_name.lstrip("/")
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme == "s3":
        from .task_evaluation_launch_preparation_worker import _s3_client

        client = _s3_client()
        client.put_object(Bucket=parsed.netloc, Key=key, Body=body, ContentType="application/json")
        response = client.get_object(Bucket=parsed.netloc, Key=key)
        observed = response["Body"].read(len(body) + 1)
    elif parsed.scheme == "gs":
        try:
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SceneObjectDiscoveryWorkerError(
                "scene_object_discovery_gcs_client_unavailable"
            ) from exc
        blob = gcs_storage.Client().bucket(parsed.netloc).blob(key)
        blob.upload_from_string(body, content_type="application/json")
        observed = blob.download_as_bytes()
    else:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_publication_prefix_unsupported"
        )
    if observed != body:
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_publication_readback_mismatch"
        )
    return {"uri": uri, "digest": digest, "size_bytes": len(body)}


def _publish_eligible_candidates(
    *,
    discovery: dict[str, Any],
    discovery_id: str,
    publication_prefix: str,
    publisher: ArtifactPublisher,
) -> dict[str, Any] | None:
    selected_artifact = None
    for candidate in discovery["candidates"]:
        if candidate.get("eligible_for_automatic_source_object") is not True:
            continue
        payload = {
            "schema_version": "scene_object_candidate_source_object.v1",
            "discovery_id": discovery_id,
            "candidate_id": candidate["candidate_id"],
            "label": candidate["label"],
            "metric_geometry_authority": candidate["metric_geometry_authority"],
            "metric_geometry": candidate["metric_geometry"],
            "candidate_claim_boundary": candidate["candidate_claim_boundary"],
        }
        artifact = publisher(
            value=payload,
            relative_name=f"{discovery_id}/{candidate['candidate_id']}.json",
            publication_prefix=publication_prefix,
        )
        candidate["source_object_artifact"] = artifact
        if candidate["candidate_id"] == discovery.get("selected_candidate_id"):
            selected_artifact = artifact
    discovery["discovery_digest"] = canonical_digest(discovery, digest_field="discovery_digest")
    return selected_artifact


def process_scene_object_discovery_queue(
    *,
    queue_root: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str,
    publication_prefix: str,
    source_commit: str | None = None,
    max_messages: int = 1,
    fetcher: ReferenceFetcher = default_reference_fetcher,
    render_materializer: RenderMaterializer = materialize_scene_object_discovery_renders,
    analyzer_executor: AnalyzerExecutor = default_analyzer_executor,
    artifact_publisher: ArtifactPublisher = default_artifact_publisher,
    input_materializer: Callable[..., dict[str, Any]] = _materialize_inputs,
) -> dict[str, Any]:
    """Advance bounded requests through the local, no-spend discovery path."""

    if (
        not isinstance(max_messages, int)
        or isinstance(max_messages, bool)
        or not 1 <= max_messages <= 16
    ):
        raise SceneObjectDiscoveryWorkerError("scene_object_discovery_max_messages_invalid")
    observed_commit = source_commit or running_worker_source_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", observed_commit):
        raise SceneObjectDiscoveryWorkerError(
            "scene_object_discovery_worker_source_commit_unproven"
        )
    root = ensure_scene_object_discovery_queue_root(queue_root)
    processed: list[dict[str, Any]] = []
    for pending in sorted((root / "pending").glob("*.json"))[:max_messages]:
        claimed = claim_scene_object_discovery_request(queue_root=root, pending_path=pending)
        if claimed is None:
            continue
        _, envelope = claimed
        request = validate_scene_object_discovery_request(envelope["request"])
        discovery_id = request["discovery_id"]
        request_digest = envelope["request_digest"]
        try:
            if request["execution"]["mode"] != "qualified_local_runtime":
                raise SceneObjectDiscoveryWorkerError(
                    "scene_object_discovery_provider_activation_required"
                )
            materialized = input_materializer(
                request=request,
                input_root=Path(input_root) / discovery_id,
                allowed_uri_prefixes=allowed_uri_prefixes,
                service_account=service_account,
                source_commit=observed_commit,
                fetcher=fetcher,
            )
            paths = _reference_paths(materialized)
            scene_analysis = _load_json_mapping(
                paths["scene.scene_analysis"],
                blocker="scene_object_discovery_scene_analysis_invalid",
            )
            renderer_qualification = _load_json_mapping(
                paths["scene.renderer_qualification"],
                blocker="scene_object_discovery_renderer_qualification_invalid",
            )
            if (
                renderer_qualification.get("production_method_input_authorized") is not True
                or renderer_qualification.get("source_splat_digest")
                != request["scene"]["source_splat"]["digest"]
            ):
                raise SceneObjectDiscoveryWorkerError(
                    "scene_object_discovery_renderer_not_qualified"
                )
            geometry = scene_analysis.get("scene_geometry")
            if not isinstance(geometry, Mapping):
                raise SceneObjectDiscoveryWorkerError(
                    "scene_object_discovery_scene_geometry_missing"
                )
            source_binding = {
                "source_splat_digest": request["scene"]["source_splat"]["digest"],
                "retained_gaussian_count": request["scene"]["retained_gaussian_count"],
                "registration_digest": request["scene"]["metric_registration"]["digest"],
            }
            camera_plan = build_full_scene_camera_plan(
                scene_geometry=geometry,
                source_splat_digest=source_binding["source_splat_digest"],
                retained_gaussian_count=source_binding["retained_gaussian_count"],
                registration_digest=source_binding["registration_digest"],
                normalization_transform_digest=scene_analysis.get("normalization_transform_digest"),
            )
            run_root = Path(output_root) / discovery_id
            render_result = render_materializer(
                source_splat_path=paths["scene.source_splat"],
                camera_plan=camera_plan,
                output_root=run_root / "survey-renders",
            )
            context = {
                "request": request,
                "source_binding": source_binding,
                "scene_analysis": scene_analysis,
                "camera_plan": camera_plan,
                "render_result": render_result,
                "render_binding": render_result["render_binding"],
                "materialized_paths": {key: str(value) for key, value in paths.items()},
                "output_root": str(run_root),
            }
            analyzer_runs = [
                analyzer_executor(backend=backend, context=context)
                for backend in request["analysis"]["analyzers"]
            ]
            discovery = compile_scene_object_discovery(
                source_binding=source_binding,
                camera_plan=camera_plan,
                render_binding=render_result["render_binding"],
                analyzer_runs=analyzer_runs,
                task_context=request["task"],
                minimum_confidence=request["analysis"]["minimum_confidence"],
                minimum_task_relevance=request["analysis"]["minimum_task_relevance"],
            )
            selected_artifact = _publish_eligible_candidates(
                discovery=discovery,
                discovery_id=discovery_id,
                publication_prefix=publication_prefix,
                publisher=artifact_publisher,
            )
            result = seal_scene_object_discovery_result(
                queue_root=root,
                discovery_id=discovery_id,
                request_digest=request_digest,
                source_commit=observed_commit,
                discovery=discovery,
                source_object_artifact=selected_artifact,
                paid_execution_performed=False,
            )
        except Exception as exc:
            blocker = (
                str(exc)
                if isinstance(exc, SceneObjectDiscoveryWorkerError)
                else f"scene_object_discovery_worker_failed:{type(exc).__name__}"
            )
            result = seal_scene_object_discovery_blocked(
                queue_root=root,
                discovery_id=discovery_id,
                request_digest=request_digest,
                source_commit=observed_commit,
                blockers=[blocker],
            )
        processed.append(result)
    return {
        "schema_version": "scene_object_discovery_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=os.getenv(QUEUE_ROOT_ENV, ""))
    parser.add_argument("--input-root", default=os.getenv(INPUT_ROOT_ENV, ""))
    parser.add_argument("--output-root", default=os.getenv(OUTPUT_ROOT_ENV, ""))
    parser.add_argument(
        "--allowed-uri-prefixes-json",
        default=os.getenv(ALLOWED_URI_PREFIXES_ENV, ""),
    )
    parser.add_argument("--service-account", default=os.getenv(SERVICE_ACCOUNT_ENV, "blueprint"))
    parser.add_argument("--publication-prefix", default=os.getenv(PUBLICATION_PREFIX_ENV, ""))
    parser.add_argument("--max-messages", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        prefixes = json.loads(args.allowed_uri_prefixes_json)
    except json.JSONDecodeError:
        prefixes = None
    if (
        not args.queue_root
        or not args.input_root
        or not args.output_root
        or not args.publication_prefix
        or not isinstance(prefixes, list)
        or not all(isinstance(item, str) and item for item in prefixes)
    ):
        print(
            json.dumps(
                {
                    "schema_version": "scene_object_discovery_queue_run.v1",
                    "status": "blocked",
                    "blockers": ["scene_object_discovery_worker_configuration_invalid"],
                    "provider_mutation_performed": False,
                    "paid_execution_requested": False,
                },
                sort_keys=True,
            )
        )
        return 2
    result = process_scene_object_discovery_queue(
        queue_root=args.queue_root,
        input_root=args.input_root,
        output_root=args.output_root,
        allowed_uri_prefixes=prefixes,
        service_account=args.service_account,
        publication_prefix=args.publication_prefix,
        max_messages=args.max_messages,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"processed", "idle"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SceneObjectDiscoveryWorkerError",
    "default_analyzer_executor",
    "default_artifact_publisher",
    "process_scene_object_discovery_queue",
]
