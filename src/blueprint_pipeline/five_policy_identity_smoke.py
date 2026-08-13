"""Five released-policy identity smoke with immutable checkpoint evidence.

This lane performs exactly one inference per selected public DROID checkpoint.
It preserves the complete native action output, but does not execute actions,
rank policies, or claim task success.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256


REGISTRY_SCHEMA_VERSION = "five_policy_proveit_registry.v1"
INPUT_SCHEMA_VERSION = "five_policy_identity_smoke_input.v1"
INPUT_RECEIPT_SCHEMA_VERSION = "five_policy_identity_smoke_input_receipt.v1"
RESULT_SCHEMA_VERSION = "five_policy_identity_smoke_result.v1"
MANIFEST_NAME = "five_policy_identity_smoke_manifest.json"
REGISTRY_NAME = "five_policy_proveit_registry.json"
EXTERIOR_NAME = "exterior_rgb.uint8"
WRIST_NAME = "wrist_rgb.uint8"
STATE_NAME = "state.json"
IMAGE_SHAPE = (224, 224, 3)
IMAGE_BYTES = 224 * 224 * 3
EXPECTED_POLICY_COUNT = 5
MAX_INPUT_BYTES = 2 * 1024 * 1024
GCS_API = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
GCS_FIELDS = (
    "name",
    "generation",
    "metageneration",
    "size",
    "md5Hash",
    "crc32c",
    "etag",
)
PHASE_LOG_PREFIX = "BLUEPRINT_FIVE_POLICY_PHASE:"


def _emit_phase(phase: str, *, candidate_id: str | None = None, **evidence: Any) -> None:
    """Emit a secret-free, machine-readable progress marker to worker logs."""

    payload: dict[str, Any] = {"phase": phase, **evidence}
    if candidate_id is not None:
        payload["candidate_id"] = candidate_id
    print(PHASE_LOG_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_gcs_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("gcs_checkpoint_items_missing")
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("gcs_checkpoint_item_not_object")
        row = {field: item.get(field) for field in GCS_FIELDS}
        if not str(row["name"] or "") or not str(row["generation"] or "").isdigit():
            raise ValueError("gcs_checkpoint_object_identity_invalid")
        if not str(row["size"] or "").isdigit() or not str(row["md5Hash"] or ""):
            raise ValueError("gcs_checkpoint_object_integrity_invalid")
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["name"]), int(row["generation"])))


def gcs_generation_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_registry(registry: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        blockers.append("five_policy_registry_schema_invalid")
    openpi = registry.get("openpi")
    if not isinstance(openpi, Mapping) or not str(openpi.get("source_revision") or ""):
        blockers.append("five_policy_registry_openpi_identity_missing")
    elif any(
        len(str(openpi.get(field) or "")) != 64
        for field in (
            "apache_2_license_sha256",
            "gemma_terms_sha256",
            "droid_readme_sha256",
        )
    ):
        blockers.append("five_policy_registry_license_identity_invalid")
    cohort = registry.get("direct_droid_execution_cohort")
    cohort = cohort if isinstance(cohort, list) else []
    ids = [str(row.get("candidate_id") or "") for row in cohort if isinstance(row, Mapping)]
    if len(cohort) != EXPECTED_POLICY_COUNT or len(set(ids)) != EXPECTED_POLICY_COUNT:
        blockers.append("five_policy_registry_exact_unique_cohort_required")
    total = 0
    for index, row in enumerate(cohort):
        if not isinstance(row, Mapping):
            blockers.append(f"five_policy_registry_candidate_not_object:{index}")
            continue
        uri = str(row.get("checkpoint_uri") or "")
        if not uri.startswith("gs://openpi-assets/checkpoints/"):
            blockers.append(f"five_policy_registry_checkpoint_uri_invalid:{index}")
        if row.get("embodiment_route") != "direct_droid_franka":
            blockers.append(f"five_policy_registry_route_not_direct:{index}")
        if row.get("native_action_chunk_rows") not in {10, 15}:
            blockers.append(f"five_policy_registry_horizon_invalid:{index}")
        if row.get("native_action_dimensions") != 8:
            blockers.append(f"five_policy_registry_action_dimensions_invalid:{index}")
        digest = str(row.get("gcs_generation_manifest_sha256") or "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            blockers.append(f"five_policy_registry_generation_digest_invalid:{index}")
        total += int(row.get("checkpoint_size_bytes") or 0)
    if total != registry.get("direct_cohort_total_checkpoint_bytes"):
        blockers.append("five_policy_registry_checkpoint_total_mismatch")
    adapters = registry.get("adapter_required_discovery")
    adapters = adapters if isinstance(adapters, list) else []
    if len(adapters) < 2 or any(
        not isinstance(row, Mapping)
        or row.get("selected_for_paid_smoke") is not False
        or not str(row.get("required_adapter") or "")
        for row in adapters
    ):
        blockers.append("five_policy_registry_adapter_discovery_invalid")
    return sorted(set(blockers))


def _deterministic_observation_bytes() -> tuple[bytes, bytes, bytes]:
    exterior = bytes((index * 17 + 23) % 256 for index in range(IMAGE_BYTES))
    wrist = bytes((index * 29 + 11) % 256 for index in range(IMAGE_BYTES))
    state = json.dumps(
        {
            "joint_position": [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8],
            "gripper_position": [0.0],
            "prompt": "Inspect the marked work-surface region and keep it visible in the wrist camera.",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return exterior, wrist, state


def build_input_bundle(*, registry_path: str | Path, output_zip: str | Path) -> dict[str, Any]:
    registry_file = Path(registry_path).expanduser().resolve()
    destination = Path(output_zip).expanduser().resolve()
    registry_bytes = registry_file.read_bytes()
    registry = json.loads(registry_bytes)
    blockers = validate_registry(registry)
    if blockers:
        raise ValueError("five_policy_registry_invalid:" + ",".join(blockers))
    exterior, wrist, state = _deterministic_observation_bytes()
    files = {
        REGISTRY_NAME: registry_bytes,
        EXTERIOR_NAME: exterior,
        WRIST_NAME: wrist,
        STATE_NAME: state,
    }
    manifest: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "candidate_count": EXPECTED_POLICY_COUNT,
        "candidate_ids": [row["candidate_id"] for row in registry["direct_droid_execution_cohort"]],
        "registry_sha256": _sha256_bytes(registry_bytes),
        "file_sha256": {name: _sha256_bytes(value) for name, value in sorted(files.items())},
        "image_shape": list(IMAGE_SHAPE),
        "query_count_per_candidate": 1,
        "purpose": "internal_noncommercial_identity_bound_inference_smoke",
        "raw_3dgs_included": False,
        "redistribution_authorized": False,
        "physical_robot_endpoint_access_allowed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name, value in sorted(files.items()):
            archive.writestr(name, value)
    receipt: dict[str, Any] = {
        "schema_version": INPUT_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle_path": str(destination),
        "bundle_sha256": _sha256_file(destination),
        "bundle_size_bytes": destination.stat().st_size,
        "manifest": manifest,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def extract_input_bundle(
    *, bundle_path: str | Path, expected_bundle_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not bundle.is_file() or bundle.is_symlink() or bundle.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("five_policy_input_bundle_missing_unsafe_or_too_large")
    if _sha256_file(bundle) != expected_bundle_sha256:
        raise ValueError("five_policy_input_bundle_sha256_mismatch")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        expected_names = {MANIFEST_NAME, REGISTRY_NAME, EXTERIOR_NAME, WRIST_NAME, STATE_NAME}
        if set(names) != expected_names or len(names) != len(expected_names):
            raise ValueError("five_policy_input_bundle_inventory_invalid")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.file_size > MAX_INPUT_BYTES:
                raise ValueError("five_policy_input_bundle_member_unsafe")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        declared = manifest.get("manifest_sha256")
        payload = dict(manifest)
        payload.pop("manifest_sha256", None)
        if declared != canonical_sha256(payload):
            raise ValueError("five_policy_input_manifest_digest_mismatch")
        output.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name in expected_names - {MANIFEST_NAME}:
            value = archive.read(name)
            if _sha256_bytes(value) != manifest.get("file_sha256", {}).get(name):
                raise ValueError(f"five_policy_input_member_digest_mismatch:{name}")
            path = output / name
            path.write_bytes(value)
            paths[name] = str(path)
    registry = json.loads(Path(paths[REGISTRY_NAME]).read_text(encoding="utf-8"))
    blockers = validate_registry(registry)
    if blockers:
        raise ValueError("five_policy_registry_invalid:" + ",".join(blockers))
    return {"manifest": manifest, "registry": registry, "paths": paths}


def _fetch_gcs_metadata(checkpoint_uri: str) -> dict[str, Any]:
    prefix = checkpoint_uri.removeprefix("gs://openpi-assets/").rstrip("/") + "/"
    query = urllib.parse.urlencode({"prefix": prefix, "versions": "true"})
    request = urllib.request.Request(f"{GCS_API}?{query}", method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        return json.loads(response.read())


def _download_checkpoint(checkpoint_uri: str) -> Path:
    try:
        from openpi.shared import download
    except ImportError as exc:  # pragma: no cover - GPU image only
        raise RuntimeError("openpi_gpu_runtime_not_installed") from exc
    return Path(download.maybe_download(checkpoint_uri)).expanduser().resolve()


def _load_policy(config_name: str, checkpoint: Path) -> Any:
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as exc:  # pragma: no cover - GPU image only
        raise RuntimeError("openpi_gpu_runtime_not_installed") from exc
    config = training_config.get_config(config_name)
    return policy_config.create_trained_policy(config, checkpoint)


def _verify_checkpoint_files(
    *, checkpoint: Path, checkpoint_uri: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    prefix = checkpoint_uri.removeprefix("gs://openpi-assets/").rstrip("/") + "/"
    verified: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        if not name.startswith(prefix):
            raise ValueError("checkpoint_metadata_object_outside_prefix")
        relative = PurePosixPath(name.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint_metadata_relative_path_invalid")
        if not relative.parts:
            continue
        local = checkpoint.joinpath(*relative.parts)
        expected_size = int(row["size"])
        if expected_size == 0 and not local.exists():
            continue
        if not local.is_file() or local.is_symlink() or local.stat().st_size != expected_size:
            raise ValueError(f"checkpoint_local_object_size_mismatch:{relative.as_posix()}")
        digest = hashlib.md5(usedforsecurity=False)
        with local.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_md5 = base64.b64encode(digest.digest()).decode("ascii")
        if actual_md5 != row["md5Hash"]:
            raise ValueError(f"checkpoint_local_object_md5_mismatch:{relative.as_posix()}")
        verified.append({"relative_path": relative.as_posix(), "size": expected_size, "md5Hash": actual_md5})
    return {
        "verified_object_count": len(verified),
        "verified_size_bytes": sum(row["size"] for row in verified),
        "local_verification_sha256": canonical_sha256({"objects": verified}),
    }


def _gpu_evidence() -> dict[str, Any]:
    try:
        import jax

        devices = [
            {"platform": str(device.platform), "kind": str(device.device_kind), "id": int(device.id)}
            for device in jax.devices()
        ]
        return {
            "jax_version": str(jax.__version__),
            "devices": devices,
            "gpu_present": any(row["platform"] == "gpu" for row in devices),
        }
    except Exception as exc:  # noqa: BLE001
        return {"gpu_present": False, "error_type": type(exc).__name__}


def _load_observation(paths: Mapping[str, str]) -> dict[str, Any]:
    import numpy as np

    exterior = np.frombuffer(Path(paths[EXTERIOR_NAME]).read_bytes(), dtype=np.uint8).reshape(IMAGE_SHAPE)
    wrist = np.frombuffer(Path(paths[WRIST_NAME]).read_bytes(), dtype=np.uint8).reshape(IMAGE_SHAPE)
    state = json.loads(Path(paths[STATE_NAME]).read_text(encoding="utf-8"))
    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.asarray(state["joint_position"], dtype=np.float32),
        "observation/gripper_position": np.asarray(state["gripper_position"], dtype=np.float32),
        "prompt": str(state["prompt"]),
    }


def run_identity_smoke(
    *,
    extracted: Mapping[str, Any],
    output_dir: str | Path,
    metadata_fetcher: Callable[[str], Mapping[str, Any]] = _fetch_gcs_metadata,
    checkpoint_downloader: Callable[[str], Path] = _download_checkpoint,
    policy_loader: Callable[[str, Path], Any] = _load_policy,
    require_gpu: bool = True,
) -> dict[str, Any]:
    import numpy as np

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = extracted["registry"]
    manifest = extracted["manifest"]
    observation = _load_observation(extracted["paths"])
    gpu = _gpu_evidence()
    _emit_phase("gpu_gate_observed", gpu_present=gpu.get("gpu_present") is True)
    blockers: list[str] = []
    if require_gpu and gpu.get("gpu_present") is not True:
        blockers.append("five_policy_smoke_gpu_not_present")
    receipts: list[dict[str, Any]] = []
    if not blockers:
        for candidate in registry["direct_droid_execution_cohort"]:
            candidate_id = str(candidate["candidate_id"])
            checkpoint_uri = str(candidate["checkpoint_uri"])
            policy = None
            started = time.time_ns()
            _emit_phase("candidate_started", candidate_id=candidate_id)
            try:
                rows = canonical_gcs_rows(metadata_fetcher(checkpoint_uri))
                generation_digest = gcs_generation_manifest_sha256(rows)
                if generation_digest != candidate["gcs_generation_manifest_sha256"]:
                    raise ValueError("checkpoint_generation_manifest_changed")
                if len(rows) != candidate["checkpoint_object_count"]:
                    raise ValueError("checkpoint_object_count_changed")
                if sum(int(row["size"]) for row in rows) != candidate["checkpoint_size_bytes"]:
                    raise ValueError("checkpoint_size_changed")
                _emit_phase(
                    "checkpoint_metadata_verified",
                    candidate_id=candidate_id,
                    object_count=len(rows),
                    size_bytes=sum(int(row["size"]) for row in rows),
                )
                _emit_phase("checkpoint_download_started", candidate_id=candidate_id)
                checkpoint = checkpoint_downloader(checkpoint_uri)
                _emit_phase("checkpoint_download_completed", candidate_id=candidate_id)
                verification = _verify_checkpoint_files(
                    checkpoint=checkpoint, checkpoint_uri=checkpoint_uri, rows=rows
                )
                _emit_phase(
                    "checkpoint_integrity_verified",
                    candidate_id=candidate_id,
                    object_count=verification["verified_object_count"],
                    size_bytes=verification["verified_size_bytes"],
                )
                _emit_phase("policy_load_started", candidate_id=candidate_id)
                policy = policy_loader(str(candidate["config_name"]), checkpoint)
                _emit_phase("policy_load_completed", candidate_id=candidate_id)
                _emit_phase("inference_started", candidate_id=candidate_id)
                raw = policy.infer(dict(observation))
                if not isinstance(raw, Mapping) or "actions" not in raw:
                    raise ValueError("policy_native_actions_missing")
                actions = np.asarray(raw["actions"], dtype=np.float64)
                expected_shape = (
                    int(candidate["native_action_chunk_rows"]),
                    int(candidate["native_action_dimensions"]),
                )
                if actions.shape != expected_shape or not np.isfinite(actions).all():
                    raise ValueError(f"policy_native_action_shape_or_finiteness_invalid:{actions.shape}")
                _emit_phase(
                    "inference_completed",
                    candidate_id=candidate_id,
                    action_shape=list(actions.shape),
                )
                rows_out = actions.tolist()
                identity = {
                    "candidate": candidate,
                    "openpi_source_revision": registry["openpi"]["source_revision"],
                    "registry_sha256": manifest["registry_sha256"],
                }
                identity_digest = canonical_sha256(identity)
                ended = time.time_ns()
                receipt: dict[str, Any] = {
                    "schema_version": "five_policy_identity_query_receipt.v1",
                    "candidate_id": candidate_id,
                    "status": "completed",
                    "policy_identity_sha256": identity_digest,
                    "checkpoint_generation_manifest_sha256": generation_digest,
                    "checkpoint_local_verification": verification,
                    "observation_manifest_sha256": manifest["manifest_sha256"],
                    "native_action_shape": list(actions.shape),
                    "native_action_rows": rows_out,
                    "native_action_sha256": canonical_sha256({"shape": list(actions.shape), "rows": rows_out}),
                    "query_started_at_ns": started,
                    "query_ended_at_ns": ended,
                    "fresh_infer_call_count": 1,
                    "fixture_or_fake": False,
                }
                receipt["receipt_sha256"] = canonical_sha256(receipt)
                receipts.append(receipt)
                write_json(output / f"{candidate_id}.query_receipt.json", receipt)
                _emit_phase("receipt_written", candidate_id=candidate_id)
            except Exception as exc:  # noqa: BLE001 - preserve per-policy failure
                blockers.append(f"five_policy_smoke_failed:{candidate_id}:{type(exc).__name__}:{exc}")
                _emit_phase(
                    "candidate_failed",
                    candidate_id=candidate_id,
                    error_type=type(exc).__name__,
                )
            finally:
                policy = None
                gc.collect()
                try:
                    import jax

                    jax.clear_caches()
                except Exception:  # noqa: BLE001
                    pass
                _emit_phase("candidate_cleanup_completed", candidate_id=candidate_id)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed" if not blockers and len(receipts) == EXPECTED_POLICY_COUNT else "blocked",
        "blockers": sorted(set(blockers)),
        "gpu_runtime": gpu,
        "registry_sha256": manifest.get("registry_sha256"),
        "observation_manifest_sha256": manifest.get("manifest_sha256"),
        "expected_candidate_count": EXPECTED_POLICY_COUNT,
        "completed_identity_query_count": len(receipts),
        "query_receipts": receipts,
        "claim_boundary": {
            "real_checkpoint_inference_observed": len(receipts) == EXPECTED_POLICY_COUNT,
            "actions_executed": False,
            "policy_ranking": False,
            "task_success": False,
            "physical_robot_execution": False,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    write_json(output / "five_policy_identity_smoke_result.json", result)
    _emit_phase(
        "smoke_completed" if result["status"] == "completed" else "smoke_blocked",
        completed_identity_query_count=len(receipts),
        expected_candidate_count=EXPECTED_POLICY_COUNT,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-input")
    build.add_argument("--registry", required=True)
    build.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-input":
        receipt = build_input_bundle(registry_path=args.registry, output_zip=args.output)
        write_json(Path(args.output).with_suffix(".receipt.json"), receipt)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INPUT_RECEIPT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "build_input_bundle",
    "canonical_gcs_rows",
    "extract_input_bundle",
    "gcs_generation_manifest_sha256",
    "run_identity_smoke",
    "validate_registry",
]
