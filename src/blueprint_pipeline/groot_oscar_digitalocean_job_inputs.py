"""Input bundle and scalar parsing helpers for the DigitalOcean closed-loop job."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .g1_kitchen_bundle_compatibility import (
    build_bundle_compatibility,
    build_source_tree_identity,
)
from .gpu_render_providers import RenderLaunchSpec
from .groot_oscar_closed_loop_image import DEFAULT_MIN_TASK_ADAPTIVE_STEPS
from .groot_oscar_closed_loop_image import SEALED_CONFIRMED_ENV


def build_digitalocean_job_parser(
    *, description: str, defaults: Mapping[str, Any]
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--audit-prepared-dir", default=None)
    parser.add_argument("--audit-objective-dir", default=None)
    parser.add_argument("--probe-digitalocean-capacity-dir", default=None)
    parser.add_argument("--wait-digitalocean-capacity-dir", default=None)
    parser.add_argument("--wait-max-attempts", type=int, default=1)
    parser.add_argument("--wait-poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--launch-when-capacity-available", action="store_true")
    parser.add_argument("--materialize-paid-resume-dir", default=None)
    parser.add_argument("--materialize-max-spend-usd", type=float, default=None)
    parser.add_argument("--acknowledge-digitalocean-query-approval", action="store_true")
    parser.add_argument("--start-frame")
    parser.add_argument("--route-file")
    parser.add_argument("--task-prompt")
    parser.add_argument("--out-dir")
    parser.add_argument("--steps", type=int, default=defaults["steps"])
    parser.add_argument("--oscar-height", type=int, default=480)
    parser.add_argument("--oscar-width", type=int, default=640)
    parser.add_argument(
        "--min-coherent-horizon-frames",
        type=int,
        default=defaults["min_coherent_horizon_frames"],
    )
    parser.add_argument("--min-steps", type=int, default=defaults["min_steps"])
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-spend-usd", type=float, default=None)
    parser.add_argument("--max-seconds", type=int, default=defaults["max_seconds"])
    parser.add_argument(
        "--max-hourly-rate-usd", type=float, default=defaults["max_hourly_rate_usd"]
    )
    parser.add_argument(
        "--container-disk-gb", type=int, default=defaults["container_disk_gb"]
    )
    parser.add_argument("--volume-gb", type=int, default=defaults["volume_gb"])
    parser.add_argument("--seed-provenance-file", default=None)
    parser.add_argument("--task-success-contract-file", default=None)
    parser.add_argument("--attempt-input-manifest-file", default=None)
    parser.add_argument("--kitchen-asset-archive-file", default=None)
    parser.add_argument("--key-prefix", default="blueprint/groot-oscar-closed-loop")
    parser.add_argument("--image-ref", default=None)
    parser.add_argument(
        "--worker-image-manifest-diagnostic",
        default=None,
        help=(
            "Path to the registry manifest diagnostic for the exact selected "
            "worker image (isaac_worker_image_manifest_diagnostic.v2). Paid "
            "launches of a digest-pinned image fail closed unless this explicit "
            "CLI binding is present and matches the selected digest."
        ),
    )
    parser.add_argument("--wam-consistency-command", default=None)
    parser.add_argument("--require-generated-video-success-label", action="store_true")
    parser.add_argument("--wam-success-label-command", default=None)
    parser.add_argument("--allow-wam-success-labeling", action="store_true")
    parser.add_argument(
        "--wam-success-label-timeout-seconds",
        type=float,
        default=defaults["wam_success_label_timeout_seconds"],
    )
    return parser

DEFAULT_MIN_TASK_COMPLETION_STEPS = DEFAULT_MIN_TASK_ADAPTIVE_STEPS
JOB_MANIFEST_FILENAME = "groot_oscar_digitalocean_closed_loop_job_manifest.json"


def runtime_contract_for_pre_spend(progress_stall_phases: Sequence[str]) -> dict[str, Any]:
    return {
        "startup_marker": "container_bash_started",
        "progress_marker": "bootstrap.json",
        "startup_timeout_seconds": 900,
        "no_progress_timeout_seconds": 900,
        "progress_stall_phases": list(progress_stall_phases),
    }

def _episode_length_contract(
    *,
    steps_cap: int | None,
    stop_on_task_completion: bool,
    min_steps_before_task_completion: int = DEFAULT_MIN_TASK_COMPLETION_STEPS,
    oscar_num_frames_arg: int | None = None,
) -> dict[str, Any]:
    return {
        "episode_length_unit": "closed_loop_control_steps",
        "stop_condition": "task_completion_or_step_cap",
        "steps_cap": steps_cap,
        "min_steps_before_task_completion": int(min_steps_before_task_completion),
        "steps_is_safety_cap": True,
        "stop_on_task_completion": bool(stop_on_task_completion),
        "oscar_num_frames_arg": oscar_num_frames_arg,
        "oscar_num_frames_scope": "per_generation_clip_not_episode_limit",
        "episode_not_bound_to_oscar_clip_frames": bool(
            stop_on_task_completion and steps_cap
        ),
    }


def _string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64_text(text: str) -> str:
    return _b64_bytes(text.encode("utf-8"))


def _json_b64(payload: Mapping[str, Any]) -> str:
    return _b64_text(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_kitchen_members(
    archive: zipfile.ZipFile,
) -> list[tuple[zipfile.ZipInfo, str]]:
    files = [member for member in archive.infolist() if not member.is_dir()]
    stage_candidates = [
        Path(member.filename)
        for member in files
        if Path(member.filename).name == "KitchenRoom.usd"
        and ".thumbs" not in Path(member.filename).parts
    ]
    if len(stage_candidates) != 1:
        raise ValueError("kitchen archive requires one canonical KitchenRoom.usd")
    prefix = stage_candidates[0].parent
    normalized: list[tuple[zipfile.ZipInfo, str]] = []
    for member in files:
        rel = Path(member.filename)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe kitchen asset archive member")
        try:
            stripped = rel.relative_to(prefix)
        except ValueError as exc:
            raise ValueError("kitchen archive member outside canonical root") from exc
        normalized.append((member, f"kitchen/{stripped.as_posix()}"))
    return normalized


def _normalized_kitchen_inventory(
    archive: zipfile.ZipFile,
    members: Sequence[tuple[zipfile.ZipInfo, str]],
    *,
    archive_sha256: str,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for member, target in members:
        data = archive.read(member)
        files.append(
            {
                "path": str(Path(target).relative_to("kitchen")),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    files.sort(key=lambda row: row["path"])
    if "KitchenRoom.usd" not in {row["path"] for row in files}:
        raise ValueError("normalized kitchen inventory missing KitchenRoom.usd")
    return {
        "schema_version": "kitchen_asset_inventory_checksums.v1",
        "main_usd": "KitchenRoom.usd",
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "archive_sha256": archive_sha256,
        "files": files,
        "normalization": {
            "tree_root": "kitchen",
            "canonical_stage": "kitchen/KitchenRoom.usd",
        },
    }


def _write_payload_bundle(
    *,
    payload_zip: Path,
    plan: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    seed_path: Path,
    task_prompt: str,
    seed_provenance: Mapping[str, Any] | None,
    task_success_contract_path: str | Path,
    kitchen_asset_archive_path: str | Path,
) -> Path:
    """Write the immutable inner payload hashed by the attempt manifest."""
    ensure_dir(payload_zip.parent)
    provenance = dict(seed_provenance or {})
    manifest = {
        "schema_version": "groot_oscar_closed_loop_payload_bundle.v1",
        "seed_filename": "initial_policy_frame.png",
        "route_filename": "route.json",
        "task_success_contract_filename": "task_success_contract.json",
        "task_prompt": task_prompt,
        "sealed_launch_plan": dict(plan),
        "seed_provenance": provenance,
        "source_tree_identity": build_source_tree_identity(Path(__file__).resolve().parents[2]),
        "compatibility": build_bundle_compatibility(),
        "self_referential_attempt_manifest_excluded": True,
        "input_artifact_sha256": {
            "initial_policy_frame": _sha256(seed_path),
            "route_payload": hashlib.sha256(
                json.dumps(dict(route_payload), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "sealed_launch_plan": hashlib.sha256(
                json.dumps(dict(plan), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "task_prompt": hashlib.sha256(task_prompt.encode()).hexdigest(),
            "task_success_contract": _sha256(Path(task_success_contract_path)),
            "kitchen_asset_archive": _sha256(Path(kitchen_asset_archive_path)),
        },
    }
    with zipfile.ZipFile(payload_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(seed_path, "initial_policy_frame.png")
        zf.writestr("route.json", json.dumps(dict(route_payload), indent=2))
        zf.writestr("task_prompt.txt", task_prompt)
        zf.writestr("sealed_launch_plan.json", json.dumps(dict(plan), indent=2))
        zf.writestr("seed_provenance.json", json.dumps(provenance, indent=2))
        zf.write(task_success_contract_path, "task_success_contract.json")
        with zipfile.ZipFile(kitchen_asset_archive_path) as kitchen_archive:
            normalized_members = _normalized_kitchen_members(kitchen_archive)
            normalized_inventory = _normalized_kitchen_inventory(
                kitchen_archive,
                normalized_members,
                archive_sha256=_sha256(Path(kitchen_asset_archive_path)),
            )
            for member, target in normalized_members:
                zf.writestr(target, kitchen_archive.read(member))
        zf.writestr(
            "kitchen_asset_inventory_checksums.json",
            json.dumps(normalized_inventory, indent=2),
        )
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    return payload_zip


def _attempt_payload_bundle_ref(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    attempt = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(dict(attempt.get("artifacts") or {}).get("bundle") or {})


def _write_input_bundle(
    *,
    bundle_zip: Path,
    plan: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    seed_path: Path,
    task_prompt: str,
    seed_provenance: Mapping[str, Any] | None = None,
    task_success_contract_path: str | Path | None = None,
    attempt_input_manifest_path: str | Path | None = None,
    kitchen_asset_archive_path: str | Path | None = None,
) -> Path:
    ensure_dir(bundle_zip.parent)
    provenance = dict(seed_provenance or {})
    if attempt_input_manifest_path and _attempt_payload_bundle_ref(
        attempt_input_manifest_path
    ):
        attempt = json.loads(Path(attempt_input_manifest_path).read_text(encoding="utf-8"))
        bundle_ref = dict(dict(attempt.get("artifacts") or {}).get("bundle") or {})
        payload_zip = Path(str(bundle_ref.get("path") or "")).expanduser().resolve()
        if not payload_zip.is_file() or _sha256(payload_zip) != bundle_ref.get("sha256"):
            raise ValueError("attempt_input_manifest_payload_bundle_digest_mismatch")
        with zipfile.ZipFile(payload_zip) as payload_archive:
            payload_names = set(payload_archive.namelist())
            required = {
                "initial_policy_frame.png",
                "route.json",
                "task_prompt.txt",
                "sealed_launch_plan.json",
                "seed_provenance.json",
                "task_success_contract.json",
                "bundle_manifest.json",
                "kitchen_asset_inventory_checksums.json",
                "kitchen/KitchenRoom.usd",
            }
            if not required <= payload_names:
                raise ValueError("attempt_payload_bundle_required_files_missing")
            payload_members = {
                name: payload_archive.read(name)
                for name in payload_archive.namelist()
                if not name.endswith("/")
            }
        payload_manifest = json.loads(payload_members["bundle_manifest.json"])
        observed_inputs = dict(payload_manifest.get("input_artifact_sha256") or {})
        expected_inputs = {
            "initial_policy_frame": _sha256(seed_path),
            "route_payload": hashlib.sha256(
                json.dumps(dict(route_payload), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "sealed_launch_plan": hashlib.sha256(
                json.dumps(dict(plan), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "task_prompt": hashlib.sha256(task_prompt.encode()).hexdigest(),
            "task_success_contract": _sha256(Path(task_success_contract_path)),
            "kitchen_asset_archive": _sha256(Path(kitchen_asset_archive_path)),
        }
        if observed_inputs != expected_inputs:
            raise ValueError("attempt_payload_bundle_inputs_mismatch")
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in payload_members.items():
                zf.writestr(name, data)
            zf.write(payload_zip, "immutable_payload_bundle.zip")
            zf.write(attempt_input_manifest_path, "attempt_input_manifest.json")
            worker_evidence_ref = dict(
                dict(attempt.get("artifacts") or {}).get(
                    "worker_image_runtime_evidence"
                )
                or {}
            )
            worker_evidence_path = Path(
                str(worker_evidence_ref.get("path") or "")
            ).expanduser().resolve()
            if (
                not worker_evidence_path.is_file()
                or _sha256(worker_evidence_path) != worker_evidence_ref.get("sha256")
            ):
                raise ValueError("attempt_worker_image_runtime_evidence_digest_mismatch")
            zf.write(
                worker_evidence_path,
                "worker_image_runtime_evidence.json",
            )
            zf.writestr(
                "transport_envelope_manifest.json",
                json.dumps(
                    {
                        "schema_version": "groot_oscar_closed_loop_transport_envelope.v1",
                        "payload_bundle_sha256": _sha256(payload_zip),
                        "attempt_input_manifest_sha256": _sha256(
                            Path(attempt_input_manifest_path)
                        ),
                        "payload_bundle_is_attempt_manifest_bundle_identity": True,
                    },
                    indent=2,
                ),
            )
        return bundle_zip
    manifest = {
        "schema_version": "groot_oscar_closed_loop_input_bundle.v2",
        "generated_at": utc_now_iso(),
        "seed_filename": "initial_policy_frame.png",
        "route_filename": "route.json",
        "seed_provenance_filename": "seed_provenance.json",
        "task_prompt": task_prompt,
        "sealed_launch_plan": dict(plan),
        "seed_provenance": provenance,
        "source_tree_identity": build_source_tree_identity(
            Path(__file__).resolve().parents[2]
        ),
        "compatibility": build_bundle_compatibility(),
        "claim_boundary": (
            "Input bundle stages the requested closed-loop prompt, route, and "
            "seed frame. It is not WAM generation, manipulation success, "
            "forward/inverse consistency, or physical robot proof."
        ),
    }
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(seed_path, "initial_policy_frame.png")
        zf.writestr("route.json", json.dumps(dict(route_payload), indent=2))
        zf.writestr("task_prompt.txt", task_prompt)
        zf.writestr("sealed_launch_plan.json", json.dumps(dict(plan), indent=2))
        zf.writestr("seed_provenance.json", json.dumps(provenance, indent=2))
        if task_success_contract_path:
            zf.write(task_success_contract_path, "task_success_contract.json")
        if attempt_input_manifest_path:
            zf.write(attempt_input_manifest_path, "attempt_input_manifest.json")
        if kitchen_asset_archive_path:
            with zipfile.ZipFile(kitchen_asset_archive_path) as kitchen_archive:
                normalized_members = _normalized_kitchen_members(kitchen_archive)
                normalized_inventory = _normalized_kitchen_inventory(
                    kitchen_archive,
                    normalized_members,
                    archive_sha256=_sha256(Path(kitchen_asset_archive_path)),
                )
                for member, target in normalized_members:
                    zf.writestr(target, kitchen_archive.read(member))
                zf.writestr(
                    "kitchen_asset_inventory_checksums.json",
                    json.dumps(normalized_inventory, indent=2),
                )
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    return bundle_zip


def _write_job_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    write_json(out_dir / JOB_MANIFEST_FILENAME, payload)
    return payload


def build_launch_spec(
    *,
    job_dir: Path,
    image_ref: str,
    start_frame: Path,
    route_payload: Mapping[str, Any],
    task_prompt: str,
    plan: Mapping[str, Any],
    launch_nonce: str,
    seed_provenance: Mapping[str, Any] | None = None,
    container_disk_gb: int = 220,
    volume_gb: int = 120,
    max_hourly_rate_usd: float = 3.5,
) -> RenderLaunchSpec:
    """Build the provider-neutral sealed-worker launch shape."""
    from .groot_oscar_digitalocean_closed_loop_job import (
        DEFAULT_MIN_GPU_RAM_MB,
        build_worker_bootstrap_script,
    )

    put_url = (job_dir / "provider_output_put_url.txt").read_text(encoding="utf-8").strip()
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
    env = {
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "CUDA_VISIBLE_DEVICES": "0",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "BLUEPRINT_INITIAL_POLICY_FRAME_B64": _b64_bytes(start_frame.read_bytes()),
        "BLUEPRINT_ROUTE_JSON_B64": _b64_text(json.dumps(dict(route_payload), indent=2)),
        "BLUEPRINT_TASK_PROMPT": task_prompt,
        "BLUEPRINT_SEALED_LAUNCH_PLAN_B64": _json_b64(plan),
        "BLUEPRINT_SEED_PROVENANCE_B64": _json_b64(seed_provenance or {}),
        "BLUEPRINT_LAUNCH_SESSION_ID": launch_nonce,
        "BLUEPRINT_WORKER_IMAGE_DIGEST": image_ref,
        SEALED_CONFIRMED_ENV: "true",
    }
    return RenderLaunchSpec(
        name="blueprint-groot-oscar-closed-loop",
        image=image_ref,
        env=env,
        bootstrap_argv=["-lc", build_worker_bootstrap_script(plan)],
        entrypoint=["bash"],
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        max_hourly_rate_usd=max_hourly_rate_usd,
        min_gpu_ram_mb=DEFAULT_MIN_GPU_RAM_MB,
    )


def _read_json_mapping(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def validate_persistent_isaac_route_start_pose(
    route_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the robot pose needed before a paid persistent executor starts."""
    route_points = list(route_payload.get("route_points") or [])
    try:
        start_pose = [float(value) for value in route_points[0]]
        start_yaw = float(route_payload.get("accepted_stance_yaw_rad"))
    except (IndexError, TypeError, ValueError):
        return {"status": "blocked", "blockers": ["persistent_isaac_route_start_pose_invalid"]}
    if len(start_pose) != 3 or not all(
        math.isfinite(value) for value in [*start_pose, start_yaw]
    ):
        return {"status": "blocked", "blockers": ["persistent_isaac_route_start_pose_invalid"]}
    return {
        "status": "passed",
        "blockers": [],
        "xyz": start_pose,
        "yaw_rad": start_yaw,
        "validated_before_paid_provider_mutation": True,
    }


def _read_json_mapping_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json_mapping(path)
    except Exception:  # noqa: BLE001
        return {}


def _argv_value(argv: Sequence[Any], flag: str) -> str | None:
    values = [str(item) for item in argv]
    try:
        idx = values.index(flag)
    except ValueError:
        return None
    next_idx = idx + 1
    if next_idx >= len(values):
        return None
    return values[next_idx]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
