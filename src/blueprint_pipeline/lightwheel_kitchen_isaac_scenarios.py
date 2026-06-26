"""Build a fail-closed Isaac Sim scenario packet for Lightwheel Kitchen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

from .common import ensure_dir, parse_gs_uri, utc_now_iso, write_json


LIGHTWHEEL_KITCHEN_ISAAC_SCHEMA_VERSION = "lightwheel_kitchen_isaac_scenarios.v1"
LIGHTWHEEL_REPO_URL = "https://github.com/LightwheelAI/Lightwheel_Kitchen"
LIGHTWHEEL_DEFAULT_COMMIT = "10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1"
LIGHTWHEEL_LICENSE = "CC-BY-NC-4.0"
MAIN_USD_RELATIVE = "Collected_KitchenRoom/KitchenRoom.usd"
THUMBNAIL_RELATIVE = "Collected_KitchenRoom/.thumbs/256x256/KitchenRoom.usd.png"
OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE = "Isaac/Robots/Unitree/G1/g1.usd"
OFFICIAL_ISAAC_UNITREE_G1_DOC_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_robots.html"
)
DEFAULT_OUTPUT_RELATIVE = "pipeline/lightwheel_kitchen_isaac_scenarios"
ISAAC_EXECUTION_MANIFEST_NAME = "lightwheel_kitchen_isaac_execution_manifest.json"
PER_SCENARIO_RESULTS_NAME = "per_scenario_results.json"
CONTACT_COLLISION_MANIFEST_NAME = "contact_collision_manifest.json"
VIDEO_ARTIFACT_CHECKS_NAME = "video_artifact_checks.json"
FINAL_READINESS_MANIFEST_NAME = "lightwheel_kitchen_isaac_final_readiness_manifest.json"
PROVIDER_PACKET_NAME = "lightwheel_kitchen_isaac_provider_packet.json"
PROVIDER_BUNDLE_NAME = "lightwheel_kitchen_isaac_provider_bundle.zip"
PROVIDER_RESULT_NAME = "lightwheel_kitchen_isaac_provider_runtime_result.json"
PROVIDER_REQUEST_NAME = "isaac_execution_request.provider.json"
PROVIDER_PRIVATE_ENV_NAME = "lightwheel_kitchen_isaac_provider_private_env.local"
RUNPOD_DIRECT_LAUNCH_REQUEST_NAME = "lightwheel_kitchen_isaac_runpod_direct_launch_request.json"
RUNPOD_DIRECT_LAUNCH_REQUEST_LOCAL_NAME = (
    "lightwheel_kitchen_isaac_runpod_direct_launch_request.local.json"
)
UNITREE_G1_MJCF_BUNDLE_RELATIVE = "robot_assets/mujoco_menagerie/unitree_g1"
DEFAULT_ISAAC_RUNTIME_IMAGE_REF = "nvcr.io/nvidia/isaac-sim:6.0.0"
ISAAC_RUNTIME_IMAGE_REF_ENV = "BLUEPRINT_LIGHTWHEEL_ISAAC_RUNTIME_IMAGE_REF"
RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV = "BLUEPRINT_RUNPOD_CONTAINER_REGISTRY_AUTH_ID"
NVIDIA_ISAAC_REQUIREMENTS_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html"
)
NVIDIA_ISAAC_CONTAINER_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html"
)
ISAAC_LAB_INSTALL_URL = "https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
ISAAC_LAB_DOCKER_URL = "https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, json.loads(json.dumps(payload, sort_keys=True)))


def _read_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _redact_signed_url_text(text: str) -> str:
    signature_param = "x-goog-" + "signature"
    return re.sub(
        rf"(?i){re.escape(signature_param)}=[^&\s\"']+",
        "x-goog-redacted-signature-param=<redacted:signed-url-signature>",
        text,
    )


def _redacted_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_signed_url_text(value)
    if isinstance(value, list):
        return [_redacted_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_redacted_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redacted_jsonable(item) for key, item in value.items()}
    return value


def _uri_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts if part)])


def _provider_uri_is_fetchable_without_extra_credentials(uri: str) -> bool:
    if not uri:
        return False
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        lowered = uri.lower()
        signature_param = "x-goog-" + "signature="
        return signature_param in lowered or "x-amz-signature=" in lowered
    return False


def _provider_uri_requires_storage_credentials(uri: str) -> bool:
    return urlparse(uri).scheme in {"gs", "s3", "r2"}


def _classify_upload_error(*, scheme: str, error: BaseException) -> str:
    text = str(error).lower()
    if scheme == "gs" and "billing account" in text and (
        "disabled" in text or "absent" in text or "accountdisabled" in text
    ):
        return "upload_failed:gs_billing_account_disabled"
    if "accessdenied" in text or "access denied" in text:
        return f"upload_failed:{scheme}_access_denied"
    if "forbidden" in text or "403" in text:
        return f"upload_failed:{scheme}_forbidden"
    return f"upload_failed:{type(error).__name__}"


def _upload_provider_file(source: Path, destination_uri: str) -> dict[str, Any]:
    parsed_uri = urlparse(destination_uri)
    scheme = parsed_uri.scheme or "file"
    try:
        if scheme == "gs":
            try:
                from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - environment dependent.
                raise RuntimeError("google-cloud-storage is required for gs:// uploads") from exc
            parsed = parse_gs_uri(destination_uri)
            client = gcs_storage.Client()
            client.bucket(parsed.bucket).blob(parsed.key).upload_from_filename(str(source))
        elif scheme == "file" or parsed_uri.scheme == "":
            destination = Path(parsed_uri.path if parsed_uri.scheme else destination_uri).expanduser()
            if not destination.is_absolute():
                destination = destination.resolve()
            ensure_dir(destination.parent)
            destination.write_bytes(source.read_bytes())
            destination_uri = str(destination)
        else:
            return {
                "status": "blocked",
                "source": str(source),
                "destination_uri": _redact_signed_url_text(destination_uri),
                "storage_scheme": scheme,
                "blockers": [f"unsupported_provider_packet_upload_scheme:{scheme}"],
                "raw_secret_values_recorded": False,
            }
    except Exception as exc:
        return {
            "status": "blocked",
            "source": str(source),
            "destination_uri": _redact_signed_url_text(destination_uri),
            "storage_scheme": scheme,
            "blockers": [_classify_upload_error(scheme=scheme, error=exc)],
            "error": _redact_signed_url_text(str(exc)),
            "raw_secret_values_recorded": False,
        }
    return {
        "status": "uploaded",
        "source": str(source),
        "destination_uri": _redact_signed_url_text(destination_uri),
        "storage_scheme": scheme,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
        "raw_secret_values_recorded": False,
    }


def _signed_url_from_private_inputs(
    private_inputs: Mapping[str, Any],
    key: str,
) -> str:
    value = private_inputs.get(key)
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return _string(value[0].get("signed_url"))
    if isinstance(value, Mapping):
        return _string(value.get("signed_url"))
    return ""


def _signed_url_entry_summary(
    *,
    private_inputs: Mapping[str, Any],
    key: str,
    private_input_path: Path | None,
) -> dict[str, Any]:
    value = private_inputs.get(key)
    row: Mapping[str, Any] = {}
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        row = value[0]
    elif isinstance(value, Mapping):
        row = value
    signed_url = _string(row.get("signed_url"))
    signature_param = "x-goog-" + "signature="
    return {
        "private_input_key": key,
        "private_input_path": str(private_input_path) if private_input_path else None,
        "signed_url_present": bool(signed_url),
        "signed_url_signature_present": signature_param in signed_url.lower(),
        "redacted_signed_url": _redact_signed_url_text(signed_url) if signed_url else None,
        "resource": row.get("resource"),
        "http_verb": row.get("http_verb"),
        "expiration": row.get("expiration"),
        "raw_signed_url_recorded_in_manifest": False,
    }


def _write_provider_private_env(
    *,
    output_dir: Path,
    private_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_url = _signed_url_from_private_inputs(private_inputs, "bundle_get")
    packet_url = _signed_url_from_private_inputs(private_inputs, "packet_get")
    runtime_result_put_url = _signed_url_from_private_inputs(private_inputs, "runtime_result_put")
    if not (bundle_url or packet_url or runtime_result_put_url):
        return {"status": "not_written", "reason": "no_private_signed_urls_present"}
    env_path = output_dir / PROVIDER_PRIVATE_ENV_NAME
    lines = [
        "# Local private provider inputs. Do not commit or paste.",
        "# Contains signed URLs; proof manifests store only redacted evidence.",
    ]
    values = {
        "BLUEPRINT_LIGHTWHEEL_PROVIDER_BUNDLE_URL": bundle_url,
        "BLUEPRINT_LIGHTWHEEL_PROVIDER_PACKET_URL": packet_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": runtime_result_put_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_CONTENT_TYPE": "application/json",
    }
    for key, value in values.items():
        if value:
            escaped = value.replace("'", "'\"'\"'")
            lines.append(f"export {key}='{escaped}'")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "written",
        "path": str(env_path),
        "env_vars": [key for key, value in values.items() if value],
        "raw_signed_urls_recorded_in_proof_manifest": False,
    }


def _resolve_unitree_g1_mjcf_root(
    *,
    explicit_root: str | Path | None,
) -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        explicit_root,
        os.environ.get("BLUEPRINT_MUJOCO_G1_MODEL_ROOT"),
        repo_root / "output" / "external_assets" / "mujoco_menagerie" / "unitree_g1",
        Path.cwd() / "output" / "external_assets" / "mujoco_menagerie" / "unitree_g1",
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if (path / "g1.xml").is_file():
            return path
    return None


def _unitree_g1_mjcf_summary(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "status": "not_found",
            "g1_xml_present": False,
            "blockers": ["unitree_g1_mjcf_source_not_materialized"],
        }
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    xml_files = [path.relative_to(root).as_posix() for path in files if path.suffix.lower() == ".xml"]
    stl_files = [path.relative_to(root).as_posix() for path in files if path.suffix.lower() == ".stl"]
    g1_xml = root / "g1.xml"
    return {
        "status": "materialized",
        "root": str(root),
        "bundle_relative_root": UNITREE_G1_MJCF_BUNDLE_RELATIVE,
        "g1_xml": str(g1_xml),
        "g1_xml_bundle_path": f"{UNITREE_G1_MJCF_BUNDLE_RELATIVE}/g1.xml",
        "g1_xml_present": g1_xml.is_file(),
        "g1_xml_sha256": _sha256(g1_xml) if g1_xml.is_file() else None,
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "xml_files": xml_files,
        "stl_file_count": len(stl_files),
        "license_present": (root / "LICENSE").is_file(),
        "license_path": str(root / "LICENSE") if (root / "LICENSE").is_file() else None,
        "source": "google-deepmind/mujoco_menagerie unitree_g1",
        "isaac_import_status": "not_run_requires_isaac_mjcf_importer_in_provider_runtime",
        "blockers": ["unitree_g1_mjcf_asset_bundled_but_isaac_importer_binding_unverified"],
    }


def _write_provider_request(
    *,
    path: Path,
    scenarios: Sequence[Mapping[str, Any]],
    unitree_g1_mjcf: Mapping[str, Any],
) -> dict[str, Any]:
    request = {
        "schema_version": "lightwheel_kitchen_isaac_provider_execution_request.v1",
        "bundle_layout": {
            "source_zip_path": "source/Lightwheel_Kitchen.zip",
            "assets_extract_dir": "runtime_assets",
            "scene_usd_relative": MAIN_USD_RELATIVE,
            "runtime_result_path": PROVIDER_RESULT_NAME,
        },
        "requested_outputs": {
            "overview_videos": True,
            "robot_pov_videos": True,
            "per_scenario_trace_jsonl": True,
            "contacts_and_collision_events": True,
        },
        "scenarios": list(scenarios),
        "robot_binding": {
            "robot_profile_id": "unitree_g1",
            "preferred_verified_isaac_usd_env": "BLUEPRINT_ISAAC_UNITREE_G1_USD",
            "official_isaac_asset_candidate": {
                "source": "NVIDIA Isaac Sim 5.0 Robot Assets documentation",
                "doc_url": OFFICIAL_ISAAC_UNITREE_G1_DOC_URL,
                "assets_root_relative_path": OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE,
                "content_browser_path": "Robots/Unitree/G1/g1.usd",
                "runtime_resolution": (
                    "isaacsim.storage.native.get_assets_root_path() + "
                    f"'/ {OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE}'"
                ).replace("/ ", "/"),
            },
            "fallback_mjcf_importer_source": {
                "source": "google-deepmind/mujoco_menagerie unitree_g1/g1.xml",
                "status": unitree_g1_mjcf.get("status"),
                "bundle_g1_xml_path": unitree_g1_mjcf.get("g1_xml_bundle_path"),
                "g1_xml_sha256": unitree_g1_mjcf.get("g1_xml_sha256"),
            },
            "current_status": "preferred_official_isaac_g1_usd_pending_runtime_resolution",
        },
        "execution_contract": {
            "navigation_policy_type": "open_loop_waypoint_following_until_obstacle_aware_planner_is_bound",
            "success_gates": [
                "target reached within 0.50 m",
                "no unrecovered fall state",
                "no persistent fixture collision violation",
                "overview and robot POV videos exist and pass artifact checks",
            ],
            "do_not_claim_navigation_success_without_gates": True,
        },
    }
    _safe_write_json(path, request)
    return request


def _write_provider_bundle(
    *,
    output_dir: Path,
    source_zip: Path,
    unitree_g1_mjcf_root: Path | None,
    runner_script: Path,
    local_execution_request_path: Path,
    provider_execution_request_path: Path,
    scenario_manifest_path: Path,
    runtime_preflight_path: Path,
    handoff_path: Path,
) -> dict[str, Any]:
    bundle_path = output_dir / PROVIDER_BUNDLE_NAME
    members = [
        (source_zip, "source/Lightwheel_Kitchen.zip", zipfile.ZIP_STORED),
        (runner_script, f"runner/{runner_script.name}", zipfile.ZIP_DEFLATED),
        (local_execution_request_path, f"request/{local_execution_request_path.name}", zipfile.ZIP_DEFLATED),
        (provider_execution_request_path, f"request/{provider_execution_request_path.name}", zipfile.ZIP_DEFLATED),
        (scenario_manifest_path, f"manifests/{scenario_manifest_path.name}", zipfile.ZIP_DEFLATED),
        (runtime_preflight_path, f"manifests/{runtime_preflight_path.name}", zipfile.ZIP_DEFLATED),
        (handoff_path, f"manifests/{handoff_path.name}", zipfile.ZIP_DEFLATED),
    ]
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for source, arcname, compression in members:
            if source.is_file():
                archive.write(source, arcname, compress_type=compression)
        if unitree_g1_mjcf_root is not None:
            for source in sorted(unitree_g1_mjcf_root.rglob("*")):
                if source.is_file():
                    archive.write(
                        source,
                        f"{UNITREE_G1_MJCF_BUNDLE_RELATIVE}/{source.relative_to(unitree_g1_mjcf_root).as_posix()}",
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
    return {
        "status": "created",
        "path": str(bundle_path),
        "name": PROVIDER_BUNDLE_NAME,
        "format": "zip",
        "size_bytes": bundle_path.stat().st_size,
        "sha256": _sha256(bundle_path),
        "member_count": len(names),
        "members": names,
        "contains_lightwheel_source_zip": "source/Lightwheel_Kitchen.zip" in names,
        "contains_provider_execution_request": f"request/{PROVIDER_REQUEST_NAME}" in names,
        "contains_runtime_runner": f"runner/{runner_script.name}" in names,
        "contains_unitree_g1_mjcf": f"{UNITREE_G1_MJCF_BUNDLE_RELATIVE}/g1.xml" in names,
        "unitree_g1_mjcf_member_count": sum(
            1 for name in names if name.startswith(f"{UNITREE_G1_MJCF_BUNDLE_RELATIVE}/")
        ),
    }


def _asset_inventory_from_zip(zip_path: Path) -> dict[str, Any]:
    extension_counts: dict[str, int] = {}
    sample_assets: list[dict[str, Any]] = []
    total_size = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            total_size += int(info.file_size)
            if info.is_dir():
                continue
            filename = Path(info.filename).name
            suffix = Path(filename).suffix.lower() or "<no_extension>"
            extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
            if suffix in {".usd", ".usda", ".usdc", ".obj", ".glb", ".gltf", ".xml", ".mjcf", ".urdf", ".stl", ".ply"}:
                sample_assets.append(
                    {
                        "path": info.filename,
                        "size_bytes": int(info.file_size),
                        "extension": suffix,
                    }
                )
    mujoco_native_extensions = {".xml", ".mjcf", ".urdf", ".obj", ".glb", ".gltf", ".stl"}
    has_mujoco_native = any(extension_counts.get(ext, 0) for ext in mujoco_native_extensions)
    return {
        "source_zip": str(zip_path),
        "source_zip_sha256": _sha256(zip_path),
        "source_zip_size_bytes": zip_path.stat().st_size,
        "uncompressed_size_bytes": total_size,
        "extension_counts": dict(sorted(extension_counts.items())),
        "sample_scene_assets": sample_assets[:80],
        "main_usd_present": any(item["path"] == MAIN_USD_RELATIVE for item in sample_assets),
        "mujoco_native_asset_present": has_mujoco_native,
        "mujoco_native_extensions_checked": sorted(mujoco_native_extensions),
    }


def _materialize_assets(*, source_zip: Path, asset_output_dir: Path) -> Path:
    ensure_dir(asset_output_dir)
    main_usd = asset_output_dir / MAIN_USD_RELATIVE
    if main_usd.is_file():
        return asset_output_dir
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(asset_output_dir)
    if not main_usd.is_file():
        raise FileNotFoundError(f"Lightwheel Kitchen archive did not extract {MAIN_USD_RELATIVE}")
    return asset_output_dir


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _command_path(name: str) -> str | None:
    return shutil.which(name)


def _run_command(command: Sequence[str], *, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"status": "not_available", "command": list(command), "reason": "command_not_found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "command": list(command),
            "reason": "timeout",
            "timeout_seconds": timeout,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_head": completed.stdout[:4000],
        "stdout_tail": completed.stdout[-4000:],
        "stderr_head": completed.stderr[:4000],
        "stderr_tail": completed.stderr[-4000:],
    }


def _usd_stage_summary(usd_path: Path) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency varies by host.
        return {"status": "not_checked", "reason": "pxr_unavailable", "error": repr(exc)}

    try:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            return {"status": "failed", "reason": "stage_open_returned_none"}
        prims = list(stage.Traverse())
        type_counts: dict[str, int] = {}
        for prim in prims:
            type_name = prim.GetTypeName()
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "proxy"],
            useExtentsHint=True,
        )
        bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        default_prim = stage.GetDefaultPrim()
        return {
            "status": "opened",
            "path": str(usd_path),
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
            "prim_count": len(prims),
            "mesh_count": sum(1 for prim in prims if prim.IsA(UsdGeom.Mesh)),
            "xformable_count": sum(1 for prim in prims if prim.IsA(UsdGeom.Xformable)),
            "type_counts_top": [
                {"type": key, "count": value}
                for key, value in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)[
                    :20
                ]
            ],
            "bbox_min": [round(float(value), 4) for value in bbox.GetMin()],
            "bbox_max": [round(float(value), 4) for value in bbox.GetMax()],
        }
    except Exception as exc:
        return {"status": "failed", "path": str(usd_path), "error": repr(exc)}


def _usdchecker_summary(usd_path: Path) -> dict[str, Any]:
    usdchecker = _command_path("usdchecker")
    if not usdchecker:
        return {"status": "not_checked", "reason": "usdchecker_unavailable"}
    result = _run_command([usdchecker, str(usd_path)], timeout=180)
    blockers: list[str] = []
    combined = f"{result.get('stdout_head', '')}\n{result.get('stderr_head', '')}"
    if result["status"] != "passed":
        if "Could not load sublayer" in combined:
            blockers.append("usd_missing_sublayer_metrics_assembler")
        if "OmniPBR.mdl" in combined or "OmniGlass.mdl" in combined:
            blockers.append("usd_requires_omniverse_mdl_material_registry")
        if "UnresolvableDependency" in combined:
            blockers.append("usd_unresolvable_dependencies")
        if "ShaderSdrCompliance" in combined:
            blockers.append("usd_shader_registry_or_type_compliance_errors")
    return {**result, "blockers": blockers}


def _usd_dependency_presence_audit(
    *,
    asset_root: Path,
    source_zip: Path,
    usdchecker: Mapping[str, Any],
) -> dict[str, Any]:
    combined = json.dumps(usdchecker, sort_keys=True)
    referenced_names = _dedupe(
        [
            *re.findall(r"UnitsAdjust-[A-Za-z0-9]+\.metricsAssembler", combined),
            *re.findall(r"Omni(?:PBR|Glass)\.mdl", combined),
            *re.findall(r"3d66Model-[A-Za-z0-9-]+\.(?:jpg|JPG|png|PNG)", combined),
        ]
    )
    if not referenced_names:
        return {
            "status": "not_applicable",
            "referenced_dependency_count": 0,
            "dependencies": [],
            "blockers": [],
        }
    with zipfile.ZipFile(source_zip) as archive:
        zip_names = archive.namelist()
    dependencies: list[dict[str, Any]] = []
    blockers: list[str] = []
    for name in referenced_names:
        materialized_matches = [str(path) for path in sorted(asset_root.rglob(name))]
        zip_matches = [path for path in zip_names if Path(path).name == name]
        present = bool(materialized_matches or zip_matches)
        dependencies.append(
            {
                "name": name,
                "materialized_present": bool(materialized_matches),
                "source_zip_present": bool(zip_matches),
                "materialized_matches": materialized_matches[:10],
                "source_zip_matches": zip_matches[:10],
                "present": present,
            }
        )
    missing_names = [item["name"] for item in dependencies if item["present"] is not True]
    if any(name.startswith("UnitsAdjust-") for name in missing_names):
        blockers.append("usd_missing_sublayer_metrics_assembler")
    if any(name in {"OmniPBR.mdl", "OmniGlass.mdl"} for name in missing_names):
        blockers.append("usd_requires_omniverse_mdl_material_registry")
    if any(name.startswith("3d66Model-") for name in missing_names):
        blockers.append("usd_unresolvable_texture_dependencies")
    return {
        "status": "missing_required_dependencies" if missing_names else "dependencies_present",
        "referenced_dependency_count": len(referenced_names),
        "missing_dependency_count": len(missing_names),
        "missing_dependency_names": missing_names,
        "dependencies": dependencies,
        "blockers": _dedupe(blockers),
    }


def _runtime_preflight() -> dict[str, Any]:
    tool_names = [
        "isaacsim",
        "isaac-sim",
        "python.sh",
        "usdcat",
        "usdchecker",
        "docker",
        "nvidia-smi",
    ]
    tools = {name: _command_path(name) for name in tool_names}
    modules = {
        "pxr": _module_available("pxr"),
        "omni": _module_available("omni"),
        "isaacsim": _module_available("isaacsim"),
        "isaaclab": _module_available("isaaclab"),
    }
    nvidia_smi = (
        _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            timeout=20,
        )
        if tools["nvidia-smi"]
        else {"status": "not_available", "reason": "nvidia_smi_not_found"}
    )
    docker = {"status": "not_checked", "reason": "docker_unavailable"}
    if tools["docker"]:
        docker_raw = _run_command(
            [
                "docker",
                "info",
                "--format",
                (
                    "{{.Architecture}}\n{{.OperatingSystem}}\n{{.MemTotal}}\n"
                    "{{if index .Runtimes \"nvidia\"}}true{{else}}false{{end}}"
                ),
            ],
            timeout=30,
        )
        docker = {"status": docker_raw["status"], "raw": docker_raw}
        if docker_raw["status"] == "passed":
            try:
                architecture, operating_system, memory_total, nvidia_runtime = (
                    _string(docker_raw.get("stdout_head")).splitlines()[:4]
                )
            except ValueError:
                architecture = operating_system = memory_total = nvidia_runtime = ""
            docker.update(
                {
                    "architecture": architecture or None,
                    "operating_system": operating_system or None,
                    "memory_total_bytes": int(memory_total) if memory_total.isdigit() else None,
                    "nvidia_runtime_present": nvidia_runtime.strip().lower() == "true",
                }
            )
    blockers: list[str] = []
    if not (tools["isaacsim"] or tools["isaac-sim"] or modules["isaacsim"] or modules["omni"]):
        blockers.append("isaac_sim_runtime_unavailable")
    if not modules["isaaclab"]:
        blockers.append("isaac_lab_python_module_unavailable")
    if nvidia_smi.get("status") != "passed":
        blockers.append("nvidia_smi_unavailable")
    if docker.get("architecture") and docker.get("architecture") != "x86_64":
        blockers.append("docker_host_architecture_not_linux_x86_64")
    if docker.get("status") == "passed" and docker.get("nvidia_runtime_present") is not True:
        blockers.append("docker_nvidia_runtime_unavailable")
    return {
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
        "tools": tools,
        "python_modules": modules,
        "nvidia_smi": nvidia_smi,
        "docker": docker,
        "provider_credentials": {
            "runpod_api_key_env_present": bool(os.environ.get("RUNPOD_API_KEY")),
            "runpod_api_key_file_env_present": bool(os.environ.get("RUNPOD_API_KEY_FILE")),
            "runpod_config_file_present": Path("~/.runpod/config.toml").expanduser().is_file(),
            "ngc_api_key_env_present": bool(os.environ.get("NGC_API_KEY")),
            "ngc_api_key_file_env_present": bool(os.environ.get("NGC_API_KEY_FILE")),
            "nvidia_api_key_env_present": bool(os.environ.get("NVIDIA_API_KEY")),
            "nvidia_api_key_file_env_present": bool(os.environ.get("NVIDIA_API_KEY_FILE")),
            "raw_secret_values_recorded": False,
        },
        "runtime_requirement_sources": {
            "nvidia_isaac_sim_requirements": NVIDIA_ISAAC_REQUIREMENTS_URL,
            "nvidia_isaac_sim_container_install": NVIDIA_ISAAC_CONTAINER_URL,
            "isaac_lab_install": ISAAC_LAB_INSTALL_URL,
            "isaac_lab_docker": ISAAC_LAB_DOCKER_URL,
        },
        "blockers": blockers,
        "isaac_local_runtime_ready": not blockers,
    }


def _default_scenarios() -> list[dict[str, Any]]:
    base = [
        (
            "entry_to_sink",
            "Navigate from the open entry side to the sink work area.",
            [-4.25, -3.35, 0.05],
            [2.20, 0.90, 0.05],
            [[-2.00, -2.55, 0.05], [0.30, -1.30, 0.05], [1.80, -0.10, 0.05]],
        ),
        (
            "fridge_to_stovetop",
            "Start near the refrigerator and stop at the stovetop.",
            [-2.80, -1.90, 0.05],
            [0.55, 1.45, 0.05],
            [[-1.70, -0.95, 0.05], [-0.35, 0.05, 0.05], [0.35, 0.85, 0.05]],
        ),
        (
            "island_counter_loop",
            "Cross the open floor and route around the island counter.",
            [4.35, -3.10, 0.05],
            [-1.05, 0.75, 0.05],
            [[3.20, -1.80, 0.05], [1.65, -0.70, 0.05], [0.10, 0.10, 0.05]],
        ),
        (
            "top_cabinet_inspection",
            "Approach the top cabinet wall for an inspection stop.",
            [4.90, 2.80, 0.05],
            [-1.20, 2.05, 0.05],
            [[3.40, 2.10, 0.05], [1.60, 1.70, 0.05], [0.00, 1.85, 0.05]],
        ),
        (
            "narrow_passage_to_sink",
            "Thread a narrow passage toward the sink without contact violations.",
            [0.00, -3.75, 0.05],
            [2.35, 1.25, 0.05],
            [[0.70, -2.10, 0.05], [1.40, -0.70, 0.05], [2.05, 0.45, 0.05]],
        ),
    ]
    scenarios: list[dict[str, Any]] = []
    for index, (scenario_id, description, spawn, target, waypoints) in enumerate(base, start=1):
        scenarios.append(
            {
                "scenario_id": f"lightwheel_kitchen_g1_{index:02d}_{scenario_id}",
                "scene_id": "lightwheel_kitchen",
                "robot_profile_id": "unitree_g1",
                "task_family": "locomotion_navigation",
                "description": description,
                "spawn_position_xyz": spawn,
                "target_position_xyz": target,
                "waypoints_xyz": waypoints,
                "success_criteria": [
                    "robot base reaches target_position_xyz within 0.50 m",
                    "no unrecovered fall state",
                    "no persistent collision violation with kitchen fixtures",
                    "simulated robot POV and overview video recorded",
                ],
                "scenario_status": "specified_not_executed",
                "execution_proven": False,
                "navigation_policy_boundary": (
                    "These are scenario specifications. They do not prove obstacle-aware "
                    "navigation until executed inside Isaac Sim/Lab with a robot asset and "
                    "controller/planner bound to the scenario."
                ),
            }
        )
    return scenarios


def _draw_previews(
    *,
    thumbnail_path: Path,
    scenarios: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    if not thumbnail_path.is_file():
        return []
    ensure_dir(output_dir)
    previews: list[dict[str, Any]] = []
    font = ImageFont.load_default()
    bounds_x = (-5.5, 6.6)
    bounds_y = (-4.6, 5.5)
    map_rect = (48, 80, 768, 680)

    def to_pixel(point: Sequence[Any]) -> tuple[int, int]:
        x = float(point[0])
        y = float(point[1])
        px = int(map_rect[0] + ((x - bounds_x[0]) / (bounds_x[1] - bounds_x[0])) * (map_rect[2] - map_rect[0]))
        py = int(map_rect[3] - ((y - bounds_y[0]) / (bounds_y[1] - bounds_y[0])) * (map_rect[3] - map_rect[1]))
        return max(map_rect[0], min(map_rect[2], px)), max(map_rect[1], min(map_rect[3], py))

    def draw_fixture(
        draw: ImageDraw.ImageDraw,
        label: str,
        lower_xy: tuple[float, float],
        upper_xy: tuple[float, float],
        color: tuple[int, int, int, int],
    ) -> None:
        x1, y1 = to_pixel((lower_xy[0], lower_xy[1]))
        x2, y2 = to_pixel((upper_xy[0], upper_xy[1]))
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        draw.rounded_rectangle(box, radius=6, fill=color, outline=(61, 65, 78, 255), width=2)
        draw.text((box[0] + 6, box[1] + 5), label, fill=(28, 31, 40, 255), font=font)

    for scenario in scenarios:
        image = Image.new("RGBA", (1280, 720), (246, 244, 238, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1280, 64), fill=(20, 24, 30, 255))
        draw.text((28, 18), _string(scenario.get("scenario_id")), fill=(255, 255, 255, 255), font=font)
        draw.text(
            (760, 18),
            "Top-down scenario storyboard only - not Isaac-rendered simulation evidence",
            fill=(255, 222, 118, 255),
            font=font,
        )
        draw.rounded_rectangle(map_rect, radius=8, fill=(236, 239, 232, 255), outline=(46, 51, 63, 255), width=2)
        for index in range(1, 6):
            gx = map_rect[0] + index * (map_rect[2] - map_rect[0]) // 6
            gy = map_rect[1] + index * (map_rect[3] - map_rect[1]) // 6
            draw.line((gx, map_rect[1], gx, map_rect[3]), fill=(207, 211, 203, 255), width=1)
            draw.line((map_rect[0], gy, map_rect[2], gy), fill=(207, 211, 203, 255), width=1)
        draw_fixture(draw, "fridge", (-2.35, -0.20), (-1.25, 1.45), (211, 223, 233, 255))
        draw_fixture(draw, "stove", (0.10, 1.45), (1.05, 2.45), (226, 215, 204, 255))
        draw_fixture(draw, "sink", (1.75, 0.75), (2.85, 1.75), (206, 226, 229, 255))
        draw_fixture(draw, "top cabinets", (-1.60, 2.00), (2.95, 2.55), (221, 226, 205, 255))
        draw_fixture(draw, "island/counter", (-1.35, -0.25), (2.35, 0.75), (229, 221, 196, 255))
        points = [
            scenario.get("spawn_position_xyz"),
            *(scenario.get("waypoints_xyz") or []),
            scenario.get("target_position_xyz"),
        ]
        pixel_points: list[tuple[int, int]] = []
        for point in points:
            if not isinstance(point, Sequence) or len(point) < 2:
                continue
            pixel_points.append(to_pixel(point))
        if len(pixel_points) >= 2:
            draw.line(pixel_points, fill=(35, 143, 215, 255), width=6, joint="curve")
        for point_index, pixel in enumerate(pixel_points):
            color = (92, 255, 125, 255) if point_index == 0 else (255, 90, 90, 255)
            if 0 < point_index < len(pixel_points) - 1:
                color = (255, 219, 80, 255)
            x, y = pixel
            draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color, outline=(0, 0, 0, 255), width=2)
            label = "S" if point_index == 0 else "T" if point_index == len(pixel_points) - 1 else str(point_index)
            draw.text((x - 4, y - 5), label, fill=(0, 0, 0, 255), font=font)
        thumbnail = Image.open(thumbnail_path).convert("RGBA").resize((420, 420))
        image.alpha_composite(thumbnail, (824, 112))
        draw.rectangle((824, 540, 1244, 628), fill=(255, 255, 255, 230))
        draw.text((840, 556), "Source USD thumbnail", fill=(20, 24, 30, 255), font=font)
        draw.text(
            (840, 584),
            "Scenario line uses authored USD coordinates, not camera pixels.",
            fill=(84, 88, 98, 255),
            font=font,
        )
        out_path = output_dir / f"{_string(scenario.get('scenario_id'))}_preview.png"
        image.save(out_path)
        previews.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "path": str(out_path),
                "simulator_rendered": False,
                "purpose": "route_storyboard_preview_not_navigation_proof",
                "size_bytes": out_path.stat().st_size,
                "sha256": _sha256(out_path),
            }
        )
    return previews


def _write_isaac_runner_script_legacy_unused(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        '''"""Isaac Sim runtime entrypoint for Blueprint Lightwheel Kitchen G1 scenarios.

Run this with Isaac Sim's Python, not stock CPython, for example:
  ./python.sh run_lightwheel_kitchen_isaac_scenarios.py --request isaac_execution_request.json
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def _resolve_path(value: str | None, *, request_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    bundle_root = request_path.parent.parent if request_path.parent.name == "request" else request_path.parent
    return (bundle_root / path).resolve()


def _resolve_scene_usd(request: dict, *, request_path: Path) -> Path:
    scene_path = _resolve_path(request.get("scene_usd_path"), request_path=request_path)
    if scene_path and scene_path.is_file():
        return scene_path
    layout = request.get("bundle_layout") if isinstance(request.get("bundle_layout"), dict) else {}
    source_zip = _resolve_path(layout.get("source_zip_path"), request_path=request_path)
    extract_dir = _resolve_path(layout.get("assets_extract_dir"), request_path=request_path)
    scene_relative = layout.get("scene_usd_relative") or "Collected_KitchenRoom/KitchenRoom.usd"
    if not source_zip or not source_zip.is_file():
        raise FileNotFoundError(f"missing provider source zip: {source_zip}")
    if not extract_dir:
        raise FileNotFoundError("missing provider assets_extract_dir")
    scene_path = extract_dir / scene_relative
    if not scene_path.is_file():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(extract_dir)
    if not scene_path.is_file():
        raise FileNotFoundError(f"provider source zip did not materialize {scene_path}")
    return scene_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    blockers = []
    try:
        scene_usd_path = _resolve_scene_usd(request, request_path=request_path)
    except Exception as exc:
        _write_json(output_path, {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "blocked",
            "blockers": ["provider_lightwheel_scene_usd_unavailable"],
            "error": repr(exc),
            "request_path": str(request_path),
        })
        return 2
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:
        _write_json(output_path, {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "blocked",
            "blockers": ["isaacsim_python_module_unavailable"],
            "error": repr(exc),
            "request_path": str(request_path),
        })
        return 2

    robot_usd = os.environ.get("BLUEPRINT_ISAAC_UNITREE_G1_USD") or request.get("unitree_g1_usd")
    if not robot_usd:
        blockers.append("missing_unitree_g1_usd_or_imported_robot_asset")

    simulation_app = SimulationApp({"headless": True})
    try:
        from omni.usd import get_context  # type: ignore
        context = get_context()
        context.open_stage(str(scene_usd_path))
        stage = context.get_stage()
        if stage is None:
            blockers.append("isaac_stage_open_failed")
        # Robot spawn/control is intentionally gated until a verified Unitree G1 USD or
        # Isaac MJCF importer path is provided in this runtime.
        status = "blocked" if blockers else "ready_for_robot_binding"
        _write_json(output_path, {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": status,
            "blockers": blockers,
            "scene_usd_path": str(scene_usd_path),
            "scenario_count": len(request.get("scenarios", [])),
            "scene_loaded_in_isaac": stage is not None,
            "unitree_g1_spawned": False,
            "scenario_videos_generated": False,
            "proof_boundary": {
                "isaac_scene_load_attempted": True,
                "unitree_g1_navigation_proven": False,
                "generated_world_rank_fidelity_result_proven": False
            },
        })
        return 0 if not blockers else 2
    finally:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else {}
            if existing.get("status") not in {"completed", "blocked", "failed"}:
                runner_phase_before_terminalizer = existing.get("provider_runtime_phase")
                inherited_blockers = list(existing.get("blockers") or [])
                if "isaac_runner_finished_without_terminal_result" not in inherited_blockers:
                    inherited_blockers.append("isaac_runner_finished_without_terminal_result")
                scenarios = _scenario_list(request)
                _write_json(
                    output_path,
                    {
                        **existing,
                        "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                        "status": "blocked",
                        "provider_runtime_phase": "runner_finished_without_terminal_result",
                        "runner_terminal_phase": "runner_finished_without_terminal_result",
                        "runner_phase_before_terminalizer": runner_phase_before_terminalizer,
                        "blockers": inherited_blockers,
                        "scenario_count": len(scenarios),
                        "scenarios_attempted": 0,
                        "scenarios_verified": 0,
                        "per_scenario_results": _blocked_before_scenario_attempts(
                            scenarios,
                            inherited_blockers,
                        ),
                        "raw_secret_values_recorded": False,
                        "raw_signed_urls_recorded": False,
                    },
                )
        except Exception as exc:
            _write_json(
                output_path,
                {
                    "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                    "status": "blocked",
                    "provider_runtime_phase": "runner_terminalization_failed",
                    "blockers": ["isaac_runner_terminalization_failed"],
                    "error": repr(exc),
                    "scenario_count": len(request.get("scenarios", [])),
                    "scenarios_attempted": 0,
                    "scenarios_verified": 0,
                    "raw_secret_values_recorded": False,
                    "raw_signed_urls_recorded": False,
                },
            )
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )


def _write_isaac_runner_script(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        '''"""Isaac Sim runtime entrypoint for Blueprint Lightwheel Kitchen G1 scenarios.

Run this with Isaac Sim's Python, not stock CPython, for example:
./python.sh run_lightwheel_kitchen_isaac_scenarios.py --request isaac_execution_request.json
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
import urllib.request
from urllib.parse import urlparse

OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE = "Isaac/Robots/Unitree/G1/g1.usd"
OFFICIAL_ISAAC_UNITREE_G1_DOC_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_robots.html"
)
ROBOT_PRIM_PATH = "/World/UnitreeG1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    signed_put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL")
    if signed_put_url:
        try:
            data = path.read_bytes()
            request = urllib.request.Request(
                signed_put_url,
                data=data,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:
            pass


def _resolve_path(value: str | None, *, request_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    bundle_root = request_path.parent.parent if request_path.parent.name == "request" else request_path.parent
    return (bundle_root / path).resolve()


def _resolve_scene_usd(request: dict, *, request_path: Path) -> Path:
    scene_path = _resolve_path(request.get("scene_usd_path"), request_path=request_path)
    if scene_path and scene_path.is_file():
        return scene_path
    layout = request.get("bundle_layout") if isinstance(request.get("bundle_layout"), dict) else {}
    source_zip = _resolve_path(layout.get("source_zip_path"), request_path=request_path)
    extract_dir = _resolve_path(layout.get("assets_extract_dir"), request_path=request_path)
    scene_relative = layout.get("scene_usd_relative") or "Collected_KitchenRoom/KitchenRoom.usd"
    if not source_zip or not source_zip.is_file():
        raise FileNotFoundError(f"missing provider source zip: {source_zip}")
    if not extract_dir:
        raise FileNotFoundError("missing provider assets_extract_dir")
    scene_path = extract_dir / scene_relative
    if not scene_path.is_file():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(extract_dir)
    if not scene_path.is_file():
        raise FileNotFoundError(f"provider source zip did not materialize {scene_path}")
    return scene_path


def _is_remote_uri(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https", "omniverse", "ov"}


def _candidate_is_file_or_remote(uri: str) -> bool:
    if _is_remote_uri(uri):
        return True
    return Path(uri).expanduser().is_file()


def _official_g1_usd_candidates() -> list[dict]:
    candidates: list[dict] = []
    env_usd = os.environ.get("BLUEPRINT_ISAAC_UNITREE_G1_USD", "").strip()
    if env_usd:
        candidates.append(
            {
                "source": "BLUEPRINT_ISAAC_UNITREE_G1_USD",
                "uri": env_usd,
                "exists_or_remote": _candidate_is_file_or_remote(env_usd),
            }
        )
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore

        assets_root = get_assets_root_path()
    except Exception as exc:
        candidates.append(
            {
                "source": "isaacsim.storage.native.get_assets_root_path",
                "uri": None,
                "exists_or_remote": False,
                "error": repr(exc),
            }
        )
    else:
        if assets_root:
            uri = assets_root.rstrip("/") + "/" + OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE
            candidates.append(
                {
                    "source": "isaac_assets_root",
                    "uri": uri,
                    "exists_or_remote": _candidate_is_file_or_remote(uri),
                }
            )
        else:
            candidates.append(
                {
                    "source": "isaac_assets_root",
                    "uri": None,
                    "exists_or_remote": False,
                    "error": "get_assets_root_path_returned_none",
                }
            )
    return candidates


def _official_g1_resolution() -> dict:
    candidates = _official_g1_usd_candidates()
    selected_uri = None
    for candidate in candidates:
        uri = candidate.get("uri")
        if uri and candidate.get("exists_or_remote"):
            selected_uri = str(uri)
            break
    blockers = []
    if not selected_uri:
        blockers.append("official_isaac_unitree_g1_usd_not_resolved")
    return {
        "preferred_binding": "official_isaac_unitree_g1_usd",
        "doc_url": OFFICIAL_ISAAC_UNITREE_G1_DOC_URL,
        "assets_root_relative_path": OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE,
        "content_browser_path": "Robots/Unitree/G1/g1.usd",
        "candidates": candidates,
        "selected_uri": selected_uri,
        "resolved": bool(selected_uri),
        "blockers": blockers,
    }


def _stage_robot_api_evidence(stage, prim_path: str) -> dict:
    evidence = {
        "robot_prim_path": prim_path,
        "robot_prim_valid": False,
        "descendant_prim_count": 0,
        "articulation_root_api_prim_count": 0,
        "collision_api_prim_count": 0,
        "rigid_body_api_prim_count": 0,
    }
    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except Exception as exc:
        evidence["usd_physics_import_error"] = repr(exc)
        return evidence
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if prim is None or not prim.IsValid():
        return evidence
    evidence["robot_prim_valid"] = True
    try:
        prim_range = iter(Usd.PrimRange(prim))
        next(prim_range, None)
        for desc in prim_range:
            evidence["descendant_prim_count"] += 1
            if desc.HasAPI(UsdPhysics.ArticulationRootAPI):
                evidence["articulation_root_api_prim_count"] += 1
            if desc.HasAPI(UsdPhysics.CollisionAPI):
                evidence["collision_api_prim_count"] += 1
            if desc.HasAPI(UsdPhysics.RigidBodyAPI):
                evidence["rigid_body_api_prim_count"] += 1
    except Exception as exc:
        evidence["usd_prim_range_traversal_error"] = repr(exc)
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        evidence["articulation_root_api_prim_count"] += 1
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        evidence["collision_api_prim_count"] += 1
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        evidence["rigid_body_api_prim_count"] += 1
    return evidence


def _add_official_g1_reference(stage, selected_uri: str | None) -> dict:
    result = {
        "robot_prim_path": ROBOT_PRIM_PATH,
        "selected_uri": None,
        "reference_added": False,
        "blockers": [],
    }
    if stage is None:
        result["blockers"].append("isaac_stage_open_failed")
        return result
    if not selected_uri:
        result["blockers"].append("official_isaac_unitree_g1_usd_reference_not_added")
        return result
    try:
        prim = stage.DefinePrim(ROBOT_PRIM_PATH, "Xform")
        prim.GetReferences().AddReference(str(selected_uri))
        result["selected_uri"] = str(selected_uri)
        result["reference_added"] = True
    except Exception as exc:
        result["reference_error"] = repr(exc)
        result["blockers"].append("official_isaac_unitree_g1_usd_reference_not_added")
    return result


def _unitree_g1_binding_result(resolution: dict, reference: dict, evidence: dict) -> dict:
    result = {
        "preferred_binding": "official_isaac_unitree_g1_usd",
        "doc_url": OFFICIAL_ISAAC_UNITREE_G1_DOC_URL,
        "assets_root_relative_path": OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE,
        "content_browser_path": "Robots/Unitree/G1/g1.usd",
        "candidates": resolution.get("candidates", []),
        "selected_uri": reference.get("selected_uri") or resolution.get("selected_uri"),
        "official_usd_resolved": bool(resolution.get("resolved")),
        "reference_added": bool(reference.get("reference_added")),
        "spawned_or_resolved": False,
        "collision_enabled_verified": False,
        "controllable_articulation_detected": False,
        "control_command_applied": False,
        "blockers": [],
        "resolution": resolution,
        "reference": reference,
        "stage_api_evidence": evidence,
    }
    for source in (resolution, reference):
        for blocker in source.get("blockers", []):
            if blocker not in result["blockers"]:
                result["blockers"].append(blocker)
    result["spawned_or_resolved"] = bool(
        evidence.get("robot_prim_valid") and evidence.get("descendant_prim_count", 0) > 0
    )
    result["collision_enabled_verified"] = bool(evidence.get("collision_api_prim_count", 0) > 0)
    result["controllable_articulation_detected"] = bool(
        evidence.get("articulation_root_api_prim_count", 0) > 0
    )
    if not result["official_usd_resolved"]:
        result["blockers"].append("official_isaac_unitree_g1_usd_not_resolved")
    if not result["reference_added"]:
        result["blockers"].append("official_isaac_unitree_g1_usd_reference_not_added")
    if not result["spawned_or_resolved"]:
        result["blockers"].append("official_isaac_unitree_g1_usd_reference_unresolved")
    if not result["collision_enabled_verified"]:
        result["blockers"].append("official_isaac_unitree_g1_collision_api_unverified")
    if not result["controllable_articulation_detected"]:
        result["blockers"].append("official_isaac_unitree_g1_articulation_api_unverified")
    result["blockers"].append("unitree_g1_control_command_not_applied")
    result["blockers"] = list(dict.fromkeys(result["blockers"]))
    return result


def _distance(a: list | tuple, b: list | tuple) -> float | None:
    try:
        return sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)) ** 0.5
    except Exception:
        return None


def _scenario_attempt_result(scenario: dict, unitree_g1_binding: dict, stage_update_ticks: int) -> dict:
    scenario_id = str(scenario.get("scenario_id") or "")
    blockers: list[str] = []
    if not unitree_g1_binding.get("spawned_or_resolved"):
        blockers.append("unitree_g1_not_spawned_or_resolved_for_scenario")
    if not unitree_g1_binding.get("collision_enabled_verified"):
        blockers.append("unitree_g1_collision_api_unverified_for_scenario")
    if not unitree_g1_binding.get("controllable_articulation_detected"):
        blockers.append("unitree_g1_articulation_api_unverified_for_scenario")
    if not unitree_g1_binding.get("control_command_applied"):
        blockers.append("unitree_g1_control_command_not_applied_for_scenario")
    if stage_update_ticks <= 0:
        blockers.append("isaac_stage_update_ticks_zero_no_physics_steps")
    target_distance = _distance(
        scenario.get("spawn_position_xyz") or [],
        scenario.get("target_position_xyz") or [],
    )
    return {
        "scenario_id": scenario_id,
        "status": "verified" if not blockers else "blocked_after_runtime_preflight",
        "execution_attempted": True,
        "execution_verified": not blockers,
        "spawn_position_xyz": scenario.get("spawn_position_xyz"),
        "target_position_xyz": scenario.get("target_position_xyz"),
        "waypoint_count": len(scenario.get("waypoints_xyz") or []),
        "initial_target_distance_m": round(target_distance, 4) if target_distance is not None else None,
        "stage_update_ticks": stage_update_ticks,
        "controller_bound": bool(unitree_g1_binding.get("control_command_applied")),
        "target_reached": False,
        "target_reach_gate_passed": False,
        "collision_gate_passed": bool(unitree_g1_binding.get("collision_enabled_verified")) and not blockers,
        "fall_gate_passed": False,
        "overview_video_path": None,
        "robot_pov_video_path": None,
        "trace_jsonl_path": None,
        "contact_collision_log_path": None,
        "blockers": blockers,
    }


def _run_scenario_attempts(scenarios: list, unitree_g1_binding: dict, stage_update_ticks: int) -> list[dict]:
    return [
        _scenario_attempt_result(dict(scenario), unitree_g1_binding, stage_update_ticks)
        for scenario in scenarios
        if isinstance(scenario, dict)
    ]


def _scenario_list(request: dict) -> list[dict]:
    return [dict(item) for item in request.get("scenarios", []) if isinstance(item, dict)]


def _blocked_before_scenario_attempts(scenarios: list[dict], blockers: list[str]) -> list[dict]:
    return [
        {
            "scenario_id": scenario.get("scenario_id"),
            "status": "blocked_before_scenario_attempt",
            "execution_attempted": False,
            "execution_verified": False,
            "blockers": blockers,
        }
        for scenario in scenarios
    ]


def _write_terminal_blocked_before_attempt(
    output_path: Path,
    request: dict,
    *,
    phase: str,
    blockers: list[str],
    exc: Exception | None = None,
    **extra: object,
) -> None:
    scenarios = _scenario_list(request)
    unique_blockers = list(dict.fromkeys(blockers))
    payload = {
        "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
        "status": "blocked",
        "provider_runtime_phase": phase,
        "runner_terminal_phase": phase,
        "blockers": unique_blockers,
        "scenario_count": len(scenarios),
        "scenarios_attempted": 0,
        "scenarios_verified": 0,
        "per_scenario_results": _blocked_before_scenario_attempts(scenarios, unique_blockers),
        "raw_secret_values_recorded": False,
        "raw_signed_urls_recorded": False,
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    if exc is not None:
        payload["error_type"] = type(exc).__name__
        payload["error"] = repr(exc)
    _write_json(output_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    _write_json(
        output_path,
        {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "running",
            "provider_runtime_phase": "runner_started",
            "blockers": ["isaac_runner_in_progress"],
            "request_path": str(request_path),
            "raw_secret_values_recorded": False,
            "raw_signed_urls_recorded": False,
        },
    )
    try:
        scene_usd_path = _resolve_scene_usd(request, request_path=request_path)
    except Exception as exc:
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "blocked",
                "blockers": ["provider_lightwheel_scene_usd_unavailable"],
                "error": repr(exc),
                "request_path": str(request_path),
            },
        )
        return 2
    _write_json(
        output_path,
        {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "running",
            "provider_runtime_phase": "runner_scene_resolved",
            "blockers": ["isaac_runner_in_progress"],
            "request_path": str(request_path),
            "scene_usd_path": str(scene_usd_path),
            "raw_secret_values_recorded": False,
            "raw_signed_urls_recorded": False,
        },
    )
    _write_json(
        output_path,
        {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "running",
            "provider_runtime_phase": "runner_importing_isaacsim",
            "blockers": ["isaac_runner_in_progress"],
            "scene_usd_path": str(scene_usd_path),
            "raw_secret_values_recorded": False,
            "raw_signed_urls_recorded": False,
        },
    )
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "blocked",
                "blockers": ["isaacsim_python_module_unavailable"],
                "error": repr(exc),
                "request_path": str(request_path),
            },
        )
        return 2
    _write_json(
        output_path,
        {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "running",
            "provider_runtime_phase": "runner_isaacsim_imported",
            "blockers": ["isaac_runner_in_progress"],
            "scene_usd_path": str(scene_usd_path),
            "raw_secret_values_recorded": False,
            "raw_signed_urls_recorded": False,
        },
    )

    _write_json(
        output_path,
        {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "running",
            "provider_runtime_phase": "runner_simulation_app_starting",
            "blockers": ["isaac_runner_in_progress"],
            "scene_usd_path": str(scene_usd_path),
            "raw_secret_values_recorded": False,
            "raw_signed_urls_recorded": False,
        },
    )
    simulation_app = SimulationApp({"headless": True})
    try:
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_simulation_app_started",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        import omni.usd  # type: ignore

        context = omni.usd.get_context()
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_opening_lightwheel_stage",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        context.open_stage(str(scene_usd_path))
        stage = context.get_stage()
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_lightwheel_stage_opened",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        if stage is None:
            blockers.append("isaac_stage_open_failed")
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_referencing_official_g1",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        unitree_g1_resolution = _official_g1_resolution()
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_official_g1_resolved",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "unitree_g1_resolution": unitree_g1_resolution,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        unitree_g1_reference = _add_official_g1_reference(
            stage,
            unitree_g1_resolution.get("selected_uri"),
        )
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_official_g1_reference_added",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "unitree_g1_resolution": unitree_g1_resolution,
                "unitree_g1_reference": unitree_g1_reference,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_robot_api_evidence_collection_starting",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "unitree_g1_resolution": unitree_g1_resolution,
                "unitree_g1_reference": unitree_g1_reference,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        try:
            robot_api_evidence = _stage_robot_api_evidence(stage, ROBOT_PRIM_PATH)
        except Exception as exc:
            _write_terminal_blocked_before_attempt(
                output_path,
                request,
                phase="runner_robot_api_evidence_failed",
                blockers=[
                    "isaac_robot_api_evidence_collection_failed",
                    "all_five_lightwheel_scenarios_not_verified",
                ],
                exc=exc,
                scene_usd_path=str(scene_usd_path),
                stage_opened=stage is not None,
                unitree_g1_resolution=unitree_g1_resolution,
                unitree_g1_reference=unitree_g1_reference,
            )
            return 2
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_unitree_g1_binding_starting",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "unitree_g1_resolution": unitree_g1_resolution,
                "unitree_g1_reference": unitree_g1_reference,
                "robot_api_evidence": robot_api_evidence,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        try:
            unitree_g1_binding = _unitree_g1_binding_result(
                unitree_g1_resolution,
                unitree_g1_reference,
                robot_api_evidence,
            )
        except Exception as exc:
            _write_terminal_blocked_before_attempt(
                output_path,
                request,
                phase="runner_unitree_g1_binding_failed",
                blockers=[
                    "unitree_g1_binding_result_failed",
                    "all_five_lightwheel_scenarios_not_verified",
                ],
                exc=exc,
                scene_usd_path=str(scene_usd_path),
                stage_opened=stage is not None,
                unitree_g1_resolution=unitree_g1_resolution,
                unitree_g1_reference=unitree_g1_reference,
                robot_api_evidence=robot_api_evidence,
            )
            return 2
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_robot_api_evidence_collected",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "unitree_g1_binding": unitree_g1_binding,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        stage_update_ticks = int(os.environ.get("BLUEPRINT_ISAAC_STAGE_UPDATE_TICKS", "0"))
        for update_index in range(stage_update_ticks):
            _write_json(
                output_path,
                {
                    "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                    "status": "running",
                    "provider_runtime_phase": "runner_lightwheel_stage_update_starting",
                    "blockers": ["isaac_runner_in_progress"],
                    "scene_usd_path": str(scene_usd_path),
                    "stage_opened": stage is not None,
                    "stage_update_index": update_index,
                    "stage_update_ticks": stage_update_ticks,
                    "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
            try:
                simulation_app.update()
            except Exception as exc:
                _write_terminal_blocked_before_attempt(
                    output_path,
                    request,
                    phase="runner_lightwheel_stage_update_failed",
                    blockers=[
                        "isaac_lightwheel_stage_update_failed",
                        "all_five_lightwheel_scenarios_not_verified",
                    ],
                    exc=exc,
                    scene_usd_path=str(scene_usd_path),
                    stage_opened=stage is not None,
                    stage_update_index=update_index,
                    stage_update_ticks=stage_update_ticks,
                    unitree_g1_binding=unitree_g1_binding,
                )
                return 2
            _write_json(
                output_path,
                {
                    "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                    "status": "running",
                    "provider_runtime_phase": "runner_lightwheel_stage_update_completed",
                    "blockers": ["isaac_runner_in_progress"],
                    "scene_usd_path": str(scene_usd_path),
                    "stage_opened": stage is not None,
                    "stage_update_index": update_index,
                    "stage_update_ticks": stage_update_ticks,
                    "raw_secret_values_recorded": False,
                    "raw_signed_urls_recorded": False,
                },
            )
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "status": "running",
                "provider_runtime_phase": "runner_scenario_attempts_starting",
                "blockers": ["isaac_runner_in_progress"],
                "scene_usd_path": str(scene_usd_path),
                "stage_opened": stage is not None,
                "stage_update_ticks": stage_update_ticks,
                "unitree_g1_binding": unitree_g1_binding,
                "raw_secret_values_recorded": False,
                "raw_signed_urls_recorded": False,
            },
        )
        try:
            scenario_results = _run_scenario_attempts(
                list(request.get("scenarios", [])),
                unitree_g1_binding,
                stage_update_ticks,
            )
        except Exception as exc:
            _write_terminal_blocked_before_attempt(
                output_path,
                request,
                phase="runner_scenario_attempt_generation_failed",
                blockers=[
                    "isaac_scenario_attempt_generation_failed",
                    "all_five_lightwheel_scenarios_not_verified",
                ],
                exc=exc,
                scene_usd_path=str(scene_usd_path),
                stage_opened=stage is not None,
                stage_update_ticks=stage_update_ticks,
                unitree_g1_binding=unitree_g1_binding,
            )
            return 2
        scenarios_verified = sum(1 for item in scenario_results if item.get("execution_verified"))
        scenario_blockers = []
        for item in scenario_results:
            for blocker in item.get("blockers", []):
                if blocker not in scenario_blockers:
                    scenario_blockers.append(blocker)
        if not unitree_g1_binding.get("spawned_or_resolved"):
            blockers.append("official_isaac_unitree_g1_usd_reference_unresolved")
        if not unitree_g1_binding.get("collision_enabled_verified"):
            blockers.append("official_isaac_unitree_g1_collision_api_unverified")
        if not unitree_g1_binding.get("controllable_articulation_detected"):
            blockers.append("official_isaac_unitree_g1_articulation_api_unverified")
        blockers.extend(unitree_g1_binding.get("blockers", []))
        blockers.extend(scenario_blockers)
        if scenarios_verified != len(request.get("scenarios", [])):
            blockers.append("all_five_lightwheel_scenarios_not_verified")
        blockers = list(dict.fromkeys(blockers))
        _write_json(
            output_path,
            {
                "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                "provider_runtime_phase": "runner_scenario_attempts_completed",
                "runner_terminal_phase": "runner_scenario_attempts_completed",
                "status": "blocked" if blockers else "completed",
                "blockers": blockers,
                "scene_usd_path": str(scene_usd_path),
                "scenario_count": len(request.get("scenarios", [])),
                "scenarios_attempted": len(scenario_results),
                "scenarios_verified": scenarios_verified,
                "per_scenario_results": scenario_results,
                "scene_loaded_in_isaac": stage is not None,
                "unitree_g1_binding": unitree_g1_binding,
                "unitree_g1_spawned": bool(unitree_g1_binding.get("spawned_or_resolved")),
                "collision_enabled_verified": bool(
                    unitree_g1_binding.get("collision_enabled_verified")
                ),
                "controllable_articulation_detected": bool(
                    unitree_g1_binding.get("controllable_articulation_detected")
                ),
                "control_command_applied": bool(unitree_g1_binding.get("control_command_applied")),
                "scenario_videos_generated": False,
                "proof_boundary": {
                    "isaac_scene_load_attempted": True,
                    "official_isaac_unitree_g1_usd_attempted": True,
                    "unitree_g1_navigation_proven": False,
                    "generated_world_rank_fidelity_result_proven": False,
                },
            },
        )
        return 0 if not blockers else 2
    finally:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else {}
            if existing.get("status") not in {"completed", "blocked", "failed"}:
                runner_phase_before_terminalizer = existing.get("provider_runtime_phase")
                inherited_blockers = list(existing.get("blockers") or [])
                if "isaac_runner_finished_without_terminal_result" not in inherited_blockers:
                    inherited_blockers.append("isaac_runner_finished_without_terminal_result")
                scenarios = _scenario_list(request)
                _write_json(
                    output_path,
                    {
                        **existing,
                        "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                        "status": "blocked",
                        "provider_runtime_phase": "runner_finished_without_terminal_result",
                        "runner_terminal_phase": "runner_finished_without_terminal_result",
                        "runner_phase_before_terminalizer": runner_phase_before_terminalizer,
                        "blockers": inherited_blockers,
                        "scenario_count": len(scenarios),
                        "scenarios_attempted": 0,
                        "scenarios_verified": 0,
                        "per_scenario_results": _blocked_before_scenario_attempts(
                            scenarios,
                            inherited_blockers,
                        ),
                        "raw_secret_values_recorded": False,
                        "raw_signed_urls_recorded": False,
                    },
                )
        except Exception as exc:
            _write_json(
                output_path,
                {
                    "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
                    "status": "blocked",
                    "provider_runtime_phase": "runner_terminalization_failed",
                    "blockers": ["isaac_runner_terminalization_failed"],
                    "error": repr(exc),
                    "scenario_count": len(request.get("scenarios", [])),
                    "scenarios_attempted": 0,
                    "scenarios_verified": 0,
                    "raw_secret_values_recorded": False,
                    "raw_signed_urls_recorded": False,
                },
            )
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )


def _provider_probe_summary(provider_proof_path: Path | None) -> dict[str, Any]:
    proof = _read_json_mapping(provider_proof_path)
    if not proof:
        return {
            "provider_probe_path": str(provider_proof_path) if provider_proof_path else None,
            "status": "not_supplied",
            "runpod_api_probe_performed": False,
            "active_pod_count_before": None,
            "active_pod_count_after": None,
            "runpod_side_effects_may_have_occurred": False,
        }
    return {
        "provider_probe_path": str(provider_proof_path),
        "status": proof.get("status"),
        "runpod_api_probe_performed": bool(proof.get("api_call_performed")),
        "runpod_side_effects_may_have_occurred": bool(
            proof.get("runpod_side_effects_may_have_occurred")
        ),
        "api_key_source": proof.get("api_key_source"),
        "active_pod_count_before": proof.get("active_pod_count_before"),
        "active_pod_count_after": proof.get("active_pod_count_after"),
        "pod_stop_performed": bool(proof.get("pod_stop_performed")),
        "blockers": proof.get("blockers") if isinstance(proof.get("blockers"), list) else [],
        "secret_values_in_artifact": bool(proof.get("secret_values_in_artifact")),
        "raw_api_key_stored": bool(proof.get("raw_api_key_stored")),
    }


def _provider_launch_contract(
    runtime: Mapping[str, Any],
    provider_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provider_credentials = _mapping(runtime.get("provider_credentials"))
    packet = _mapping(provider_packet)
    packet_bundle = _mapping(packet.get("asset_bundle"))
    packet_fetchability = _mapping(packet_bundle.get("provider_fetchability"))
    packet_launch = _mapping(packet.get("launch_contract"))
    missing_inputs: list[str] = []
    if packet_fetchability.get("provider_fetchable_by_runpod") is not True:
        missing_inputs.append("provider_fetchable_lightwheel_capture_root_or_asset_bundle_uri")
    if (
        packet_launch.get("versioned_isaac_worker_image_ref_present") is not True
        and packet_launch.get("selected_isaac_runtime_image_ref_present") is not True
    ):
        missing_inputs.append("versioned_provider_fetchable_BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF")
    if (
        packet_launch.get("direct_isaac_base_image_launch_supported_by_packet") is not True
        and packet_launch.get("versioned_isaac_worker_image_ref_present") is not True
    ):
        missing_inputs.append("provider_fetchable_robot_eval_worker_manifest_uri")
    if not _string(packet_launch.get("required_command_inside_isaac_python")):
        missing_inputs.append("isaac_runtime_preflight_command_for_provider_worker")
    if packet_launch.get("artifact_output_or_runtime_manifest_put_present") is not True:
        missing_inputs.append(
            "provider_writable_artifact_output_uri_or_signed_runtime_manifest_put_url"
        )
    return {
        "status": "ready_for_provider_runtime_probe" if not missing_inputs else "blocked",
        "provider": "runpod",
        "live_api_credentials_available": bool(
            provider_credentials.get("runpod_api_key_env_present")
            or provider_credentials.get("runpod_api_key_file_env_present")
            or provider_credentials.get("runpod_config_file_present")
        ),
        "required_missing_inputs": _dedupe(missing_inputs),
        "ngc_credentials": {
            "ngc_api_key_file_env_present": bool(provider_credentials.get("ngc_api_key_file_env_present")),
            "ngc_api_key_env_present": bool(provider_credentials.get("ngc_api_key_env_present")),
            "nvidia_api_key_file_env_present": bool(
                provider_credentials.get("nvidia_api_key_file_env_present")
            ),
            "nvidia_api_key_env_present": bool(provider_credentials.get("nvidia_api_key_env_present")),
            "raw_secret_values_recorded": False,
        },
        "gpu_policy": {
            "preferred_rtx_gpu_classes": [
                "L40S",
                "RTX 6000 Ada",
                "RTX A6000",
                "RTX 4090",
            ],
            "avoid_as_first_isaac_rendering_target": ["A100", "H100"],
        },
    }


def _provider_runtime_probe_shell_script() -> str:
    return """set -u
cd /workspace
if [ -x /isaac-sim/python.sh ]; then
  BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON=/isaac-sim/python.sh
elif command -v python3 >/dev/null 2>&1; then
  BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON="$(command -v python3)"
else
  RESULT=/workspace/lightwheel_kitchen_isaac_provider_runtime_result.json
  printf '{"schema_version":"lightwheel_kitchen_isaac_runtime_result.v1","status":"blocked","provider_runtime_phase":"provider_runtime_python_unavailable","blockers":["provider_runtime_python_unavailable"],"raw_secret_values_recorded":false}\\n' > "$RESULT"
  if command -v curl >/dev/null 2>&1 && [ -n "${BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL:-}" ]; then
    curl -fsS -X PUT -H 'Content-Type: application/json' --data-binary @"$RESULT" "$BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL" >/dev/null || true
  fi
  exit 0
fi
cat > /workspace/blueprint_lightwheel_probe_result.py <<'PYUTIL'
import argparse
import datetime as _dt
import json
import os
import platform
import socket
import urllib.request
from pathlib import Path


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write(path: Path, phase: str, status: str, blockers: list[str], merge_existing: bool) -> dict:
    existing = _load(path) if merge_existing else {}
    payload = dict(existing)
    payload.setdefault("schema_version", "lightwheel_kitchen_isaac_runtime_result.v1")
    if status != "keep":
        payload["status"] = status
    payload["generated_at"] = _now()
    payload["provider_runtime_phase"] = phase
    payload["provider_runtime_host"] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    history = payload.get("provider_runtime_phase_history")
    if not isinstance(history, list):
        history = []
    history.append({"phase": phase, "status": payload.get("status"), "generated_at": payload["generated_at"]})
    payload["provider_runtime_phase_history"] = history[-40:]
    if blockers:
        merged = list(payload.get("blockers") or [])
        for blocker in blockers:
            if blocker not in merged:
                merged.append(blocker)
        payload["blockers"] = merged
    payload["raw_secret_values_recorded"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return payload


def _upload(path: Path) -> bool:
    put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")
    if not put_url:
        return False
    data = path.read_bytes()
    request = urllib.request.Request(
        put_url,
        data=data,
        method="PUT",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(request, timeout=60).read()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--status", default="running")
    parser.add_argument("--blocker", action="append", default=[])
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    _write(
        output,
        phase=args.phase,
        status=args.status,
        blockers=list(args.blocker),
        merge_existing=args.merge_existing,
    )
    if args.upload:
        _upload(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYUTIL
RESULT=/workspace/lightwheel_kitchen_isaac_provider_runtime_result.json
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_runtime_probe_started --status running --blocker provider_runtime_probe_in_progress --upload || true
if ! "$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" - <<'PYDL'
import os
import urllib.request

url = os.environ.get("BLUEPRINT_EVAL_MANIFEST_URI") or os.environ.get("BLUEPRINT_LIGHTWHEEL_PROVIDER_BUNDLE_URL")
if not url:
    raise RuntimeError("missing signed provider bundle URL")
urllib.request.urlretrieve(url, "lightwheel_kitchen_isaac_provider_bundle.zip")
PYDL
then
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_bundle_download_failed --status blocked --blocker provider_bundle_download_failed --upload || true
exit 0
fi
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_bundle_downloaded --status running --blocker provider_runtime_probe_in_progress --merge-existing --upload || true
if ! "$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" -m zipfile -e lightwheel_kitchen_isaac_provider_bundle.zip lightwheel_kitchen_isaac_provider_bundle; then
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_bundle_extract_failed --status blocked --blocker provider_bundle_extract_failed --merge-existing --upload || true
exit 0
fi
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_bundle_extracted --status running --blocker provider_runtime_probe_in_progress --merge-existing --upload || true
if ! cd lightwheel_kitchen_isaac_provider_bundle; then
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_bundle_workdir_missing --status blocked --blocker provider_bundle_workdir_missing --merge-existing --upload || true
exit 0
fi
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output "$RESULT" --phase provider_isaac_command_starting --status running --blocker provider_isaac_command_in_progress --merge-existing --upload || true
BLUEPRINT_LIGHTWHEEL_ISAAC_TIMEOUT_SECONDS="${BLUEPRINT_LIGHTWHEEL_ISAAC_TIMEOUT_SECONDS:-360}"
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" - <<'PYSUPERVISE'
import os
import signal
import subprocess
import sys
import time

timeout_seconds = int(os.environ.get("BLUEPRINT_LIGHTWHEEL_ISAAC_TIMEOUT_SECONDS", "360"))
python = os.environ["BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON"]
env = os.environ.copy()
env.setdefault("ACCEPT_EULA", "Y")
env.setdefault("PRIVACY_CONSENT", "Y")
command = [
    python,
    "runner/run_lightwheel_kitchen_isaac_scenarios.py",
    "--request",
    "request/isaac_execution_request.provider.json",
    "--output",
    "lightwheel_kitchen_isaac_provider_runtime_result.json",
]
kwargs = {"env": env}
if hasattr(os, "setsid"):
    kwargs["preexec_fn"] = os.setsid
process = subprocess.Popen(command, **kwargs)
try:
    exit_code = process.wait(timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=15)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass
    raise SystemExit(124)
raise SystemExit(exit_code if 0 <= exit_code <= 255 else 1)
PYSUPERVISE
rc=$?
if [ "$rc" -eq 124 ]; then
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output lightwheel_kitchen_isaac_provider_runtime_result.json --phase provider_isaac_command_timeout --status blocked --blocker provider_isaac_command_timeout --merge-existing || true
fi
if [ ! -f lightwheel_kitchen_isaac_provider_runtime_result.json ]; then
 "$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output lightwheel_kitchen_isaac_provider_runtime_result.json --phase provider_runtime_result_missing_after_isaac_command --status blocked --blocker provider_runtime_result_missing_after_isaac_command || true
fi
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" - "$rc" <<'PYFINAL'
import datetime as _dt
import json
import sys
from pathlib import Path

rc = int(sys.argv[1])
path = Path("lightwheel_kitchen_isaac_provider_runtime_result.json")
payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
status = payload.get("status")
if status not in {"completed", "blocked", "failed"}:
    runner_phase_before_provider_finalizer = payload.get("provider_runtime_phase")
    blockers = list(payload.get("blockers") or [])
    if rc == 124:
        blocker = "provider_isaac_command_timeout"
    elif rc != 0:
        blocker = f"provider_isaac_command_exit_code_{rc}"
    else:
        blocker = "provider_isaac_command_finished_without_terminal_result"
    if blocker not in blockers:
        blockers.append(blocker)
    payload.update(
        {
            "schema_version": "lightwheel_kitchen_isaac_runtime_result.v1",
            "status": "blocked",
            "provider_runtime_phase": "provider_runtime_finished_without_terminal_result",
            "runner_phase_before_provider_finalizer": runner_phase_before_provider_finalizer,
            "provider_isaac_command_exit_code": rc,
            "blockers": blockers,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "raw_secret_values_recorded": False,
            "raw_signed_urls_recorded": False,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PYFINAL
"$BLUEPRINT_LIGHTWHEEL_ISAAC_PYTHON" /workspace/blueprint_lightwheel_probe_result.py --output lightwheel_kitchen_isaac_provider_runtime_result.json --phase provider_runtime_final_upload --status keep --merge-existing --upload || true
exit 0
"""


def _runpod_direct_launch_request_shape(
    *,
    command: str,
    manifest_uri: str,
    runtime_result_put_url_present: bool,
    selected_runtime_image_ref: str,
    container_registry_auth_id: str = "",
) -> dict[str, Any]:
    image_shape: dict[str, Any] = {
        "configured_image_ref": selected_runtime_image_ref,
        "configured_image_ref_present": bool(selected_runtime_image_ref),
        "configured_image_ref_is_versioned": ":" in selected_runtime_image_ref,
        "configured_image_ref_fetchable_by_provider": bool(selected_runtime_image_ref),
        "image_family": "isaac-sim-base-direct",
        "image_ref_source": "selected_isaac_runtime_image_ref",
        "container_registry_auth_id_present": bool(container_registry_auth_id),
        "container_registry_auth_id_source": (
            RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV if container_registry_auth_id else None
        ),
    }
    if container_registry_auth_id:
        image_shape["container_registry_auth_id"] = container_registry_auth_id
    return {
        "schema_version": "robot_eval_gpu_provider_launch_request.v1",
        "job_id": "lightwheel-kitchen-isaac-walkthrough2",
        "provider": "runpod",
        "status": "request_manifest_ready",
        "operation": "lightwheel_kitchen_isaac_direct_runtime_probe",
        "provider_request_shape": {
            "api_payload_is_provider_adapter_template": True,
            "api_payload_values_are_redacted": False,
            "operation": "lightwheel_kitchen_isaac_direct_runtime_probe",
            "image": image_shape,
"docker_entrypoint": ["bash"],
"docker_start_cmd": ["-lc", command],
"entrypoint_source": "nvidia_isaac_sim_container_docs_require_bash_entrypoint",
"command": command,
            "environment": {
                "secret_env_var_names": ["RUNPOD_API_KEY"],
                "secret_values_in_artifact": False,
                "signed_url_values_in_public_artifact": False,
                "plaintext_env_var_names": ["ACCEPT_EULA", "PRIVACY_CONSENT"],
                "plaintext_env_values": {
                    "ACCEPT_EULA": "Y",
                    "PRIVACY_CONSENT": "Y",
                },
            },
            "inputs": {
                "manifest_uri": manifest_uri,
                "manifest_uri_kind": "signed_lightwheel_provider_bundle_zip",
                "manifest_uri_fetchable_by_provider": bool(manifest_uri),
                "capture_root_bundle_uri": manifest_uri,
                "artifact_output_uri_required": False,
                "artifact_output_uri": None,
                "runtime_manifest_signed_put_url_passed_by_private_env": runtime_result_put_url_present,
                "simulator": "isaac_sim",
            },
            "runtime_preflight": {
                "simulator": "isaac_sim",
                "robot_profile_id": "unitree_g1",
                "preferred_robot_asset": "official_isaac_unitree_g1_usd",
            },
            "gpu": {
                "gpu_count": 1,
                "provider_gpu_priority": [
                    "NVIDIA L40S",
                    "NVIDIA RTX 6000 Ada Generation",
                    "NVIDIA RTX A6000",
                    "NVIDIA GeForce RTX 4090",
                ],
                "preferred_gpu_type_id": "NVIDIA L40S",
                "disallowed_gpu_classes": ["A100", "H100"],
                "volume_in_gb": 80,
                "container_disk_in_gb": 120,
                "min_vcpu_count": 8,
                "min_memory_in_gb": 32,
            },
            "limits": {
                "max_active_workers": 1,
                "hard_timeout_seconds": 900,
                "idle_timeout_seconds": 60,
                "external_watchdog_ttl_seconds": 1200,
                "scale_to_zero_default": True,
            },
        },
    }


def _write_runpod_direct_launch_request(
    *,
    output_dir: Path,
    provider_packet: Mapping[str, Any],
    provider_private_signed_url_file: Path | None,
) -> dict[str, Any]:
    private_inputs = _read_json_mapping(provider_private_signed_url_file)
    bundle_url = _signed_url_from_private_inputs(private_inputs, "bundle_get")
    runtime_result_put_url = _signed_url_from_private_inputs(private_inputs, "runtime_result_put")
    launch_contract = _mapping(provider_packet.get("launch_contract"))
    selected_runtime_image_ref = (
        _string(launch_contract.get("selected_isaac_runtime_image_ref"))
        or DEFAULT_ISAAC_RUNTIME_IMAGE_REF
    )
    container_registry_auth_id = _string(os.environ.get(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV))
    command = _provider_runtime_probe_shell_script()
    private_request = _runpod_direct_launch_request_shape(
        command=command,
        manifest_uri=bundle_url,
        runtime_result_put_url_present=bool(runtime_result_put_url),
        selected_runtime_image_ref=selected_runtime_image_ref,
        container_registry_auth_id=container_registry_auth_id,
    )
    private_request["private_inputs"] = {
        "provider_private_signed_url_file": str(provider_private_signed_url_file)
        if provider_private_signed_url_file
        else None,
        "bundle_get_signed_url_present": bool(bundle_url),
        "runtime_result_put_signed_url_present": bool(runtime_result_put_url),
        "raw_signed_urls_recorded_in_public_manifest": False,
    }
    public_request = _redacted_jsonable(private_request)
    public_request["provider_request_shape"]["api_payload_values_are_redacted"] = True
    if container_registry_auth_id:
        public_request["provider_request_shape"]["image"][
            "container_registry_auth_id"
        ] = "<redacted:runpod-container-registry-auth-id>"
    public_request["private_request_path"] = str(output_dir / RUNPOD_DIRECT_LAUNCH_REQUEST_LOCAL_NAME)
    public_request["raw_signed_urls_recorded_in_manifest"] = False
    public_request["status"] = (
        "request_manifest_ready"
        if bundle_url and runtime_result_put_url
        else "blocked_private_signed_urls_missing"
    )
    public_request["blockers"] = _dedupe(
        [
            *([] if bundle_url else ["missing_bundle_get_signed_url"]),
            *([] if runtime_result_put_url else ["missing_runtime_result_put_signed_url"]),
        ]
    )
    public_path = output_dir / RUNPOD_DIRECT_LAUNCH_REQUEST_NAME
    local_path = output_dir / RUNPOD_DIRECT_LAUNCH_REQUEST_LOCAL_NAME
    _safe_write_json(public_path, public_request)
    if bundle_url and runtime_result_put_url:
        _safe_write_json(local_path, private_request)
        local_path.chmod(0o600)
    return {
        "status": public_request["status"],
        "public_request_path": str(public_path),
        "private_request_path": str(local_path),
        "private_request_written": bool(bundle_url and runtime_result_put_url),
        "command_has_phase_heartbeats": True,
        "runtime_result_put_private_env_required": True,
        "blockers": public_request["blockers"],
        "raw_signed_urls_recorded_in_public_manifest": False,
    }


def _write_provider_packet(
    *,
    output_dir: Path,
    source_zip: Path,
    bundle: Mapping[str, Any],
    unitree_g1_mjcf: Mapping[str, Any],
    provider_execution_request_path: Path,
    runtime: Mapping[str, Any],
    provider_proof_path: Path | None,
    provider_artifact_root_uri: str | None,
    upload_provider_packet: bool,
    provider_private_signed_url_file: Path | None,
    repo_commit: str,
    selected_isaac_runtime_image_ref: str | None = None,
) -> dict[str, Any]:
    packet_path = output_dir / PROVIDER_PACKET_NAME
    provider_bundle_uri = (
        _uri_join(provider_artifact_root_uri, PROVIDER_BUNDLE_NAME)
        if provider_artifact_root_uri
        else ""
    )
    provider_packet_uri = (
        _uri_join(provider_artifact_root_uri, PROVIDER_PACKET_NAME)
        if provider_artifact_root_uri
        else ""
    )

    upload_results: list[dict[str, Any]] = []
    if upload_provider_packet and provider_bundle_uri:
        upload_results.append(
            {
                "artifact": "provider_bundle",
                **_upload_provider_file(Path(str(bundle.get("path"))), provider_bundle_uri),
            }
        )
    elif upload_provider_packet:
        upload_results.append(
            {
                "artifact": "provider_bundle",
                "status": "blocked",
                "blockers": ["missing_provider_artifact_root_uri"],
                "raw_secret_values_recorded": False,
            }
        )

    private_inputs = _read_json_mapping(provider_private_signed_url_file)
    signed_url_evidence = {
        "private_input_path": str(provider_private_signed_url_file)
        if provider_private_signed_url_file
        else None,
        "private_input_file_present": bool(
            provider_private_signed_url_file and provider_private_signed_url_file.is_file()
        ),
        "bundle_get": _signed_url_entry_summary(
            private_inputs=private_inputs,
            key="bundle_get",
            private_input_path=provider_private_signed_url_file,
        ),
        "packet_get": _signed_url_entry_summary(
            private_inputs=private_inputs,
            key="packet_get",
            private_input_path=provider_private_signed_url_file,
        ),
        "runtime_result_put": _signed_url_entry_summary(
            private_inputs=private_inputs,
            key="runtime_result_put",
            private_input_path=provider_private_signed_url_file,
        ),
    }
    private_env = _write_provider_private_env(output_dir=output_dir, private_inputs=private_inputs)

    runtime_credentials = _mapping(runtime.get("provider_credentials"))
    versioned_image_ref = _string(os.environ.get("BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF"))
    generic_image_ref = _string(os.environ.get("BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"))
    selected_runtime_image_ref = (
        _string(selected_isaac_runtime_image_ref)
        or _string(os.environ.get(ISAAC_RUNTIME_IMAGE_REF_ENV))
        or DEFAULT_ISAAC_RUNTIME_IMAGE_REF
    )
    runpod_registry_auth_id = _string(os.environ.get(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV))
    artifact_output_uri = _string(os.environ.get("BLUEPRINT_ARTIFACT_OUTPUT_URI"))
    runtime_manifest_put_present = bool(os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    unitree_g1_usd = _string(os.environ.get("BLUEPRINT_ISAAC_UNITREE_G1_USD"))
    private_bundle_get_present = bool(signed_url_evidence["bundle_get"]["signed_url_signature_present"])
    private_runtime_put_present = bool(
        signed_url_evidence["runtime_result_put"]["signed_url_signature_present"]
    )
    provider_fetchable = (
        _provider_uri_is_fetchable_without_extra_credentials(provider_bundle_uri)
        or private_bundle_get_present
    )
    storage_credentials_required = _provider_uri_requires_storage_credentials(provider_bundle_uri)
    upload_blockers = [
        str(blocker)
        for result in upload_results
        for blocker in result.get("blockers", [])
        if isinstance(result.get("blockers"), list)
    ]
    required_missing_inputs = []
    if not provider_fetchable:
        required_missing_inputs.append(
            "provider_fetchable_lightwheel_asset_bundle_signed_get_url_or_runtime_gcs_credentials"
        )
    if not (versioned_image_ref or generic_image_ref or selected_runtime_image_ref):
        required_missing_inputs.append("versioned_provider_fetchable_BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF")
    if not (artifact_output_uri or runtime_manifest_put_present or private_runtime_put_present):
        required_missing_inputs.append(
            "provider_writable_artifact_output_uri_or_signed_runtime_manifest_put_url"
        )
    unitree_mjcf_materialized = unitree_g1_mjcf.get("status") == "materialized"
    execution_blockers = [
        "official_isaac_unitree_g1_usd_runtime_probe_not_run",
        "provider_isaac_runtime_not_launched",
        "provider_runtime_result_missing",
        "all_five_lightwheel_scenarios_not_executed",
    ]
    packet_blockers = _dedupe([*upload_blockers, *required_missing_inputs, *execution_blockers])
    bundle_upload = next(
        (result for result in upload_results if result.get("artifact") == "provider_bundle"),
        {"status": "not_requested", "raw_secret_values_recorded": False},
    )
    generated_at = utc_now_iso()
    if required_missing_inputs:
        status = (
            "provider_packet_uploaded_external_inputs_blocked"
            if bundle_upload.get("status") == "uploaded"
            else "provider_packet_prepared_external_inputs_blocked"
        )
    else:
        status = (
            "provider_packet_uploaded_ready_for_runtime_probe"
            if bundle_upload.get("status") == "uploaded"
            else "provider_packet_prepared_ready_for_runtime_probe"
        )
    if upload_blockers:
        status = "provider_packet_upload_blocked"
    packet: dict[str, Any] = {
        "schema_version": "lightwheel_kitchen_isaac_provider_packet.v1",
        "generated_at": generated_at,
        "status": status,
        "provider": "runpod",
        "provider_packet_path": str(packet_path),
        "source_repository": {
            "url": LIGHTWHEEL_REPO_URL,
            "commit": repo_commit,
            "license": LIGHTWHEEL_LICENSE,
            "license_boundary": "Non-commercial license; do not use for commercial delivery without rights review.",
        },
        "asset_bundle": {
            **dict(bundle),
            "source_zip": {
                "path": str(source_zip),
                "sha256": _sha256(source_zip),
                "size_bytes": source_zip.stat().st_size,
            },
            "provider_bundle_uri": _redact_signed_url_text(provider_bundle_uri) or None,
            "provider_packet_uri": _redact_signed_url_text(provider_packet_uri) or None,
            "upload_requested": upload_provider_packet,
            "upload_status": bundle_upload.get("status"),
            "upload_result": _redacted_jsonable(bundle_upload),
            "provider_fetchability": {
                "provider_fetchable_by_runpod": provider_fetchable,
                "uri_requires_runtime_storage_credentials": storage_credentials_required,
                "reason": (
                    "private signed GET URL detected"
                    if private_bundle_get_present
                    else "signed HTTP(S) URL detected"
                    if provider_fetchable
                    else "gs/s3/r2 or local URI still requires signed GET URL or provider runtime credentials"
                ),
            },
        },
        "private_provider_inputs": {
            "signed_url_evidence": signed_url_evidence,
            "private_env": private_env,
            "raw_signed_urls_recorded_in_manifest": False,
        },
        "provider_execution_request": {
            "path": str(provider_execution_request_path),
            "bundle_member": f"request/{PROVIDER_REQUEST_NAME}",
            "schema_version": "lightwheel_kitchen_isaac_provider_execution_request.v1",
        },
        "unitree_g1_binding": {
            "preferred_verified_isaac_usd_env": "BLUEPRINT_ISAAC_UNITREE_G1_USD",
            "preferred_verified_isaac_usd_env_present": bool(unitree_g1_usd),
            "official_isaac_usd_candidate": {
                "source": "NVIDIA Isaac Sim 5.0 Robot Assets documentation",
                "doc_url": OFFICIAL_ISAAC_UNITREE_G1_DOC_URL,
                "assets_root_relative_path": OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE,
                "content_browser_path": "Robots/Unitree/G1/g1.usd",
                "runtime_resolution_status": "pending_provider_isaac_runtime_probe",
            },
            "mjcf_fallback": dict(unitree_g1_mjcf),
            "mjcf_asset_bundled": unitree_mjcf_materialized
            and bool(bundle.get("contains_unitree_g1_mjcf")),
            "isaac_import_or_spawn_verified": False,
            "collision_enabled_verified": False,
            "controllable_verified": False,
        },
        "launch_contract": {
            "versioned_isaac_worker_image_ref_present": bool(versioned_image_ref or generic_image_ref),
            "selected_isaac_runtime_image_ref": selected_runtime_image_ref,
            "selected_isaac_runtime_image_ref_source": (
                "argument"
                if _string(selected_isaac_runtime_image_ref)
                else ISAAC_RUNTIME_IMAGE_REF_ENV
                if os.environ.get(ISAAC_RUNTIME_IMAGE_REF_ENV)
                else "default"
            ),
            "selected_isaac_runtime_image_ref_present": bool(selected_runtime_image_ref),
            "selected_isaac_runtime_image_ref_is_versioned": bool(
                selected_runtime_image_ref
                and not selected_runtime_image_ref.endswith((":latest", ":local", ":dev", ":test"))
            ),
            "selected_isaac_runtime_image_ref_fetchable_check": "docker_manifest_inspect_passed_or_required",
            "direct_isaac_base_image_launch_supported_by_packet": bool(selected_runtime_image_ref),
            "worker_image_env_vars": [
                "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
                "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF",
            ],
            "artifact_output_or_runtime_manifest_put_present": bool(
                artifact_output_uri or runtime_manifest_put_present or private_runtime_put_present
            ),
            "artifact_output_uri_env_present": bool(artifact_output_uri),
            "runtime_manifest_signed_put_env_present": bool(
                runtime_manifest_put_present or private_runtime_put_present
            ),
            "runtime_manifest_signed_put_private_file_present": private_runtime_put_present,
            "unitree_g1_binding_env_present": bool(unitree_g1_usd),
            "unitree_g1_mjcf_asset_bundled": unitree_mjcf_materialized
            and bool(bundle.get("contains_unitree_g1_mjcf")),
            "unitree_g1_binding_env_var": "BLUEPRINT_ISAAC_UNITREE_G1_USD",
            "ngc_or_nvidia_credentials_present": bool(
                runtime_credentials.get("ngc_api_key_file_env_present")
                or runtime_credentials.get("ngc_api_key_env_present")
                or runtime_credentials.get("nvidia_api_key_file_env_present")
                or runtime_credentials.get("nvidia_api_key_env_present")
            ),
            "runpod_container_registry_auth_id_present": bool(runpod_registry_auth_id),
            "runpod_container_registry_auth_id_source": (
                RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV if runpod_registry_auth_id else None
            ),
            "raw_registry_credentials_recorded": False,
            "required_command_inside_isaac_python": (
                "./python.sh runner/run_lightwheel_kitchen_isaac_scenarios.py "
                f"--request request/{PROVIDER_REQUEST_NAME} --output {PROVIDER_RESULT_NAME}"
            ),
            "expected_runtime_result": PROVIDER_RESULT_NAME,
        },
        "runpod_policy": {
            "bounded_spend_required": True,
            "max_live_pods": 1,
            "prove_active_pod_count_zero_after_run": True,
            "preferred_rtx_gpu_classes": [
                "L40S",
                "RTX 6000 Ada",
                "RTX A6000",
                "RTX 4090",
            ],
            "avoid_as_first_isaac_rendering_target": ["A100", "H100"],
        },
        "runtime_preflight_blockers": list(runtime.get("blockers", [])),
        "provider_probe": _provider_probe_summary(provider_proof_path),
        "required_missing_inputs": _dedupe(required_missing_inputs),
        "execution_blockers": execution_blockers,
        "blockers": packet_blockers,
        "proof_boundary": {
            "provider_packet_created": True,
            "provider_bundle_uploaded": bundle_upload.get("status") == "uploaded",
            "provider_bundle_fetchable_without_extra_credentials": provider_fetchable,
            "runpod_pod_launched_for_lightwheel_isaac": False,
            "isaac_scene_loaded_in_provider_runtime": False,
            "unitree_g1_imported_or_spawned": False,
            "unitree_g1_collision_enabled": False,
            "unitree_g1_controllable": False,
            "five_scenarios_executed": False,
            "overview_videos_generated": False,
            "robot_pov_videos_generated": False,
            "target_reaching_gates_passed": False,
            "collision_fall_gates_passed": False,
            "generated_world_rank_fidelity_result_proven": False,
            "customer_delivery_readiness_proven": False,
        },
        "raw_secret_values_recorded": False,
    }
    _safe_write_json(packet_path, packet)
    if upload_provider_packet and provider_packet_uri:
        packet_upload = {
            "artifact": "provider_packet",
            **_upload_provider_file(packet_path, provider_packet_uri),
        }
        packet["provider_packet_upload"] = _redacted_jsonable(packet_upload)
        if packet_upload.get("status") != "uploaded":
            packet["status"] = "provider_packet_upload_blocked"
            packet["blockers"] = _dedupe(
                [
                    *[str(item) for item in packet.get("blockers", [])],
                    *[
                        str(blocker)
                        for blocker in packet_upload.get("blockers", [])
                        if isinstance(packet_upload.get("blockers"), list)
                    ],
                ]
            )
        _safe_write_json(packet_path, packet)
        if packet_upload.get("status") == "uploaded":
            _upload_provider_file(packet_path, provider_packet_uri)
    return packet


def _write_blocked_execution_outputs(
    *,
    output_dir: Path,
    scenarios: Sequence[Mapping[str, Any]],
    scene_usd: Path,
    runtime_preflight_path: Path,
    handoff_path: Path,
    execution_request_path: Path,
    blockers: Sequence[str],
    runtime: Mapping[str, Any],
    usdchecker: Mapping[str, Any],
    dependency_audit: Mapping[str, Any],
    stage_summary: Mapping[str, Any],
    provider_proof_path: Path | None,
    provider_packet_path: Path | None,
) -> dict[str, str]:
    provider_probe = _provider_probe_summary(provider_proof_path)
    provider_packet = _read_json_mapping(provider_packet_path)
    provider_contract = _provider_launch_contract(runtime, provider_packet)
    generated_at = utc_now_iso()
    global_blockers = _dedupe(
        [
            *[str(item) for item in blockers],
            *[
                str(item)
                for item in provider_packet.get("blockers", [])
                if isinstance(provider_packet.get("blockers"), list)
            ],
        *(
            ["lightwheel_isaac_provider_launch_contract_blocked"]
            if provider_contract.get("status") == "blocked"
            else []
        ),
            "lightwheel_isaac_scenarios_not_executed",
        ]
    )
    provider_packet_summary = {
        "path": str(provider_packet_path) if provider_packet_path else None,
        "status": provider_packet.get("status"),
        "provider": provider_packet.get("provider"),
        "asset_bundle": _mapping(provider_packet.get("asset_bundle")),
        "required_missing_inputs": provider_packet.get("required_missing_inputs")
        if isinstance(provider_packet.get("required_missing_inputs"), list)
        else [],
        "execution_blockers": provider_packet.get("execution_blockers")
        if isinstance(provider_packet.get("execution_blockers"), list)
        else [],
        "raw_secret_values_recorded": bool(provider_packet.get("raw_secret_values_recorded")),
    }

    per_scenario_path = output_dir / PER_SCENARIO_RESULTS_NAME
    per_scenario_results = {
        "schema_version": "lightwheel_kitchen_isaac_per_scenario_results.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "scenario_count": len(scenarios),
        "scenarios_completed": 0,
        "scenarios_blocked": len(scenarios),
        "results": [
            {
                "scenario_id": scenario.get("scenario_id"),
                "status": "blocked_not_executed",
                "execution_proven": False,
                "unitree_g1_spawned": False,
                "controller_bound": False,
                "navigation_policy_type": "not_executed",
                "target_reached": False,
                "target_reach_gate_passed": False,
                "collision_gate_passed": False,
                "fall_gate_passed": False,
                "overview_video_path": None,
                "robot_pov_video_path": None,
                "trace_jsonl_path": None,
                "contact_collision_log_path": None,
                "blockers": global_blockers,
                "proof_boundary": (
                    "Scenario is specified but no Isaac Sim/Lab runtime executed it; "
                    "no target-reaching, collision, fall, video, or robot POV claim is proven."
                ),
            }
            for scenario in scenarios
        ],
    }
    _safe_write_json(per_scenario_path, per_scenario_results)

    contact_path = output_dir / CONTACT_COLLISION_MANIFEST_NAME
    contact_manifest = {
        "schema_version": "lightwheel_kitchen_contact_collision_manifest.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "contact_logging_executed": False,
        "collision_gate_evaluated": False,
        "persistent_collision_violations": None,
        "scenario_count": len(scenarios),
        "scenario_logs": [],
        "blockers": global_blockers,
        "proof_boundary": "No Isaac physics/contact step ran, so contact and collision safety are unproven.",
    }
    _safe_write_json(contact_path, contact_manifest)

    video_checks_path = output_dir / VIDEO_ARTIFACT_CHECKS_NAME
    video_checks = {
        "schema_version": "lightwheel_kitchen_video_artifact_checks.v1",
        "generated_at": generated_at,
        "status": "blocked_no_final_videos",
        "checks_performed": False,
        "required_when_videos_exist": {
            "exists": True,
            "nonzero_bytes": True,
            "frame_count_gt_1": True,
            "minimum_resolution": "1280x720",
            "nonblank_frames": True,
        },
        "overview_videos": [],
        "robot_pov_videos": [],
        "blockers": global_blockers,
    }
    _safe_write_json(video_checks_path, video_checks)

    execution_path = output_dir / ISAAC_EXECUTION_MANIFEST_NAME
    execution_manifest = {
        "schema_version": "lightwheel_kitchen_isaac_execution_manifest.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "scene_usd_path": str(scene_usd),
        "runtime_preflight_manifest": str(runtime_preflight_path),
        "handoff_manifest": str(handoff_path),
        "isaac_execution_request": str(execution_request_path),
        "per_scenario_results": str(per_scenario_path),
        "contact_collision_manifest": str(contact_path),
        "video_artifact_checks": str(video_checks_path),
        "provider_probe": provider_probe,
        "provider_packet": provider_packet_summary,
        "provider_launch_contract": provider_contract,
        "scenario_count": len(scenarios),
        "scenarios_executed": 0,
        "overview_videos_generated": 0,
        "robot_pov_videos_generated": 0,
        "trace_jsonl_generated": 0,
        "contact_collision_logs_generated": 0,
        "blockers": global_blockers,
        "evidence": {
            "usd_stage_status": stage_summary.get("status"),
            "usdchecker_status": usdchecker.get("status"),
            "usdchecker_blockers": usdchecker.get("blockers"),
            "usd_dependency_audit_status": dependency_audit.get("status"),
            "usd_dependency_audit_blockers": dependency_audit.get("blockers"),
            "usd_missing_dependency_names": dependency_audit.get("missing_dependency_names"),
            "runtime_preflight_status": runtime.get("status"),
            "runtime_preflight_blockers": runtime.get("blockers"),
            "local_isaac_runtime_ready": bool(runtime.get("isaac_local_runtime_ready")),
        },
        "proof_boundary": {
            "lightwheel_usd_scene_materialized": True,
            "isaac_scene_loaded_in_runtime": False,
            "isaac_sim_execution_proven": False,
            "isaac_lab_execution_proven": False,
            "unitree_g1_asset_imported_or_spawned": False,
            "unitree_g1_collision_enabled": False,
            "unitree_g1_controllable": False,
            "unitree_g1_navigation_proven": False,
            "target_reaching_gates_passed": False,
            "collision_fall_gates_passed": False,
            "generated_world_rank_fidelity_result_proven": False,
            "real_robot_pov": False,
            "customer_delivery_readiness_proven": False,
        },
    }
    _safe_write_json(execution_path, execution_manifest)

    readiness_path = output_dir / FINAL_READINESS_MANIFEST_NAME
    readiness = {
        "schema_version": "lightwheel_kitchen_isaac_final_readiness.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "execution_manifest": str(execution_path),
        "per_scenario_results": str(per_scenario_path),
        "runtime_preflight_manifest": str(runtime_preflight_path),
        "provider_probe": provider_probe,
        "provider_packet": provider_packet_summary,
        "rights_boundary": {
            "lightwheel_source_license": LIGHTWHEEL_LICENSE,
            "license_boundary": "Non-commercial license; do not use for commercial delivery without rights review.",
        },
        "final_blockers": global_blockers,
        "readiness": {
            "ready_for_local_isaac_execution": False,
            "ready_for_provider_isaac_execution": False,
            "ready_for_customer_delivery": False,
        },
        "next_unblocked_step": (
            "Provide a provider-fetchable Isaac Sim/Lab worker image and manifest/output URIs, "
            "or run the generated request on a Linux RTX host with Isaac Sim/Lab and a verified "
            "Unitree G1 USD/importer binding."
        ),
    }
    _safe_write_json(readiness_path, readiness)

    return {
        "execution_manifest": str(execution_path),
        "per_scenario_results": str(per_scenario_path),
        "contact_collision_manifest": str(contact_path),
        "video_artifact_checks": str(video_checks_path),
        "final_readiness_manifest": str(readiness_path),
    }


def build_lightwheel_kitchen_isaac_scenarios(
    *,
    capture_root: str | Path | None = None,
    source_zip: str | Path | None = None,
    source_repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    asset_output_dir: str | Path | None = None,
    provider_proof: str | Path | None = None,
    provider_artifact_root_uri: str | None = None,
    upload_provider_packet: bool = False,
    provider_private_signed_url_file: str | Path | None = None,
    unitree_g1_mjcf_root: str | Path | None = None,
    isaac_runtime_image_ref: str | None = None,
    repo_commit: str = LIGHTWHEEL_DEFAULT_COMMIT,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve() if capture_root else None
    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (root / DEFAULT_OUTPUT_RELATIVE if root else Path("output/lightwheel_kitchen_isaac_scenarios").resolve())
    )
    ensure_dir(out_dir)
    repo_root = Path(source_repo_root).expanduser().resolve() if source_repo_root else None
    zip_path = Path(source_zip).expanduser().resolve() if source_zip else None
    if zip_path is None and repo_root:
        candidate = repo_root / "Lightwheel_Kitchen.zip"
        zip_path = candidate if candidate.is_file() else None
    if zip_path is None or not zip_path.is_file():
        raise FileNotFoundError("--source-zip or --source-repo-root with Lightwheel_Kitchen.zip is required")

    assets_dir = (
        Path(asset_output_dir).expanduser().resolve()
        if asset_output_dir
        else out_dir / "assets"
    )
    materialized_root = _materialize_assets(source_zip=zip_path, asset_output_dir=assets_dir)
    scene_usd = materialized_root / MAIN_USD_RELATIVE
    thumbnail = materialized_root / THUMBNAIL_RELATIVE
    inventory = _asset_inventory_from_zip(zip_path)
    stage_summary = _usd_stage_summary(scene_usd)
    usdchecker = _usdchecker_summary(scene_usd)
    dependency_audit = _usd_dependency_presence_audit(
        asset_root=materialized_root,
        source_zip=zip_path,
        usdchecker=usdchecker,
    )
    unitree_mjcf_root = _resolve_unitree_g1_mjcf_root(explicit_root=unitree_g1_mjcf_root)
    unitree_mjcf_summary = _unitree_g1_mjcf_summary(unitree_mjcf_root)
    runtime = _runtime_preflight()
    scenarios = _default_scenarios()
    preview_records = _draw_previews(
        thumbnail_path=thumbnail,
        scenarios=scenarios,
        output_dir=out_dir / "scenario_preview_frames",
    )

    scenario_manifest_path = out_dir / "lightwheel_kitchen_scenarios.json"
    scenario_manifest = {
        "schema_version": "lightwheel_kitchen_g1_scenario_specs.v1",
        "generated_at": utc_now_iso(),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "scenario_preview_frames": preview_records,
        "scenario_execution_status": "not_executed",
        "unitree_g1_navigation_proven": False,
        "proof_boundary": {
            "scenario_specs_authored": True,
            "isaac_sim_execution_proven": False,
            "unitree_g1_spawned_in_lightwheel_kitchen": False,
            "navigation_policy_success_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }
    _safe_write_json(scenario_manifest_path, scenario_manifest)

    runner_script = out_dir / "run_lightwheel_kitchen_isaac_scenarios.py"
    _write_isaac_runner_script(runner_script)
    execution_request_path = out_dir / "isaac_execution_request.json"
    execution_request = {
        "schema_version": "lightwheel_kitchen_isaac_execution_request.v1",
        "scene_usd_path": str(scene_usd),
        "scenarios": scenarios,
        "requested_outputs": {
            "overview_videos": True,
            "robot_pov_videos": True,
            "per_scenario_trace_jsonl": True,
            "contacts_and_collision_events": True,
        },
        "robot_binding": {
            "robot_profile_id": "unitree_g1",
            "preferred_binding": "BLUEPRINT_ISAAC_UNITREE_G1_USD",
            "official_isaac_asset_candidate": {
                "source": "NVIDIA Isaac Sim 5.0 Robot Assets documentation",
                "doc_url": OFFICIAL_ISAAC_UNITREE_G1_DOC_URL,
                "assets_root_relative_path": OFFICIAL_ISAAC_UNITREE_G1_USD_RELATIVE,
                "content_browser_path": "Robots/Unitree/G1/g1.usd",
            },
            "fallback_binding": "Isaac MJCF importer from MuJoCo Menagerie G1 only if official USD fails",
            "current_status": "blocked_until_official_isaac_g1_usd_runtime_resolution_is_verified",
        },
        "runner_script": str(runner_script),
    }
    _safe_write_json(execution_request_path, execution_request)

    blockers: list[str] = []
    if not inventory.get("main_usd_present"):
        blockers.append("lightwheel_kitchen_main_usd_missing")
    if usdchecker.get("status") != "passed":
        blockers.append("lightwheel_usdchecker_failed")
        blockers.extend(str(item) for item in usdchecker.get("blockers", []))
    if dependency_audit.get("status") == "missing_required_dependencies":
        blockers.extend(str(item) for item in dependency_audit.get("blockers", []))
    blockers.extend(str(item) for item in runtime.get("blockers", []))
    blockers.append("official_isaac_unitree_g1_usd_resolution_inside_runtime_unverified")
    blockers = list(dict.fromkeys(blockers))
    local_ready = not blockers

    preflight_path = out_dir / "lightwheel_kitchen_isaac_runtime_preflight.json"
    preflight = {
        "schema_version": "lightwheel_kitchen_isaac_runtime_preflight.v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if local_ready else "blocked",
        "asset_inventory": inventory,
        "scene_usd_path": str(scene_usd),
        "scene_usd_sha256": _sha256(scene_usd),
        "thumbnail_path": str(thumbnail) if thumbnail.is_file() else None,
        "usd_stage_summary": stage_summary,
        "usdchecker": usdchecker,
        "usd_dependency_presence_audit": dependency_audit,
        "unitree_g1_mjcf_fallback": unitree_mjcf_summary,
        "runtime_preflight": runtime,
        "blockers": blockers,
    }
    _safe_write_json(preflight_path, preflight)

    handoff_path = out_dir / "lightwheel_kitchen_isaac_handoff_manifest.json"
    handoff = {
        "schema_version": LIGHTWHEEL_KITCHEN_ISAAC_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "ready_for_isaac_execution" if local_ready else "blocked",
        "capture_root": str(root) if root else None,
        "source_repository": {
            "url": LIGHTWHEEL_REPO_URL,
            "commit": repo_commit,
            "license": LIGHTWHEEL_LICENSE,
            "license_boundary": "Non-commercial license; do not use for commercial delivery without rights review.",
        },
        "asset_root": str(materialized_root),
        "scene_usd_path": str(scene_usd),
        "mujoco_path_decision": {
            "use_mujoco": False,
            "reason": "Lightwheel Kitchen archive exposes USD/textures only and no native MJCF/URDF/OBJ/GLB scene path.",
            "mujoco_native_asset_present": inventory.get("mujoco_native_asset_present"),
        },
        "isaac_path_decision": {
            "use_isaac_sim_or_lab": True,
            "reason": "Scene is authored as an Isaac Sim USD kitchen with textures, physics joints, and collisions.",
            "local_runtime_ready": local_ready,
        },
        "scenario_count": len(scenarios),
        "scenario_manifest": str(scenario_manifest_path),
        "runtime_preflight_manifest": str(preflight_path),
        "isaac_execution_request": str(execution_request_path),
        "isaac_runner_script": str(runner_script),
        "scenario_preview_frames": preview_records,
        "scenario_execution_status": "not_executed",
        "videos_generated": False,
        "unitree_g1_spawned_in_lightwheel_kitchen": False,
        "unitree_g1_navigation_proven": False,
        "blockers": blockers,
        "next_external_step": (
            "Run the generated Isaac request on Linux x86_64 with Isaac Sim/Lab, an RTX/RT-core "
            "NVIDIA GPU, NVIDIA Container Toolkit or native Isaac install, NGC access if using "
            "containers, and a verified Unitree G1 USD or Isaac MJCF importer binding."
        ),
        "proof_boundary": {
            "scenario_specs_authored": True,
            "lightwheel_usd_scene_materialized": True,
            "isaac_sim_execution_proven": False,
            "isaac_lab_execution_proven": False,
            "simulated_unitree_g1_navigation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "real_robot_pov": False,
            "customer_delivery_readiness_proven": False,
        },
    }
    _safe_write_json(handoff_path, handoff)

    provider_execution_request_path = out_dir / PROVIDER_REQUEST_NAME
    _write_provider_request(
        path=provider_execution_request_path,
        scenarios=scenarios,
        unitree_g1_mjcf=unitree_mjcf_summary,
    )
    provider_bundle = _write_provider_bundle(
        output_dir=out_dir,
        source_zip=zip_path,
        unitree_g1_mjcf_root=unitree_mjcf_root,
        runner_script=runner_script,
        local_execution_request_path=execution_request_path,
        provider_execution_request_path=provider_execution_request_path,
        scenario_manifest_path=scenario_manifest_path,
        runtime_preflight_path=preflight_path,
        handoff_path=handoff_path,
    )
    provider_proof_path = (
        Path(provider_proof).expanduser().resolve()
        if provider_proof
        else None
    )
    provider_private_signed_url_path = (
        Path(provider_private_signed_url_file).expanduser().resolve()
        if provider_private_signed_url_file
        else None
    )
    provider_packet = _write_provider_packet(
        output_dir=out_dir,
        source_zip=zip_path,
        bundle=provider_bundle,
        unitree_g1_mjcf=unitree_mjcf_summary,
        provider_execution_request_path=provider_execution_request_path,
        runtime=runtime,
        provider_proof_path=provider_proof_path,
        provider_artifact_root_uri=provider_artifact_root_uri,
        upload_provider_packet=upload_provider_packet,
        provider_private_signed_url_file=provider_private_signed_url_path,
        repo_commit=repo_commit,
        selected_isaac_runtime_image_ref=isaac_runtime_image_ref,
    )
    runpod_direct_launch_request = _write_runpod_direct_launch_request(
        output_dir=out_dir,
        provider_packet=provider_packet,
        provider_private_signed_url_file=provider_private_signed_url_path,
    )
    final_outputs = _write_blocked_execution_outputs(
        output_dir=out_dir,
        scenarios=scenarios,
        scene_usd=scene_usd,
        runtime_preflight_path=preflight_path,
        handoff_path=handoff_path,
        execution_request_path=execution_request_path,
        blockers=blockers,
        runtime=runtime,
        usdchecker=usdchecker,
        dependency_audit=dependency_audit,
        stage_summary=stage_summary,
        provider_proof_path=provider_proof_path,
        provider_packet_path=out_dir / PROVIDER_PACKET_NAME,
    )
    handoff["final_output_manifests"] = final_outputs
    handoff["scenario_execution_status"] = "blocked_not_executed"
    handoff["provider_packet_manifest"] = str(out_dir / PROVIDER_PACKET_NAME)
    handoff["provider_bundle"] = provider_bundle
    handoff["provider_packet_status"] = provider_packet.get("status")
    handoff["provider_packet_blockers"] = provider_packet.get("blockers")
    handoff["runpod_direct_launch_request"] = runpod_direct_launch_request
    handoff["provider_probe_manifest"] = (
        str(provider_proof_path) if provider_proof_path else None
    )
    _safe_write_json(handoff_path, handoff)
    return {**handoff, "manifest_path": str(handoff_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root")
    parser.add_argument("--source-zip")
    parser.add_argument("--source-repo-root")
    parser.add_argument("--repo-commit", default=LIGHTWHEEL_DEFAULT_COMMIT)
    parser.add_argument("--output-dir")
    parser.add_argument("--asset-output-dir")
    parser.add_argument("--provider-proof")
    parser.add_argument("--provider-artifact-root-uri")
    parser.add_argument("--upload-provider-packet", action="store_true")
    parser.add_argument("--provider-private-signed-url-file")
    parser.add_argument("--unitree-g1-mjcf-root")
    parser.add_argument("--isaac-runtime-image-ref")
    args = parser.parse_args(argv)
    result = build_lightwheel_kitchen_isaac_scenarios(
        capture_root=args.capture_root or os.environ.get("BLUEPRINT_CAPTURE_ROOT"),
        source_zip=args.source_zip,
        source_repo_root=args.source_repo_root,
        output_dir=args.output_dir,
        asset_output_dir=args.asset_output_dir,
        provider_proof=args.provider_proof,
        provider_artifact_root_uri=args.provider_artifact_root_uri,
        upload_provider_packet=args.upload_provider_packet,
        provider_private_signed_url_file=args.provider_private_signed_url_file,
        unitree_g1_mjcf_root=args.unitree_g1_mjcf_root,
        isaac_runtime_image_ref=args.isaac_runtime_image_ref,
        repo_commit=args.repo_commit,
    )
    print(json.dumps({"status": result["status"], "manifest_path": result["manifest_path"]}, indent=2))
    return 0 if result.get("status") == "ready_for_isaac_execution" else 2


if __name__ == "__main__":
    raise SystemExit(main())
