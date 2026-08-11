"""Bind the exact ADP-009A Content Agents bundle to successful CLI dry-runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from pxr import Usd, UsdGeom

from .adp_content_agents_vast import (
    CONTENT_IMAGE_MODEL,
    CONTENT_LLM_MODEL,
    CONTENT_LLM_REASONING_EFFORT,
    SOURCE_COMMIT,
    SOURCE_TREE,
)
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .hosted_image_generation_preflight import (
    SCHEMA_VERSION as IMAGE_PREFLIGHT_SCHEMA_VERSION,
    materialize_hosted_image_generation_preflight,
)
from .hosted_model_inference_preflight import (
    PROBE_PROFILE as LLM_PROBE_PROFILE,
    REQUIRED_CAPABILITIES as LLM_REQUIRED_CAPABILITIES,
    SCHEMA_VERSION as LLM_PREFLIGHT_SCHEMA_VERSION,
    materialize_hosted_model_inference_preflight,
)
from .provider_archive import ProviderArchiveError, extract_provider_archive


SCHEMA_VERSION = "adp_content_agents_bundle_config_preflight.v2"
LOCAL_SCHEMA_VERSION = "adp_content_agents_local_bundle_config_preflight.v1"
STATIC_SCHEMA_VERSION = "adp_content_agents_static_bundle_config_preflight.v1"
LOCAL_IMAGE = "blueprint/adp009a-content-agents:0.5.2"
LOCAL_IMAGE_PLATFORM = "linux/arm64"
# A container build is not bit-reproducible: the base tag moves and uv
# re-resolves. Pinning one build output made the gate unrepeatable once that
# image was gone. Each admitted image ID is therefore recorded together with
# the recipe that produced it - the checked-in Dockerfile bytes, the base
# image, and the pinned Content Agents source tree - so a rebuild of the
# reviewed recipe can be admitted explicitly while a stray image still fails
# closed.
LOCAL_IMAGE_RECIPE = {
    "dockerfile_relative_path": (
        "docs/arm_decision_proof_v1/assets/"
        "adp009a_usd_content_agents_linux_arm64.Dockerfile"
    ),
    "dockerfile_sha256": (
        "sha256:9992a691a70c59448d2b11bb89213324405b93b3ffb50cb96e7c7141b4c3610e"
    ),
    "base_image": "ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
    "content_agents_source_tree": "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3",
}
LOCAL_IMAGE_ADMITTED_IDS = {
    # Original reviewed build (2026-08-06 tranche).
    "sha256:459fc2a13688d198a3c81faecd4e511ac14701d0e284e9a7bdf57587debea574": (
        LOCAL_IMAGE_RECIPE
    ),
    # Rebuild of the same recipe on 2026-08-09 after the first image was gone.
    "sha256:574b6650842081226da7e63e403e535bd7258aaa83b4f1b805882d067d181703": (
        LOCAL_IMAGE_RECIPE
    ),
}
SECRET_ENV_NAMES = ("OPENAI_API_KEY",)
REQUIRED_MODELS = (CONTENT_LLM_MODEL, CONTENT_IMAGE_MODEL)
ORCHESTRATOR_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_MEMBERS = {
    "material": "provider_runtime/configs/material_agent.yaml",
    "texture": "provider_runtime/configs/texture_agent.yaml",
    "physics": "provider_runtime/configs/physics_agent.yaml",
}
ENTRYPOINTS = {
    "material": "material-agent",
    "texture": "texture-agent",
    "physics": "physics-agent",
}
DRY_RUN_MARKERS = {
    "material": "Dry run complete",
    "texture": "Dry run -- execution plan",
    "physics": "Dry run complete",
}
MATERIAL_VALIDATE_MARKER = "Pipeline completed successfully"
USD_BBOX_MARKER = "BLUEPRINT_CONTENT_AGENTS_DEFAULT_PURPOSE_BBOX_OK"
LOCAL_NO_PAID_SECRET = "__BLUEPRINT_LOCAL_CONFIG_PREFLIGHT_NO_PAID_SECRET__"


def usd_bbox_script(input_usd_name: str) -> str:
    """Assert NVIDIA 0.5.2 can bound the bundle's own input at default purpose.

    The property is variant-neutral: every mesh must compute the default
    purpose and the default prim must have a non-empty default-purpose world
    bound. Naming one scene's prim path made this probe unusable for any other
    admitted input.
    """

    return (
        "from pxr import Usd,UsdGeom;"
        f"s=Usd.Stage.Open('/bundle/provider_runtime/input/{input_usd_name}');"
        "d=s.GetDefaultPrim();"
        "assert d.IsValid();"
        "ms=[q for q in s.Traverse() if q.IsA(UsdGeom.Mesh)];"
        "assert ms;"
        "assert all("
        "UsdGeom.Mesh(q).ComputePurpose()==UsdGeom.Tokens.default_ for q in ms);"
        "r=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])"
        ".ComputeWorldBound(d).ComputeAlignedRange();"
        "assert not r.IsEmpty();"
        f"print('{USD_BBOX_MARKER}')"
    )


class ContentAgentsBundlePreflightError(ValueError):
    """The exact-bundle preflight could not derive passing evidence."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContentAgentsBundlePreflightError("bundle_receipt_not_json_object")
    return dict(value)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    try:
        extract_provider_archive(archive_path, destination)
    except ProviderArchiveError as exc:
        raise ContentAgentsBundlePreflightError(
            f"bundle_zip_extraction_invalid:{exc}"
        ) from exc


def _secret() -> str:
    for name in SECRET_ENV_NAMES:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    path = Path("~/.blueprint-secrets/openai_api_key").expanduser()
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _probe_model_access(secret: str, output: Path) -> dict[str, Any]:
    """Prove real capabilities; a model-catalog lookup is not admission."""

    llm = materialize_hosted_model_inference_preflight(
        output_path=output / "content-agents-llm-capability.json",
        backend="openai",
        model=CONTENT_LLM_MODEL,
        reasoning_effort=CONTENT_LLM_REASONING_EFFORT,
        secret_loader=lambda _backend: (secret, "inherited_openai_secret"),
    )
    image = materialize_hosted_image_generation_preflight(
        output_path=output / "content-agents-image-capability.json",
        model=CONTENT_IMAGE_MODEL,
        secret_loader=lambda: (secret, "inherited_openai_secret"),
    )
    if llm.get("status") != "qualified":
        raise ContentAgentsBundlePreflightError(
            f"openai_model_capability_probe_failed:{CONTENT_LLM_MODEL}"
        )
    if image.get("status") != "qualified":
        raise ContentAgentsBundlePreflightError(
            f"openai_model_capability_probe_failed:{CONTENT_IMAGE_MODEL}"
        )
    return {
        "provider": "openai",
        "models": {
            CONTENT_LLM_MODEL: llm,
            CONTENT_IMAGE_MODEL: image,
        },
        "paid_inference_performed": True,
        "uploaded_scene_bytes": False,
    }


def _valid_model_access(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    models = value.get("models")
    if not isinstance(models, Mapping):
        return False
    llm = models.get(CONTENT_LLM_MODEL)
    image = models.get(CONTENT_IMAGE_MODEL)
    image_output = image.get("output") if isinstance(image, Mapping) else None
    if not isinstance(image_output, Mapping):
        image_output = {}
    return bool(
        value.get("provider") == "openai"
        and value.get("paid_inference_performed") is True
        and value.get("uploaded_scene_bytes") is False
        and isinstance(llm, Mapping)
        and llm.get("schema_version") == LLM_PREFLIGHT_SCHEMA_VERSION
        and llm.get("status") == "qualified"
        and llm.get("backend") == "openai"
        and llm.get("model") == CONTENT_LLM_MODEL
        and llm.get("reasoning_effort") == CONTENT_LLM_REASONING_EFFORT
        and llm.get("probe_profile") == LLM_PROBE_PROFILE
        and sorted(llm.get("verified_capabilities") or [])
        == sorted(LLM_REQUIRED_CAPABILITIES)
        and llm.get("blockers") in ([], None)
        and llm.get("receipt_digest")
        == canonical_digest(llm, digest_field="receipt_digest")
        and isinstance(image, Mapping)
        and image.get("schema_version") == IMAGE_PREFLIGHT_SCHEMA_VERSION
        and image.get("status") == "qualified"
        and image.get("provider") == "openai"
        and image.get("model") == CONTENT_IMAGE_MODEL
        and image_output.get("width") == 1024
        and image_output.get("height") == 1024
        and image_output.get("bytes_retained") is False
        and image.get("uploaded_scene_bytes") is False
        and image.get("blockers") in ([], None)
        and image.get("receipt_digest")
        == canonical_digest(image, digest_field="receipt_digest")
    )


def _redact(value: str, secrets: Sequence[str]) -> str:
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _admitted_local_image_record(value: Any) -> bool:
    """Accept only a recorded image whose ID and recipe are both admitted."""

    if not isinstance(value, Mapping):
        return False
    recipe = LOCAL_IMAGE_ADMITTED_IDS.get(str(value.get("id")))
    return bool(
        recipe is not None
        and value.get("reference") == LOCAL_IMAGE
        and value.get("platform") == LOCAL_IMAGE_PLATFORM
        and dict(value.get("recipe") or {}) == dict(recipe)
    )


def _inspect_image(*, docker: str, image: str) -> dict[str, Any]:
    result = subprocess.run(
        [docker, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ContentAgentsBundlePreflightError("local_preflight_image_missing")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContentAgentsBundlePreflightError("local_preflight_image_inspect_invalid") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
        raise ContentAgentsBundlePreflightError("local_preflight_image_inspect_invalid")
    value = dict(values[0])
    platform = f"{value.get('Os')}/{value.get('Architecture')}"
    recipe = LOCAL_IMAGE_ADMITTED_IDS.get(str(value.get("Id")))
    if recipe is None or platform != LOCAL_IMAGE_PLATFORM:
        raise ContentAgentsBundlePreflightError("local_preflight_image_identity_mismatch")
    return {
        "reference": image,
        "id": str(value["Id"]),
        "platform": platform,
        "recipe": dict(recipe),
    }


def _orchestrator_source_identity() -> dict[str, Any]:
    values: dict[str, str] = {}
    for role, arguments in (
        ("commit", ("rev-parse", "HEAD")),
        ("tree", ("rev-parse", "HEAD^{tree}")),
        ("dirty", ("status", "--porcelain")),
    ):
        result = subprocess.run(
            ["git", "-C", str(ORCHESTRATOR_REPO_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ContentAgentsBundlePreflightError("orchestrator_source_identity_missing")
        values[role] = result.stdout.strip()
    if values["dirty"]:
        raise ContentAgentsBundlePreflightError("orchestrator_source_checkout_dirty")
    return {"commit": values["commit"], "tree": values["tree"], "checkout_clean": True}


def _bundle_config_records(bundle_path: Path) -> dict[str, dict[str, Any]]:
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            members = set(archive.namelist())
            records: dict[str, dict[str, Any]] = {}
            for name, member in CONFIG_MEMBERS.items():
                if member not in members:
                    raise ContentAgentsBundlePreflightError(f"bundle_config_missing:{name}")
                value = archive.read(member)
                records[name] = {
                    "member": member,
                    "size_bytes": len(value),
                    "sha256": _sha256_bytes(value),
                }
            return records
    except zipfile.BadZipFile as exc:
        raise ContentAgentsBundlePreflightError("bundle_zip_invalid") from exc


def _bundle_static_input_records(bundle_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            members = set(archive.namelist())
            required = {
                "provider_runtime/content_agents_source.zip",
                "provider_runtime/input/reference.png",
                "provider_runtime/input/source_asset.usda",
                "provider_runtime/run_adp_content_agents_provider_runtime.sh",
                "provider_runtime/adp_content_agents_provider_runner.py",
            }
            missing = sorted(required - members)
            if missing:
                raise ContentAgentsBundlePreflightError(
                    "bundle_static_member_missing:" + ",".join(missing)
                )
            input_usds = sorted(
                name
                for name in members
                if name.startswith("provider_runtime/input/")
                and name.endswith((".usd", ".usda", ".usdc"))
            )
            if input_usds != ["provider_runtime/input/source_asset.usda"]:
                raise ContentAgentsBundlePreflightError(
                    "bundle_static_input_usd_ambiguous"
                )
            source_zip_bytes = archive.read("provider_runtime/content_agents_source.zip")
            return {
                "source_archive": {
                    "member": "provider_runtime/content_agents_source.zip",
                    "size_bytes": len(source_zip_bytes),
                    "sha256": _sha256_bytes(source_zip_bytes),
                },
                "reference_image": {
                    "member": "provider_runtime/input/reference.png",
                    "size_bytes": len(
                        archive.read("provider_runtime/input/reference.png")
                    ),
                    "sha256": _sha256_bytes(
                        archive.read("provider_runtime/input/reference.png")
                    ),
                },
                "input_usd": {
                    "member": "provider_runtime/input/source_asset.usda",
                    "size_bytes": len(
                        archive.read("provider_runtime/input/source_asset.usda")
                    ),
                    "sha256": _sha256_bytes(
                        archive.read("provider_runtime/input/source_asset.usda")
                    ),
                },
            }
    except zipfile.BadZipFile as exc:
        raise ContentAgentsBundlePreflightError("bundle_zip_invalid") from exc


def _bundle_config_semantics(bundle_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            rows: dict[str, Any] = {}
            target_paths: set[str] = set()
            for name, member in CONFIG_MEMBERS.items():
                try:
                    config = yaml.safe_load(archive.read(member))
                except yaml.YAMLError as exc:
                    raise ContentAgentsBundlePreflightError(
                        f"bundle_static_config_yaml_invalid:{name}"
                    ) from exc
                if not isinstance(config, Mapping):
                    raise ContentAgentsBundlePreflightError(
                        f"bundle_static_config_yaml_invalid:{name}"
                    )
                input_section = config.get("input")
                if not isinstance(input_section, Mapping):
                    raise ContentAgentsBundlePreflightError(
                        f"bundle_static_config_input_invalid:{name}"
                    )
                if input_section.get("usd_path") != "../input/source_asset.usda":
                    raise ContentAgentsBundlePreflightError(
                        f"bundle_static_config_usd_path_invalid:{name}"
                    )
                references = input_section.get("reference_images", [])
                if references and references != ["../input/reference.png"]:
                    raise ContentAgentsBundlePreflightError(
                        f"bundle_static_config_reference_invalid:{name}"
                    )
                for field in (
                    "target_prims",
                    "physics_target_prim_paths",
                    "appearance_target_prim_paths",
                ):
                    values = config.get(field)
                    if isinstance(values, list):
                        target_paths.update(str(value) for value in values)
                material_textures = config.get("material_textures")
                if isinstance(material_textures, Mapping):
                    for row in material_textures.values():
                        if isinstance(row, Mapping) and isinstance(
                            row.get("target_prim_paths"), list
                        ):
                            target_paths.update(
                                str(value) for value in row["target_prim_paths"]
                            )
                rows[name] = {
                    "member": member,
                    "usd_path": input_section.get("usd_path"),
                    "reference_images": list(references) if references else [],
                    "parsed_yaml_mapping": True,
                }
            return {
                "configs": rows,
                "declared_target_prim_paths": sorted(target_paths),
            }
    except zipfile.BadZipFile as exc:
        raise ContentAgentsBundlePreflightError("bundle_zip_invalid") from exc


def _inspect_input_usd_static(bundle_path: Path, target_prim_paths: Sequence[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="adp-content-agents-static-") as raw:
        expanded = Path(raw) / "bundle"
        ensure_dir(expanded)
        _safe_extract(bundle_path, expanded)
        source_zip = expanded / "provider_runtime/content_agents_source.zip"
        source = expanded / "provider_runtime/content_agents_source"
        ensure_dir(source)
        _safe_extract(source_zip, source)
        stage = Usd.Stage.Open(
            str(expanded / "provider_runtime/input/source_asset.usda")
        )
        if stage is None:
            raise ContentAgentsBundlePreflightError("bundle_static_input_usd_open_failed")
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            raise ContentAgentsBundlePreflightError("bundle_static_input_default_prim_missing")
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ContentAgentsBundlePreflightError("bundle_static_input_mesh_missing")
        non_default = [
            prim.GetPath().pathString
            for prim in meshes
            if UsdGeom.Mesh(prim).ComputePurpose() != UsdGeom.Tokens.default_
        ]
        if non_default:
            raise ContentAgentsBundlePreflightError(
                "bundle_static_input_mesh_purpose_invalid"
            )
        missing_targets = [
            path
            for path in target_prim_paths
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Mesh)
        ]
        if missing_targets:
            raise ContentAgentsBundlePreflightError(
                "bundle_static_input_target_mesh_missing:" + ",".join(missing_targets)
            )
        bound = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(default_prim).ComputeAlignedRange()
        if bound.IsEmpty():
            raise ContentAgentsBundlePreflightError(
                "bundle_static_input_default_purpose_bbox_empty"
            )
        return {
            "default_prim_path": default_prim.GetPath().pathString,
            "mesh_count": len(meshes),
            "default_purpose_mesh_count": len(meshes),
            "declared_target_mesh_count": len(target_prim_paths),
            "bbox_min": [float(value) for value in bound.GetMin()],
            "bbox_max": [float(value) for value in bound.GetMax()],
            "source_archive_extractable": True,
            "input_usd_opened": True,
        }


def materialize_static_bundle_config_preflight(
    *,
    bundle_receipt_path: str | Path,
    evidence_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Inspect exact Content Agents bundle/config/input bytes without Docker/API."""

    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    output = Path(evidence_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ContentAgentsBundlePreflightError("preflight_evidence_dir_not_empty")
    ensure_dir(output)
    bundle_receipt = _read_json(receipt_path)
    bundle_path = Path(str(bundle_receipt.get("bundle_path") or "")).expanduser().resolve()
    if (
        bundle_receipt.get("status") != "ready"
        or bundle_receipt.get("source_commit") != SOURCE_COMMIT
        or bundle_receipt.get("source_tree") != SOURCE_TREE
        or not bundle_path.is_file()
        or _sha256_file(bundle_path) != bundle_receipt.get("bundle_sha256")
    ):
        raise ContentAgentsBundlePreflightError("bundle_receipt_binding_invalid")
    config_records = _bundle_config_records(bundle_path)
    input_records = _bundle_static_input_records(bundle_path)
    config_semantics = _bundle_config_semantics(bundle_path)
    input_usd = _inspect_input_usd_static(
        bundle_path, config_semantics["declared_target_prim_paths"]
    )
    receipt: dict[str, Any] = {
        "schema_version": STATIC_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "generated_by": "blueprint_pipeline.adp_content_agents_bundle_preflight",
        "orchestrator_source_identity": _orchestrator_source_identity(),
        "status": "static_passed_docker_and_paid_model_access_not_checked",
        "bundle_receipt_path": str(receipt_path),
        "bundle_receipt_sha256": _sha256_file(receipt_path),
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256_file(bundle_path),
        "content_agents_source_commit": SOURCE_COMMIT,
        "content_agents_source_tree": SOURCE_TREE,
        "configs": config_records,
        "config_semantics": config_semantics,
        "input_records": input_records,
        "input_usd": input_usd,
        "docker_executed": False,
        "docker_network_disabled": None,
        "paid_model_access_required": False,
        "provider_mutations_performed": 0,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [
            "content_agents_local_docker_config_preflight_missing",
            "content_agents_paid_model_access_preflight_missing",
        ],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(output / "adp_content_agents_static_bundle_config_preflight.json", receipt)
    return receipt


def materialize_bundle_config_preflight(
    *,
    bundle_receipt_path: str | Path,
    evidence_dir: str | Path,
    docker: str = "docker",
    image: str = LOCAL_IMAGE,
    generated_at: str | None = None,
    require_paid_model_access: bool = True,
) -> dict[str, Any]:
    """Execute all three upstream dry-runs against the exact upload bundle."""

    if image != LOCAL_IMAGE:
        raise ContentAgentsBundlePreflightError("local_preflight_image_not_frozen")
    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    output = Path(evidence_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ContentAgentsBundlePreflightError("preflight_evidence_dir_not_empty")
    ensure_dir(output)
    bundle_receipt = _read_json(receipt_path)
    bundle_path = Path(str(bundle_receipt.get("bundle_path") or "")).expanduser().resolve()
    if (
        bundle_receipt.get("status") != "ready"
        or bundle_receipt.get("source_commit") != SOURCE_COMMIT
        or bundle_receipt.get("source_tree") != SOURCE_TREE
        or not bundle_path.is_file()
        or _sha256_file(bundle_path) != bundle_receipt.get("bundle_sha256")
    ):
        raise ContentAgentsBundlePreflightError("bundle_receipt_binding_invalid")
    config_records = _bundle_config_records(bundle_path)
    image_record = _inspect_image(docker=docker, image=image)
    source_identity = _orchestrator_source_identity()
    if require_paid_model_access:
        secret = _secret()
        if not secret:
            raise ContentAgentsBundlePreflightError("openai_secret_missing")
        model_access = _probe_model_access(secret, output)
    else:
        secret = LOCAL_NO_PAID_SECRET
        model_access = {
            "provider": "openai",
            "status": "not_executed",
            "models": {
                CONTENT_LLM_MODEL: {"status": "not_executed"},
                CONTENT_IMAGE_MODEL: {"status": "not_executed"},
            },
            "paid_inference_performed": False,
            "uploaded_scene_bytes": False,
            "blockers": ["content_agents_paid_model_access_preflight_missing"],
        }
    environment = os.environ.copy()
    for name in SECRET_ENV_NAMES:
        environment[name] = secret

    executions: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="adp-content-agents-preflight-", dir=output.parent) as raw:
        expanded = Path(raw) / "bundle"
        ensure_dir(expanded)
        _safe_extract(bundle_path, expanded)
        runtime = expanded / "provider_runtime"
        source_zip = runtime / "content_agents_source.zip"
        source = runtime / "content_agents_source"
        ensure_dir(source)
        _safe_extract(source_zip, source)
        for name in ("material", "texture", "physics"):
            command = [
                docker,
                "run",
                "--rm",
                "--platform",
                LOCAL_IMAGE_PLATFORM,
                *(["--network", "none"] if not require_paid_model_access else []),
                "-v",
                f"{expanded}:/bundle",
                "-w",
                "/bundle/provider_runtime",
            ]
            for env_name in SECRET_ENV_NAMES:
                command.extend(["-e", env_name])
            command.extend(
                [
                    "--entrypoint",
                    ENTRYPOINTS[name],
                    image,
                    "run",
                    "/bundle/" + CONFIG_MEMBERS[name],
                    "--dry-run",
                ]
            )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            log_text = _redact(result.stdout + result.stderr, (secret,))
            log_path = output / f"{ENTRYPOINTS[name]}.log"
            log_path.write_text(log_text, encoding="utf-8")
            marker = DRY_RUN_MARKERS[name]
            if result.returncode != 0:
                raise ContentAgentsBundlePreflightError(f"dry_run_failed:{name}")
            if marker not in log_text:
                raise ContentAgentsBundlePreflightError(f"dry_run_marker_missing:{name}")
            executions[name] = {
                "entrypoint": ENTRYPOINTS[name],
                "arguments": ["run", "/bundle/" + CONFIG_MEMBERS[name], "--dry-run"],
                "secret_environment_names_passed_by_name": list(SECRET_ENV_NAMES),
                "returncode": result.returncode,
                "required_marker": marker,
                "log_path": str(log_path),
                "log_size_bytes": log_path.stat().st_size,
                "log_sha256": _sha256_file(log_path),
            }
        validation_name = "material_validate_input"
        validation_arguments = [
            "run",
            "/bundle/" + CONFIG_MEMBERS["material"],
            "--only",
            "validate_input",
            "--clean",
        ]
        validation_command = [
            docker,
            "run",
                "--rm",
                "--platform",
                LOCAL_IMAGE_PLATFORM,
                *(["--network", "none"] if not require_paid_model_access else []),
                "-v",
                f"{expanded}:/bundle",
                "-w",
            "/bundle/provider_runtime",
            "-e",
            "OPENAI_API_KEY",
            "--entrypoint",
            ENTRYPOINTS["material"],
            image,
            *validation_arguments,
        ]
        validation_result = subprocess.run(
            validation_command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        validation_text = _redact(
            validation_result.stdout + validation_result.stderr, (secret,)
        )
        validation_log = output / "material-agent-validate-input.log"
        validation_log.write_text(validation_text, encoding="utf-8")
        if validation_result.returncode != 0:
            raise ContentAgentsBundlePreflightError("material_validate_input_failed")
        if MATERIAL_VALIDATE_MARKER not in validation_text:
            raise ContentAgentsBundlePreflightError(
                "material_validate_input_marker_missing"
            )
        executions[validation_name] = {
            "entrypoint": ENTRYPOINTS["material"],
            "arguments": validation_arguments,
            "secret_environment_names_passed_by_name": ["OPENAI_API_KEY"],
            "returncode": validation_result.returncode,
            "required_marker": MATERIAL_VALIDATE_MARKER,
            "log_path": str(validation_log),
            "log_size_bytes": validation_log.stat().st_size,
            "log_sha256": _sha256_file(validation_log),
        }
        input_names = sorted(
            path.name
            for path in (expanded / "provider_runtime/input").glob("*.usda")
        )
        if len(input_names) != 1:
            raise ContentAgentsBundlePreflightError(
                "usd_default_purpose_bbox_input_ambiguous"
            )
        bbox_arguments = ["-c", usd_bbox_script(input_names[0])]
        bbox_command = [
            docker,
            "run",
                "--rm",
                "--platform",
                LOCAL_IMAGE_PLATFORM,
                *(["--network", "none"] if not require_paid_model_access else []),
                "-v",
                f"{expanded}:/bundle",
                "--entrypoint",
            "python",
            image,
            *bbox_arguments,
        ]
        bbox_result = subprocess.run(
            bbox_command,
            check=False,
            capture_output=True,
            text=True,
        )
        bbox_text = bbox_result.stdout + bbox_result.stderr
        bbox_log = output / "usd-default-purpose-bbox.log"
        bbox_log.write_text(bbox_text, encoding="utf-8")
        if bbox_result.returncode != 0 or USD_BBOX_MARKER not in bbox_text:
            raise ContentAgentsBundlePreflightError("usd_default_purpose_bbox_probe_failed")
        executions["usd_default_purpose_bbox"] = {
            "entrypoint": "python",
            "arguments": bbox_arguments,
            "secret_environment_names_passed_by_name": [],
            "returncode": bbox_result.returncode,
            "required_marker": USD_BBOX_MARKER,
            "log_path": str(bbox_log),
            "log_size_bytes": bbox_log.stat().st_size,
            "log_sha256": _sha256_file(bbox_log),
        }

    receipt: dict[str, Any] = {
        "schema_version": (
            SCHEMA_VERSION if require_paid_model_access else LOCAL_SCHEMA_VERSION
        ),
        "generated_at": generated_at or utc_now_iso(),
        "generated_by": "blueprint_pipeline.adp_content_agents_bundle_preflight",
        "orchestrator_source_identity": source_identity,
        "status": (
            "passed"
            if require_paid_model_access
            else "local_passed_paid_model_access_not_checked"
        ),
        "bundle_receipt_path": str(receipt_path),
        "bundle_receipt_sha256": _sha256_file(receipt_path),
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256_file(bundle_path),
        "content_agents_source_commit": SOURCE_COMMIT,
        "content_agents_source_tree": SOURCE_TREE,
        "local_container_image": image_record,
        "model_access": model_access,
        "configs": config_records,
        "executions": executions,
        "all_required_dry_runs_executed": True,
        "docker_network_disabled": not require_paid_model_access,
        "paid_model_access_required": require_paid_model_access,
        "provider_mutations_performed": 0,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": (
            []
            if require_paid_model_access
            else ["content_agents_paid_model_access_preflight_missing"]
        ),
        "receipt_digest": "",
    }
    serialized = json.dumps(receipt, sort_keys=True)
    if secret in serialized:
        raise ContentAgentsBundlePreflightError("raw_secret_value_recorded")
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(output / "adp_content_agents_bundle_config_preflight.json", receipt)
    return receipt


def materialize_local_bundle_config_preflight(
    *,
    bundle_receipt_path: str | Path,
    evidence_dir: str | Path,
    docker: str = "docker",
    image: str = LOCAL_IMAGE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run only local Content Agents bundle/config/container checks.

    This deliberately does not probe OpenAI model or image-generation access and
    runs container checks with Docker networking disabled. It is sufficient to
    clear locally decidable bundle/config blockers, but it is not an execution
    admission receipt.
    """

    return materialize_bundle_config_preflight(
        bundle_receipt_path=bundle_receipt_path,
        evidence_dir=evidence_dir,
        docker=docker,
        image=image,
        generated_at=generated_at,
        require_paid_model_access=False,
    )


def validate_bundle_config_preflight(
    *,
    preflight: Mapping[str, Any],
    prepared_bundle: Mapping[str, Any],
    preflight_receipt_path: str | Path,
    expected_orchestrator_source_commit: str,
) -> list[str]:
    """Re-derive every local binding used by paid admission."""

    blockers: list[str] = []
    preflight_path = Path(preflight_receipt_path).expanduser().resolve()
    evidence_root = preflight_path.parent
    bundle_path = Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
    source_identity = preflight.get("orchestrator_source_identity")
    if not isinstance(source_identity, Mapping):
        source_identity = {}
    if (
        preflight.get("schema_version") != SCHEMA_VERSION
        or preflight.get("generated_by")
        != "blueprint_pipeline.adp_content_agents_bundle_preflight"
        or preflight.get("status") != "passed"
        or source_identity.get("commit") != expected_orchestrator_source_commit
        or source_identity.get("checkout_clean") is not True
        or preflight.get("receipt_digest")
        != canonical_digest(preflight, digest_field="receipt_digest")
        or preflight.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or preflight.get("bundle_path") != str(bundle_path)
        or preflight.get("content_agents_source_commit") != SOURCE_COMMIT
        or preflight.get("content_agents_source_tree") != SOURCE_TREE
        or not _admitted_local_image_record(preflight.get("local_container_image"))
        or not _valid_model_access(preflight.get("model_access"))
        or preflight.get("all_required_dry_runs_executed") is not True
        or preflight.get("provider_mutations_performed") != 0
        or preflight.get("paid_resource_allocated") is not False
        or preflight.get("raw_secret_values_recorded") is not False
        or preflight.get("blockers") not in ([], None)
    ):
        blockers.append("adp_content_agents_config_preflight_binding_invalid")

    try:
        observed_configs = _bundle_config_records(bundle_path)
    except (OSError, ContentAgentsBundlePreflightError):
        observed_configs = {}
    if preflight.get("configs") != observed_configs:
        blockers.append("adp_content_agents_config_preflight_config_digest_mismatch")
    executions = preflight.get("executions")
    if not isinstance(executions, Mapping):
        blockers.append("adp_content_agents_config_preflight_execution_missing")
        executions = {}
    for name in ("material", "texture", "physics"):
        row = executions.get(name)
        if not isinstance(row, Mapping):
            blockers.append(f"adp_content_agents_config_preflight_execution_missing:{name}")
            continue
        log_path = Path(str(row.get("log_path") or "")).expanduser().resolve()
        if evidence_root != log_path.parent:
            blockers.append(f"adp_content_agents_config_preflight_log_outside_evidence:{name}")
        expected = {
            "entrypoint": ENTRYPOINTS[name],
            "arguments": ["run", "/bundle/" + CONFIG_MEMBERS[name], "--dry-run"],
            "secret_environment_names_passed_by_name": list(SECRET_ENV_NAMES),
            "returncode": 0,
            "required_marker": DRY_RUN_MARKERS[name],
            "log_path": str(log_path),
            "log_size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
            "log_sha256": _sha256_file(log_path) if log_path.is_file() else "",
        }
        log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        if dict(row) != expected or DRY_RUN_MARKERS[name] not in log_text:
            blockers.append(f"adp_content_agents_config_preflight_execution_invalid:{name}")
    validation_row = executions.get("material_validate_input")
    if not isinstance(validation_row, Mapping):
        blockers.append(
            "adp_content_agents_config_preflight_execution_missing:material_validate_input"
        )
    else:
        validation_log = Path(
            str(validation_row.get("log_path") or "")
        ).expanduser().resolve()
        if evidence_root != validation_log.parent:
            blockers.append(
                "adp_content_agents_config_preflight_log_outside_evidence:material_validate_input"
            )
        validation_arguments = [
            "run",
            "/bundle/" + CONFIG_MEMBERS["material"],
            "--only",
            "validate_input",
            "--clean",
        ]
        expected_validation = {
            "entrypoint": ENTRYPOINTS["material"],
            "arguments": validation_arguments,
            "secret_environment_names_passed_by_name": ["OPENAI_API_KEY"],
            "returncode": 0,
            "required_marker": MATERIAL_VALIDATE_MARKER,
            "log_path": str(validation_log),
            "log_size_bytes": (
                validation_log.stat().st_size if validation_log.is_file() else 0
            ),
            "log_sha256": (
                _sha256_file(validation_log) if validation_log.is_file() else ""
            ),
        }
        validation_text = (
            validation_log.read_text(encoding="utf-8")
            if validation_log.is_file()
            else ""
        )
        if (
            dict(validation_row) != expected_validation
            or MATERIAL_VALIDATE_MARKER not in validation_text
        ):
            blockers.append(
                "adp_content_agents_config_preflight_execution_invalid:material_validate_input"
            )
    bbox_row = executions.get("usd_default_purpose_bbox")
    if not isinstance(bbox_row, Mapping):
        blockers.append(
            "adp_content_agents_config_preflight_execution_missing:usd_default_purpose_bbox"
        )
    else:
        bbox_log = Path(str(bbox_row.get("log_path") or "")).expanduser().resolve()
        if evidence_root != bbox_log.parent:
            blockers.append(
                "adp_content_agents_config_preflight_log_outside_evidence:usd_default_purpose_bbox"
            )
        # The probe script is derived from the bundle's own input name, so the
        # validator re-derives it from the recorded arguments rather than a
        # single scene's constant.
        recorded_arguments = list(bbox_row.get("arguments") or [])
        recorded_script = recorded_arguments[1] if len(recorded_arguments) == 2 else ""
        input_name = ""
        marker = "/bundle/provider_runtime/input/"
        if marker in recorded_script:
            input_name = recorded_script.split(marker, 1)[1].split("'", 1)[0]
        expected_bbox = {
            "entrypoint": "python",
            "arguments": ["-c", usd_bbox_script(input_name)] if input_name else [],
            "secret_environment_names_passed_by_name": [],
            "returncode": 0,
            "required_marker": USD_BBOX_MARKER,
            "log_path": str(bbox_log),
            "log_size_bytes": bbox_log.stat().st_size if bbox_log.is_file() else 0,
            "log_sha256": _sha256_file(bbox_log) if bbox_log.is_file() else "",
        }
        bbox_text = bbox_log.read_text(encoding="utf-8") if bbox_log.is_file() else ""
        if dict(bbox_row) != expected_bbox or USD_BBOX_MARKER not in bbox_text:
            blockers.append(
                "adp_content_agents_config_preflight_execution_invalid:usd_default_purpose_bbox"
            )
    return sorted(set(blockers))


def validate_local_bundle_config_preflight(
    *,
    preflight: Mapping[str, Any],
    prepared_bundle: Mapping[str, Any],
    preflight_receipt_path: str | Path,
    expected_orchestrator_source_commit: str,
) -> list[str]:
    """Re-derive every local no-paid binding used before model-access admission."""

    blockers: list[str] = []
    preflight_path = Path(preflight_receipt_path).expanduser().resolve()
    evidence_root = preflight_path.parent
    bundle_path = Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
    source_identity = preflight.get("orchestrator_source_identity")
    if not isinstance(source_identity, Mapping):
        source_identity = {}
    model_access = preflight.get("model_access")
    if not isinstance(model_access, Mapping):
        model_access = {}
    if (
        preflight.get("schema_version") != LOCAL_SCHEMA_VERSION
        or preflight.get("generated_by")
        != "blueprint_pipeline.adp_content_agents_bundle_preflight"
        or preflight.get("status") != "local_passed_paid_model_access_not_checked"
        or source_identity.get("commit") != expected_orchestrator_source_commit
        or source_identity.get("checkout_clean") is not True
        or preflight.get("receipt_digest")
        != canonical_digest(preflight, digest_field="receipt_digest")
        or preflight.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or preflight.get("bundle_path") != str(bundle_path)
        or preflight.get("content_agents_source_commit") != SOURCE_COMMIT
        or preflight.get("content_agents_source_tree") != SOURCE_TREE
        or not _admitted_local_image_record(preflight.get("local_container_image"))
        or model_access.get("paid_inference_performed") is not False
        or model_access.get("uploaded_scene_bytes") is not False
        or model_access.get("blockers")
        != ["content_agents_paid_model_access_preflight_missing"]
        or preflight.get("all_required_dry_runs_executed") is not True
        or preflight.get("docker_network_disabled") is not True
        or preflight.get("paid_model_access_required") is not False
        or preflight.get("provider_mutations_performed") != 0
        or preflight.get("paid_resource_allocated") is not False
        or preflight.get("raw_secret_values_recorded") is not False
        or preflight.get("blockers")
        != ["content_agents_paid_model_access_preflight_missing"]
    ):
        blockers.append("adp_content_agents_local_config_preflight_binding_invalid")

    try:
        observed_configs = _bundle_config_records(bundle_path)
    except (OSError, ContentAgentsBundlePreflightError):
        observed_configs = {}
    if preflight.get("configs") != observed_configs:
        blockers.append(
            "adp_content_agents_local_config_preflight_config_digest_mismatch"
        )
    executions = preflight.get("executions")
    if not isinstance(executions, Mapping):
        blockers.append("adp_content_agents_local_config_preflight_execution_missing")
        executions = {}
    for name in ("material", "texture", "physics"):
        row = executions.get(name)
        if not isinstance(row, Mapping):
            blockers.append(
                f"adp_content_agents_local_config_preflight_execution_missing:{name}"
            )
            continue
        log_path = Path(str(row.get("log_path") or "")).expanduser().resolve()
        if evidence_root != log_path.parent:
            blockers.append(
                f"adp_content_agents_local_config_preflight_log_outside_evidence:{name}"
            )
        expected = {
            "entrypoint": ENTRYPOINTS[name],
            "arguments": ["run", "/bundle/" + CONFIG_MEMBERS[name], "--dry-run"],
            "secret_environment_names_passed_by_name": list(SECRET_ENV_NAMES),
            "returncode": 0,
            "required_marker": DRY_RUN_MARKERS[name],
            "log_path": str(log_path),
            "log_size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
            "log_sha256": _sha256_file(log_path) if log_path.is_file() else "",
        }
        log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        if dict(row) != expected or DRY_RUN_MARKERS[name] not in log_text:
            blockers.append(
                f"adp_content_agents_local_config_preflight_execution_invalid:{name}"
            )
    validation_row = executions.get("material_validate_input")
    if not isinstance(validation_row, Mapping):
        blockers.append(
            "adp_content_agents_local_config_preflight_execution_missing:"
            "material_validate_input"
        )
    else:
        validation_log = Path(
            str(validation_row.get("log_path") or "")
        ).expanduser().resolve()
        if evidence_root != validation_log.parent:
            blockers.append(
                "adp_content_agents_local_config_preflight_log_outside_evidence:"
                "material_validate_input"
            )
        validation_arguments = [
            "run",
            "/bundle/" + CONFIG_MEMBERS["material"],
            "--only",
            "validate_input",
            "--clean",
        ]
        expected_validation = {
            "entrypoint": ENTRYPOINTS["material"],
            "arguments": validation_arguments,
            "secret_environment_names_passed_by_name": ["OPENAI_API_KEY"],
            "returncode": 0,
            "required_marker": MATERIAL_VALIDATE_MARKER,
            "log_path": str(validation_log),
            "log_size_bytes": (
                validation_log.stat().st_size if validation_log.is_file() else 0
            ),
            "log_sha256": (
                _sha256_file(validation_log) if validation_log.is_file() else ""
            ),
        }
        validation_text = (
            validation_log.read_text(encoding="utf-8")
            if validation_log.is_file()
            else ""
        )
        if (
            dict(validation_row) != expected_validation
            or MATERIAL_VALIDATE_MARKER not in validation_text
        ):
            blockers.append(
                "adp_content_agents_local_config_preflight_execution_invalid:"
                "material_validate_input"
            )
    bbox_row = executions.get("usd_default_purpose_bbox")
    if not isinstance(bbox_row, Mapping):
        blockers.append(
            "adp_content_agents_local_config_preflight_execution_missing:"
            "usd_default_purpose_bbox"
        )
    else:
        bbox_log = Path(str(bbox_row.get("log_path") or "")).expanduser().resolve()
        if evidence_root != bbox_log.parent:
            blockers.append(
                "adp_content_agents_local_config_preflight_log_outside_evidence:"
                "usd_default_purpose_bbox"
            )
        recorded_arguments = list(bbox_row.get("arguments") or [])
        recorded_script = recorded_arguments[1] if len(recorded_arguments) == 2 else ""
        input_name = ""
        marker = "/bundle/provider_runtime/input/"
        if marker in recorded_script:
            input_name = recorded_script.split(marker, 1)[1].split("'", 1)[0]
        expected_bbox = {
            "entrypoint": "python",
            "arguments": ["-c", usd_bbox_script(input_name)] if input_name else [],
            "secret_environment_names_passed_by_name": [],
            "returncode": 0,
            "required_marker": USD_BBOX_MARKER,
            "log_path": str(bbox_log),
            "log_size_bytes": bbox_log.stat().st_size if bbox_log.is_file() else 0,
            "log_sha256": _sha256_file(bbox_log) if bbox_log.is_file() else "",
        }
        bbox_text = bbox_log.read_text(encoding="utf-8") if bbox_log.is_file() else ""
        if dict(bbox_row) != expected_bbox or USD_BBOX_MARKER not in bbox_text:
            blockers.append(
                "adp_content_agents_local_config_preflight_execution_invalid:"
                "usd_default_purpose_bbox"
            )
    return sorted(set(blockers))


def validate_static_bundle_config_preflight(
    *,
    preflight: Mapping[str, Any],
    prepared_bundle: Mapping[str, Any],
    preflight_receipt_path: str | Path,
    expected_orchestrator_source_commit: str,
) -> list[str]:
    """Re-derive Dockerless static bundle/config/input-USD bindings."""

    blockers: list[str] = []
    bundle_path = Path(str(prepared_bundle.get("bundle_path") or "")).expanduser().resolve()
    source_identity = preflight.get("orchestrator_source_identity")
    if not isinstance(source_identity, Mapping):
        source_identity = {}
    if (
        preflight.get("schema_version") != STATIC_SCHEMA_VERSION
        or preflight.get("generated_by")
        != "blueprint_pipeline.adp_content_agents_bundle_preflight"
        or preflight.get("status")
        != "static_passed_docker_and_paid_model_access_not_checked"
        or source_identity.get("commit") != expected_orchestrator_source_commit
        or source_identity.get("checkout_clean") is not True
        or preflight.get("receipt_digest")
        != canonical_digest(preflight, digest_field="receipt_digest")
        or preflight.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or preflight.get("bundle_path") != str(bundle_path)
        or preflight.get("content_agents_source_commit") != SOURCE_COMMIT
        or preflight.get("content_agents_source_tree") != SOURCE_TREE
        or preflight.get("docker_executed") is not False
        or preflight.get("paid_model_access_required") is not False
        or preflight.get("provider_mutations_performed") != 0
        or preflight.get("paid_resource_allocated") is not False
        or preflight.get("raw_secret_values_recorded") is not False
        or preflight.get("blockers")
        != [
            "content_agents_local_docker_config_preflight_missing",
            "content_agents_paid_model_access_preflight_missing",
        ]
    ):
        blockers.append("adp_content_agents_static_config_preflight_binding_invalid")
    try:
        observed_configs = _bundle_config_records(bundle_path)
        input_records = _bundle_static_input_records(bundle_path)
        config_semantics = _bundle_config_semantics(bundle_path)
        input_usd = _inspect_input_usd_static(
            bundle_path, config_semantics["declared_target_prim_paths"]
        )
    except (OSError, ContentAgentsBundlePreflightError):
        observed_configs = {}
        input_records = {}
        config_semantics = {}
        input_usd = {}
    if preflight.get("configs") != observed_configs:
        blockers.append(
            "adp_content_agents_static_config_preflight_config_digest_mismatch"
        )
    if preflight.get("input_records") != input_records:
        blockers.append(
            "adp_content_agents_static_config_preflight_input_digest_mismatch"
        )
    if preflight.get("config_semantics") != config_semantics:
        blockers.append(
            "adp_content_agents_static_config_preflight_semantics_mismatch"
        )
    if preflight.get("input_usd") != input_usd:
        blockers.append("adp_content_agents_static_config_preflight_usd_mismatch")
    return sorted(set(blockers))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute exact Content Agents config dry-runs before GPU allocation."
    )
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument(
        "--local-no-paid-model-access",
        action="store_true",
        help=(
            "Run only local bundle/container/config checks with Docker networking "
            "disabled; do not probe paid model or image-generation access."
        ),
    )
    parser.add_argument(
        "--static-no-docker-no-paid-model-access",
        action="store_true",
        help=(
            "Inspect bundle/config/input USD bytes only; do not run Docker and "
            "do not probe paid model or image-generation access."
        ),
    )
    args = parser.parse_args(argv)
    if args.local_no_paid_model_access and args.static_no_docker_no_paid_model_access:
        raise ContentAgentsBundlePreflightError("preflight_mode_ambiguous")
    if args.static_no_docker_no_paid_model_access:
        receipt = materialize_static_bundle_config_preflight(
            bundle_receipt_path=args.bundle_receipt,
            evidence_dir=args.evidence_dir,
        )
    elif args.local_no_paid_model_access:
        receipt = materialize_local_bundle_config_preflight(
            bundle_receipt_path=args.bundle_receipt,
            evidence_dir=args.evidence_dir,
            docker=args.docker,
        )
    else:
        receipt = materialize_bundle_config_preflight(
            bundle_receipt_path=args.bundle_receipt,
            evidence_dir=args.evidence_dir,
            docker=args.docker,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
