"""Bind the exact ADP-009A Content Agents bundle to successful CLI dry-runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp_content_agents_vast import SOURCE_COMMIT, SOURCE_TREE
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp_content_agents_bundle_config_preflight.v1"
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
REQUIRED_MODELS = ("gpt-4.1", "gpt-image-1")
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
USD_BBOX_SCRIPT = (
    "from pxr import Usd,UsdGeom;"
    "s=Usd.Stage.Open('/bundle/provider_runtime/input/"
    "adp009a_840313_canned_beverage_control.usda');"
    "p=UsdGeom.Mesh.Get(s,'/canned_beverage/visuals/body');"
    "r=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])"
    ".ComputeWorldBound(p.GetPrim()).ComputeAlignedRange();"
    "assert p.ComputePurpose()==UsdGeom.Tokens.default_ and not r.IsEmpty();"
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
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ContentAgentsBundlePreflightError("bundle_zip_path_traversal")
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise ContentAgentsBundlePreflightError("bundle_zip_invalid") from exc


def _secret() -> str:
    for name in SECRET_ENV_NAMES:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    path = Path("~/.blueprint-secrets/openai_api_key").expanduser()
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _probe_model_access(secret: str) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for model in REQUIRED_MODELS:
        request = urllib.request.Request(
            "https://api.openai.com/v1/models/" + model,
            headers={
                "Authorization": "Bearer " + secret,
                "User-Agent": "BlueprintContentAgentsPreflight/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.loads(response.read())
                status = int(response.status)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise ContentAgentsBundlePreflightError(
                f"openai_model_access_probe_failed:{model}"
            ) from exc
        if status != 200 or not isinstance(value, Mapping) or value.get("id") != model:
            raise ContentAgentsBundlePreflightError(
                f"openai_model_access_probe_failed:{model}"
            )
        models[model] = {"http_status": status, "returned_id": str(value["id"])}
    return {"provider": "openai", "models": models, "paid_inference_performed": False}


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


def materialize_bundle_config_preflight(
    *,
    bundle_receipt_path: str | Path,
    evidence_dir: str | Path,
    docker: str = "docker",
    image: str = LOCAL_IMAGE,
    generated_at: str | None = None,
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
    secret = _secret()
    if not secret:
        raise ContentAgentsBundlePreflightError("openai_secret_missing")
    model_access = _probe_model_access(secret)
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
        bbox_arguments = ["-c", USD_BBOX_SCRIPT]
        bbox_command = [
            docker,
            "run",
            "--rm",
            "--platform",
            LOCAL_IMAGE_PLATFORM,
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
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "generated_by": "blueprint_pipeline.adp_content_agents_bundle_preflight",
        "orchestrator_source_identity": source_identity,
        "status": "passed",
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
        "provider_mutations_performed": 0,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    serialized = json.dumps(receipt, sort_keys=True)
    if secret in serialized:
        raise ContentAgentsBundlePreflightError("raw_secret_value_recorded")
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(output / "adp_content_agents_bundle_config_preflight.json", receipt)
    return receipt


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
        or preflight.get("model_access")
        != {
            "provider": "openai",
            "models": {
                model: {"http_status": 200, "returned_id": model}
                for model in REQUIRED_MODELS
            },
            "paid_inference_performed": False,
        }
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
        expected_bbox = {
            "entrypoint": "python",
            "arguments": ["-c", USD_BBOX_SCRIPT],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute exact Content Agents config dry-runs before GPU allocation."
    )
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args(argv)
    receipt = materialize_bundle_config_preflight(
        bundle_receipt_path=args.bundle_receipt,
        evidence_dir=args.evidence_dir,
        docker=args.docker,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
