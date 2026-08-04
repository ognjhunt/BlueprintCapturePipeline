"""Canonical zero-retry Vast execution for the bounded ADP-009A Content Agents case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import yaml

from .common import ensure_dir, utc_now_iso, write_json
from .paid_resource_admission import PaidResourceAdmissionGrant
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-usd-content-agents"
RESULT_SCHEMA_VERSION = "adp_content_agents_vast_run.v1"
SOURCE_COMMIT = "36dbf3f274f8e256637230a05a085853f65cc175"
SOURCE_TREE = "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3"
SOURCE_VERSION = "0.5.2"
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10"
)
REFERENCE_IMAGE_SHA256 = (
    "sha256:80954198df572d782e095d8670e0d4e8ceea530c8fe53c8476a487d1aebe137f"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/content-agents"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
_FORWARDED_SECRET_NAMES = (
    "OPENAI_API_KEY",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_executable(path: Path, source: Path) -> None:
    shutil.copy2(source, path)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _validate_remote_configs(
    *, source: Path, config_sources: Mapping[str, Path]
) -> None:
    payloads = {
        name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, path in config_sources.items()
    }
    material_path = (
        source
        / "apps/material_agent/data/materials/material_libs_default/materials.yaml"
    )
    material = dict(payloads.get("material_agent.yaml") or {})
    texture = dict(payloads.get("texture_agent.yaml") or {})
    physics = dict(payloads.get("physics_agent.yaml") or {})
    texture_config = dict(texture.get("texture") or {})
    texture_spec = dict((texture.get("material_textures") or {}).get("green_can") or {})
    physics_steps = dict(physics.get("steps") or {})
    material_steps = dict(material.get("steps") or {})
    material_predict = dict(material_steps.get("predict") or {})
    material_vlm = dict(material_predict.get("vlm") or {})
    material_llm = dict(material_predict.get("llm") or {})
    material_validation = dict(material_steps.get("validate_input") or {})
    identify_vlm = dict((physics_steps.get("identify_asset") or {}).get("vlm") or {})
    identify_enabled = (physics_steps.get("identify_asset") or {}).get("enabled")
    predict_vlm = dict((physics_steps.get("predict") or {}).get("vlm") or {})
    if (
        not material_path.is_file()
        or (material.get("materials") or {}).get("path")
        != "../content_agents_source/apps/material_agent/data/materials/"
        "material_libs_default/materials.yaml"
        or texture_config.get("uv_target_prim_paths")
        != ["/canned_beverage/visuals/body"]
        or texture_spec.get("material_path")
        != "/canned_beverage/materials/green_can"
        or texture_spec.get("prim_paths") != ["/canned_beverage/visuals/body"]
        or texture_config.get("image_gen")
        != {"backend": "openai", "model": "gpt-image-1"}
        or material_validation.get("on_failure") != "warn"
        or (material_steps.get("validate_output") or {}).get("on_failure") != "warn"
        or material_vlm.get("backend") != "openai"
        or material_vlm.get("model") != "gpt-4.1"
        or material_llm.get("backend") != "openai"
        or material_llm.get("model") != "gpt-4.1"
        or identify_enabled is not False
        or identify_vlm.get("backend") != "openai"
        or identify_vlm.get("model") != "gpt-4.1"
        or predict_vlm.get("backend") != "openai"
        or predict_vlm.get("model") != "gpt-4.1"
    ):
        raise ValueError("adp_content_agents_remote_config_contract_invalid")


def _deterministic_zip(source_root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(source_root.parent).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def build_content_agents_vast_bundle(
    *,
    repo_root: str | Path,
    content_agents_root: str | Path,
    reference_image_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build one immutable bundle without licensed InteriorGS or SAGE bytes."""

    repo = Path(repo_root).expanduser().resolve()
    source = Path(content_agents_root).expanduser().resolve()
    reference_source = Path(reference_image_path).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_content_agents_bundle_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime / "configs")
    ensure_dir(runtime / "input")
    head = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    dirty = bool(_git(source, "status", "--porcelain"))
    if head != SOURCE_COMMIT or tree != SOURCE_TREE or dirty:
        raise ValueError("adp_content_agents_source_identity_mismatch")
    if (
        not reference_source.is_file()
        or _sha256(reference_source) != REFERENCE_IMAGE_SHA256
        or reference_source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
    ):
        raise ValueError("adp_content_agents_reference_image_identity_mismatch")

    source_zip = runtime / "content_agents_source.zip"
    subprocess.run(
        ["git", "-C", str(source), "archive", "--format=zip", f"--output={source_zip}", "HEAD"],
        check=True,
    )
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_content_agents_provider_runtime.sh",
        scripts / "run_adp_content_agents_provider_runtime.sh",
    )
    shutil.copy2(
        scripts / "adp_content_agents_provider_runner.py",
        runtime / "adp_content_agents_provider_runner.py",
    )
    assets = repo / "docs" / "arm_decision_proof_v1" / "assets"
    config_sources = {
        "material_agent.yaml": assets / "adp009a_content_agents_material.vast.yaml",
        "texture_agent.yaml": assets / "adp009a_content_agents_texture.vast.yaml",
        "physics_agent.yaml": assets / "adp009a_content_agents_physics.vast.yaml",
    }
    _validate_remote_configs(source=source, config_sources=config_sources)
    for name, path in config_sources.items():
        shutil.copy2(path, runtime / "configs" / name)
    usd_source = assets / "adp009a_840313_canned_beverage_control.usda"
    shutil.copy2(usd_source, runtime / "input" / usd_source.name)
    reference_name = "adp009a_840313_canned_beverage_control_reference.png"
    shutil.copy2(reference_source, runtime / "input" / reference_name)

    entrypoint = runtime / "run_adp_content_agents_provider_runtime.sh"
    runner = runtime / "adp_content_agents_provider_runner.py"
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_content_agents",
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=runner.read_text(encoding="utf-8"),
    )
    generated = generated_at or utc_now_iso()
    readiness = {
        "schema_version": "adp_content_agents_provider_bundle.v1",
        "generated_at": generated,
        "status": "ready" if not blockers else "blocked",
        "source_repository": "https://github.com/NVIDIA-Omniverse/usd-content-agents",
        "source_commit": head,
        "source_tree": tree,
        "source_version": SOURCE_VERSION,
        "container_image": DEFAULT_IMAGE,
        "container_platform": "linux/amd64",
        "source_archive_sha256": _sha256(source_zip),
        "input_usd_sha256": _sha256(usd_source),
        "reference_image_sha256": _sha256(reference_source),
        "reference_image_authority": "blueprint_cad_render_not_interiorgs_dataset_bytes",
        "runtime_entrypoint": "provider_runtime/run_adp_content_agents_provider_runtime.sh",
        "remote_config_contract_validated": True,
        "expected_output_filename": "adp_content_agents_vast_result.json",
        "material_agent_planned": True,
        "texture_agent_planned": True,
        "physics_agent_planned": True,
        "validation_agent_planned": True,
        "joint_agent_inapplicable_single_rigid_body": True,
        "local_bundle_ready_for_remote_staging": not blockers,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_content_agents_provider_manifest.json", readiness)
    bundle_path = job / "adp_content_agents_provider_runtime_bundle.zip"
    _deterministic_zip(runtime, bundle_path)
    receipt = {
        **readiness,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    write_json(job / "adp_content_agents_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger = _read_json(job / "adp_content_agents_vast_session_budget.json")
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(
        max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd
    )
    return max(0, min(runtime_minutes, spend_minutes))


def _extract(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {"status": "blocked", "blockers": ["content_agents_provider_output_zip_missing"]}
    if destination.exists() and any(destination.iterdir()):
        return {
            "status": "blocked",
            "blockers": ["content_agents_provider_output_destination_not_empty"],
        }
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("content_agents_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("content_agents_provider_output_zip_invalid")
    result_path = destination / "adp_content_agents_vast_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("content_agents_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


def _model_secret() -> str:
    for name in _FORWARDED_SECRET_NAMES:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    path = Path("~/.blueprint-secrets/openai_api_key").expanduser()
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


@contextmanager
def _authority_environment():
    names = (
        *_VAST_MUTATION_ENV,
        _VAST_SINGLE_ATTEMPT_ENV,
        *_FORWARDED_SECRET_NAMES,
        "BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS",
    )
    previous = {name: os.environ.get(name) for name in names}
    secret = _model_secret()
    if not secret:
        raise ValueError("adp_content_agents_openai_secret_missing")
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        for name in _FORWARDED_SECRET_NAMES:
            os.environ[name] = secret
        os.environ["BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS"] = ",".join(
            _FORWARDED_SECRET_NAMES
        )
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_content_agents_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 3.0,
    hard_ttl_seconds: int = 7200,
    public_image: str = DEFAULT_IMAGE,
) -> dict[str, Any]:
    """Run one Content Agents attempt and always require provider-zero afterward."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_content_agents_container_image_not_frozen")
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("adp_content_agents_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_content_agents_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_content_agents_paid_resource_admission_grant_missing")

    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 45:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["adp_content_agents_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["content_agents_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind="adp_content_agents",
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=64,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=remaining_minutes * 60,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "adp_content_agents_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090", "RTX A6000", "L40S", "A100"),
                prefer_isaac_rt=False,
                instance_label_prefix="blueprint-adp-content-agents-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_content_agents_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["content_agents_full_execution_not_completed"])
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("content_agents_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("content_agents_object_store_provider_zero_not_proven")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(provider_run / "vast_teardown_manifest.json"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    write_json(job / "adp_content_agents_vast_result.json", result)
    return result


__all__ = [
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "REFERENCE_IMAGE_SHA256",
    "build_content_agents_vast_bundle",
    "run_content_agents_vast",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable ADP-009A Content Agents Vast bundle."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--content-agents-root", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    receipt = build_content_agents_vast_bundle(
        repo_root=args.repo_root,
        content_agents_root=args.content_agents_root,
        reference_image_path=args.reference_image,
        job_dir=args.job_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
