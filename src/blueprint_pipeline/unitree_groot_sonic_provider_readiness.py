"""No-spend readiness audit for GR00T/SONIC provider comparison runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, write_json
from .secret_artifact_policy import (
    redacted_secret_file_status,
    secret_path_disclosure_policy,
)
from .unitree_groot_sonic_wam_image_remote_build_packet import DEFAULT_IMAGE_REF


SCHEMA_VERSION = "unitree_groot_sonic_provider_readiness.v1"
DEFAULT_OUTPUT = "output/unitree_groot_sonic_provider_readiness.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return {}
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _mapping(data)


def _file_status(path: str | Path) -> dict[str, Any]:
    return redacted_secret_file_status(
        path,
        path_source="default_blueprint_secret_file_path",
        raw_secret_field="raw_secret_values_recorded",
    )


def _registry_manifest_status(image_ref: str) -> dict[str, Any]:
    if not image_ref:
        return {
            "status": "missing",
            "blockers": ["missing_sealed_image_ref"],
            "image_ref": None,
        }
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", image_ref],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    payload: dict[str, Any] = {
        "image_ref": image_ref,
        "command": "docker buildx imagetools inspect --raw <image-ref>",
        "exit_code": result.returncode,
        "raw_registry_response_recorded": False,
    }
    if result.returncode != 0:
        payload.update(
            {
                "status": "missing",
                "blockers": ["sealed_image_not_registry_fetchable"],
                "stderr_tail": result.stderr[-500:],
            }
        )
        return payload
    try:
        manifest = json.loads(result.stdout)
    except Exception:
        manifest = {}
    payload.update(
        {
            "status": "present",
            "blockers": [],
            "media_type": manifest.get("mediaType"),
            "manifest_available": bool(manifest),
        }
    )
    return payload


def _provider_status(
    *,
    provider: str,
    token_file: str,
    image_present: bool,
    extra_blockers: list[str] | None = None,
) -> dict[str, Any]:
    token = _file_status(token_file)
    blockers: list[str] = []
    if not token["present"]:
        blockers.append(f"{provider}_credential_missing")
    if not image_present:
        blockers.append("sealed_image_not_registry_fetchable")
    blockers.extend(extra_blockers or [])
    return {
        "provider": provider,
        "status": "ready_for_paid_canary" if not blockers else "blocked_before_paid_canary",
        "blockers": sorted(set(blockers)),
        "credential": token,
        "sealed_image_registry_present": image_present,
        "paid_launch_allowed_by_readiness": not blockers,
    }


def build_provider_readiness(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    image_ref: str = DEFAULT_IMAGE_REF,
    remote_build_packet_manifest: str | Path | None = None,
    object_store_staging_manifest: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or _utc_now_iso()
    registry = _registry_manifest_status(image_ref)
    image_present = registry.get("status") == "present"
    remote_packet = _read_json(remote_build_packet_manifest)
    staging = _read_json(object_store_staging_manifest)
    remote_packet_ready = remote_packet.get("status") == "ready"
    staging_ready = staging.get("status") == "completed"

    global_blockers: list[str] = []
    if not image_present:
        global_blockers.append("sealed_image_not_registry_fetchable")
    if remote_build_packet_manifest and not remote_packet_ready:
        global_blockers.append("remote_build_packet_not_ready")
    if object_store_staging_manifest and not staging_ready:
        global_blockers.append("remote_build_packet_object_store_staging_not_ready")

    providers = {
        "runpod": _provider_status(
            provider="runpod",
            token_file="~/.blueprint-secrets/runpod_api_key",
            image_present=image_present,
        ),
        "digitalocean": _provider_status(
            provider="digitalocean",
            token_file="~/.blueprint-secrets/digitalocean_api_token",
            image_present=image_present,
        ),
    }
    paid_runtime_allowed = image_present and all(
        provider["paid_launch_allowed_by_readiness"] for provider in providers.values()
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_paid_provider_canaries"
        if paid_runtime_allowed and not global_blockers
        else "blocked_before_paid_provider_canaries",
        "blockers": sorted(set(global_blockers)),
        "image": registry,
        "remote_build_packet": {
            "manifest_path": str(Path(remote_build_packet_manifest).expanduser())
            if remote_build_packet_manifest
            else None,
            "status": remote_packet.get("status") if remote_packet else "not_provided",
            "tarball_path": remote_packet.get("tarball_path"),
            "ready": remote_packet_ready,
        },
        "object_store_staging": {
            "manifest_path": str(Path(object_store_staging_manifest).expanduser())
            if object_store_staging_manifest
            else None,
            "status": staging.get("status") if staging else "not_provided",
            "bundle_key": staging.get("bundle_key"),
            "expires_at": _mapping(staging.get("presigned_url_expiry")).get("expires_at"),
            "ready": staging_ready,
            "raw_url_values_recorded": False,
        },
        "providers": providers,
        "paid_runtime_comparison_allowed": paid_runtime_allowed and not global_blockers,
        "next_required_action": (
            "run_remote_build_packet_and_push_sealed_image"
            if not image_present
            else "run_paid_provider_startup_canaries_before_task_episode"
        ),
        "objective_status": {
            "both_provider_comparison_complete": False,
            "non_white_matte_g1_kitchen_task_success_proven": False,
            "semantic_task_success_pass_proven": False,
        },
        "raw_secret_values_recorded": False,
        "secret_artifact_policy": secret_path_disclosure_policy(),
        "claim_boundary": {
            "readiness_audit_is_no_spend": True,
            "readiness_audit_is_not_provider_startup": True,
            "readiness_audit_is_not_policy_inference": True,
            "readiness_audit_is_not_task_success": True,
        },
    }
    resolved_output = Path(output_path).expanduser()
    ensure_dir(resolved_output.parent)
    write_json(resolved_output, payload)
    payload["output_path"] = str(resolved_output)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--image-ref", default=DEFAULT_IMAGE_REF)
    parser.add_argument("--remote-build-packet-manifest")
    parser.add_argument("--object-store-staging-manifest")
    args = parser.parse_args(argv)
    payload = build_provider_readiness(
        output_path=args.output_path,
        image_ref=args.image_ref,
        remote_build_packet_manifest=args.remote_build_packet_manifest,
        object_store_staging_manifest=args.object_store_staging_manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
