"""Live pipeline setup and readiness preflight.

This command does not start provider jobs, upload data, or run live SDK
operators by default. It answers the operational question: what can run from
this machine now, what is configured through local env files, and what exact
inputs are still missing before the proof-bounded Arena/package pipeline can
cross into live external execution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .agent_operator_runtime import (
    CODEX_CLI_HOST_OAUTH_ENV,
    LIVE_AGENTS_SDK_ENV,
    LIVE_CODEX_SDK_ENV,
    codex_cli_path,
)
from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .safe_env import load_env_files


LIVE_PIPELINE_SETUP_SCHEMA_VERSION = "blueprint_live_pipeline_setup.v1"

OPENAI_AUTH_FACTS: Dict[str, Any] = {
    "repo_cli_api_auth": {
        "auth_method": "OPENAI_API_KEY",
        "usable_by_repo_subprocess": True,
        "secret_value_exported": False,
        "source": "OpenAI API authentication uses bearer API keys for server/CLI requests.",
    },
    "chatgpt_pro_oauth": {
        "auth_method": "host_oauth_or_chatgpt_subscription",
        "usable_by_repo_subprocess": False,
        "secret_value_exported": False,
        "source": (
            "ChatGPT Pro/Codex OAuth can be used by the host application tools where those "
            "tools are triggered, but this repo cannot read or forward that OAuth token."
        ),
    },
    "codex_cli_host_oauth": {
        "auth_method": "codex_cli_authenticated_by_host_or_user_profile",
        "usable_by_repo_subprocess": True,
        "secret_value_exported": False,
        "required_gate": CODEX_CLI_HOST_OAUTH_ENV,
        "source": (
            "The repo can invoke an installed Codex CLI when explicitly gated; auth remains "
            "owned by the Codex CLI profile rather than exported into pipeline manifests."
        ),
    },
    "command_hook_oauth": {
        "auth_method": "external_command_or_connector_managed_oauth",
        "usable_by_repo_subprocess": True,
        "secret_value_exported": False,
        "source": (
            "Provider-specific OAuth is allowed when the configured vision/delivery/operator "
            "command owns the OAuth flow and returns deterministic artifacts."
        ),
    },
}

CONTROL_PLANE_NOT_PROOF = {
    "runner_role": "optional_always_on_control_plane",
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "notes": [
        "A CPU droplet can schedule jobs, host manifests, sync repos, and run watchdogs.",
        "It is not GPU/Arena execution proof unless owner-system simulator logs and artifacts are ingested.",
    ],
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return _truthy(os.getenv(name))


def _module_available(candidates: Sequence[str]) -> bool:
    return any(importlib.util.find_spec(candidate) is not None for candidate in candidates)


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _env_status(name: str, *, truthy_required: bool = False) -> Dict[str, Any]:
    present = bool(_string(os.getenv(name)))
    truthy = _env_truthy(name)
    return {
        "name": name,
        "present": present,
        "truthy": truthy,
        "ready": truthy if truthy_required else present,
        "value_redacted": present,
    }


def _first_executable(command: str | None) -> str | None:
    text = _string(command)
    if not text:
        return None
    try:
        parts = shlex.split(text)
    except ValueError:
        return None
    if not parts:
        return None
    if "=" in parts[0] and len(parts) > 1:
        return parts[1]
    return parts[0]


def _command_status(command: str | None) -> Dict[str, Any]:
    executable = _first_executable(command)
    if not executable:
        return {
            "configured": False,
            "executable": None,
            "executable_found": False,
            "ready": False,
        }
    found = bool(shutil.which(executable)) if "/" not in executable else Path(executable).exists()
    return {
        "configured": True,
        "executable": executable,
        "executable_found": found,
        "ready": found,
    }


def _artifact_status(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "ready": False}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "ready": path.is_file(),
    }


def _arena_results_status(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "arena_results_dir": None,
            "status": "not_configured",
            "ready": False,
            "blockers": ["arena_results_dir_not_provided"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    if not path.is_dir():
        return {
            "arena_results_dir": str(path),
            "status": "blocked",
            "ready": False,
            "blockers": ["arena_results_dir_missing"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    json_artifacts = sorted(item for item in path.rglob("*.json") if item.is_file())
    recognized = [
        str(item.relative_to(path))
        for item in json_artifacts
        if item.name
        in {
            "rollout_manifest.json",
            "shard_manifest.json",
            "artifact_manifest.json",
            "metrics.json",
            "results.json",
        }
    ]
    if not json_artifacts:
        return {
            "arena_results_dir": str(path),
            "status": "blocked",
            "ready": False,
            "blockers": ["arena_results_dir_has_no_json_artifacts"],
            "json_artifact_count": 0,
            "recognized_artifacts": [],
        }
    return {
        "arena_results_dir": str(path),
        "status": "ready_for_ingest",
        "ready": True,
        "blockers": [],
        "json_artifact_count": len(json_artifacts),
        "recognized_artifacts": recognized,
        "proof_boundary": "existing result artifacts are ingest inputs, not simulator execution proof",
    }


def _capture_upstream_truth(capture_root: Path | None) -> Dict[str, Any]:
    if capture_root is None:
        return {
            "status": "not_checked",
            "fields_present": {},
            "blockers": ["capture_root_not_provided"],
        }
    descriptor = _read_optional_mapping(capture_root / "capture_descriptor.json")
    raw_manifest = _read_optional_mapping(capture_root / "raw" / "manifest.json")
    opportunity = _read_optional_mapping(capture_root / "pipeline" / "opportunity_handoff.json")
    sources = [descriptor, raw_manifest, opportunity]
    fields = ("site_submission_id", "request_id", "buyer_request_id", "capture_job_id")
    present: Dict[str, bool] = {}
    for field in fields:
        present[field] = any(bool(_string(source.get(field))) for source in sources)
    if not present["request_id"] and present["site_submission_id"]:
        present["request_id"] = True
    blockers = [f"missing_webapp_{field}" for field, ok in present.items() if not ok]
    return {
        "status": "ready" if not blockers else "blocked",
        "fields_present": present,
        "blockers": blockers,
    }


def _package_audit_status(package_dir: Path | None) -> Dict[str, Any]:
    if package_dir is None:
        return {
            "status": "not_checked",
            "blockers": ["package_dir_not_provided"],
            "artifact": _artifact_status(None),
        }
    audit_path = package_dir / "arena_package_proof_boundary_audit.json"
    audit = _read_optional_mapping(audit_path)
    blockers = list(audit.get("blockers") or []) if audit else ["arena_package_audit_missing"]
    return {
        "status": "ready" if audit.get("status") == "passed" else "blocked",
        "artifact": _artifact_status(audit_path),
        "audit_status": audit.get("status"),
        "blockers": blockers,
    }


def _digitalocean_read(
    *,
    allow_read: bool,
    token_env: str,
    droplet_name: str | None,
    droplet_ip: str | None,
    timeout_seconds: int,
) -> Dict[str, Any]:
    token_present = bool(_string(os.getenv(token_env)))
    base = {
        "provider": "digitalocean",
        "droplet_name": _string(droplet_name) or None,
        "droplet_ip": _string(droplet_ip) or None,
        "api_token_env": token_env,
        "api_token_present": token_present,
        "api_token_value_redacted": token_present,
        "api_read_allowed": allow_read,
        "control_plane_boundary": dict(CONTROL_PLANE_NOT_PROOF),
    }
    if not allow_read:
        return {
            **base,
            "status": "configured_advisory" if (droplet_name or droplet_ip) else "not_configured",
            "blockers": [],
            "notes": [
                "DigitalOcean API read is optional advisory control-plane evidence.",
                "Set the read gate and token only when droplet inventory verification is needed.",
            ],
        }
    blockers: List[str] = []
    if not token_present:
        blockers.append(f"missing_env_{token_env}")
    if not droplet_name and not droplet_ip:
        blockers.append("missing_droplet_name_or_ip")
    if blockers:
        return {**base, "status": "blocked", "blockers": blockers}

    query = {}
    if droplet_name:
        query["name"] = _string(droplet_name)
    url = "https://api.digitalocean.com/v2/droplets"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {os.environ[token_env]}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "blocked",
            "blockers": [f"digitalocean_api_read_failed:{type(exc).__name__}"],
        }

    droplets = payload.get("droplets") if isinstance(payload, Mapping) else []
    matches: List[Dict[str, Any]] = []
    for droplet in droplets if isinstance(droplets, list) else []:
        if not isinstance(droplet, Mapping):
            continue
        networks = droplet.get("networks") if isinstance(droplet.get("networks"), Mapping) else {}
        public_ips = [
            network.get("ip_address")
            for network in networks.get("v4", [])
            if isinstance(network, Mapping) and network.get("type") == "public"
        ]
        if droplet_ip and droplet_ip not in public_ips:
            continue
        matches.append(
            {
                "id": droplet.get("id"),
                "name": droplet.get("name"),
                "status": droplet.get("status"),
                "region": (droplet.get("region") or {}).get("slug")
                if isinstance(droplet.get("region"), Mapping)
                else None,
                "memory_mb": droplet.get("memory"),
                "vcpus": droplet.get("vcpus"),
                "disk_gb": droplet.get("disk"),
                "public_ipv4_present": bool(public_ips),
                "image": (droplet.get("image") or {}).get("slug")
                if isinstance(droplet.get("image"), Mapping)
                else None,
                "gpu_proof": False,
            }
        )
    return {
        **base,
        "status": "ready_control_plane" if matches else "blocked",
        "blockers": [] if matches else ["digitalocean_droplet_not_found"],
        "matches": matches,
    }


def _section(status: str, blockers: Sequence[str], **extra: Any) -> Dict[str, Any]:
    return {
        "status": status,
        "ready": status.startswith("ready"),
        "blockers": list(blockers),
        **extra,
    }


def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _restore_env(original_env: Mapping[str, str]) -> None:
    for key in list(os.environ):
        if key not in original_env:
            os.environ.pop(key, None)
    for key, value in original_env.items():
        os.environ[key] = value


def _overall_status(sections: Mapping[str, Mapping[str, Any]]) -> str:
    live_required = (
        "real_arena_execution",
        "rollout_vision_labeling",
        "delivery_upload",
        "live_agents_operator",
        "live_codex_operator",
        "webapp_upstream_truth",
    )
    if all(sections[name].get("ready") for name in live_required):
        return "ready_for_live_external_execution"
    if sections.get("local_deterministic_lane", {}).get("ready"):
        return "local_ready_live_external_blocked"
    return "blocked"


def _next_inputs_needed(sections: Mapping[str, Mapping[str, Any]]) -> List[str]:
    next_inputs = [
        "Export only the live gates you intend to use; do not enable spend/external actions globally."
    ]
    if not sections.get("real_arena_execution", {}).get("ready"):
        next_inputs.append(
            "Provide an owner-system Arena simulator command or existing Arena results directory."
        )
    if not sections.get("rollout_vision_labeling", {}).get("ready"):
        next_inputs.append(
            "Provide a vision-labeling command if model-derived rollout labels are required."
        )
    if not sections.get("delivery_upload", {}).get("ready"):
        next_inputs.append("Provide a delivery command if signed/uploaded package access is required.")
    if not (
        sections.get("live_agents_operator", {}).get("ready")
        and sections.get("live_codex_operator", {}).get("ready")
    ):
        next_inputs.append(
            "Install OpenAI Agents/Codex SDK dependencies and provide API credentials for repo CLI "
            "live operators, or explicitly allow Codex CLI host-OAuth execution."
        )
    next_inputs.append(
        "Use the DigitalOcean droplet as a control plane only unless GPU/Arena proof is produced "
        "and ingested."
    )
    return next_inputs


def build_live_pipeline_setup_manifest(
    *,
    capture_root: str | Path | None = None,
    package_dir: str | Path | None = None,
    arena_results_dir: str | Path | None = None,
    simulator_command: str | None = None,
    vision_labeling_command: str | None = None,
    delivery_command: str | None = None,
    load_local_env: bool = True,
    allow_digitalocean_read: bool = False,
    digitalocean_token_env: str = "DIGITALOCEAN_ACCESS_TOKEN",
    digitalocean_droplet_name: str | None = None,
    digitalocean_droplet_ip: str | None = None,
    output_path: str | Path | None = None,
    timeout_seconds: int = 15,
) -> Dict[str, Any]:
    original_env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        capture_path = Path(capture_root).resolve() if capture_root else None
        package_path = Path(package_dir).resolve() if package_dir else None
        if capture_path and not package_path:
            context = resolve_local_capture_context(capture_path)
            package_path = context.pipeline_root / "arena_eval_package"
        env_roots = _unique_paths(
            [repo_root, Path.cwd(), capture_path] if capture_path else [repo_root, Path.cwd()]
        )
        env_summary = (
            load_env_files(env_roots)
            if load_local_env
            else {
                "files": [],
                "loaded_keys": [],
                "skipped_existing_keys": [],
                "skipped_placeholder_keys": [],
            }
        )
        generated_at = utc_now_iso()

        env = {
            name: _env_status(name, truthy_required=True)
            for name in (
                "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
                "BLUEPRINT_ALLOW_GPU_PROVISIONING",
                "BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING",
                "BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD",
                LIVE_AGENTS_SDK_ENV,
                LIVE_CODEX_SDK_ENV,
                CODEX_CLI_HOST_OAUTH_ENV,
                "BLUEPRINT_ALLOW_AGENT_EXTERNAL_ACTIONS",
                "BLUEPRINT_ALLOW_AGENT_SPEND_ACTIONS",
            )
        }
        secrets = {
            name: _env_status(name)
            for name in (
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_GENAI_API_KEY",
                "WORLDLABS_API_KEY",
                "PIPELINE_SYNC_TOKEN",
                digitalocean_token_env,
            )
        }
        modules = {
            "openai_agents_sdk": _module_available(("agents", "openai_agents")),
            "openai_codex_sdk": _module_available(("openai_codex",)),
            "pandas": _module_available(("pandas",)),
            "pyarrow": _module_available(("pyarrow",)),
            "h5py": _module_available(("h5py",)),
            "rlds": _module_available(("rlds",)),
            "lerobot": _module_available(("lerobot",)),
        }
        commands = {
            "ffmpeg": {
                "ready": bool(shutil.which("ffmpeg")),
                "path_present": bool(shutil.which("ffmpeg")),
            },
            "simulator": _command_status(simulator_command),
            "vision_labeling": _command_status(
                vision_labeling_command or os.getenv("BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND")
            ),
            "delivery_upload": _command_status(
                delivery_command or os.getenv("BLUEPRINT_PACKAGE_DELIVERY_UPLOAD_COMMAND")
            ),
            "codex_cli": _command_status(codex_cli_path() or "codex"),
        }

        arena_results = Path(arena_results_dir).resolve() if arena_results_dir else None
        arena_result_artifacts = {
            **_arena_results_status(arena_results),
            "rollout_manifest": _artifact_status(arena_results / "rollout_manifest.json")
            if arena_results
            else _artifact_status(None),
            "shard_manifest": _artifact_status(arena_results / "shard_manifest.json")
            if arena_results
            else _artifact_status(None),
        }
        local_blockers: List[str] = []
        if not commands["ffmpeg"]["ready"]:
            local_blockers.append("missing_ffmpeg_for_clip_keyframe_paths")
        simulator_path_ready = (
            env["BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]["ready"]
            and commands["simulator"]["ready"]
        )
        arena_results_ready = bool(arena_result_artifacts["ready"])
        real_arena_blockers = [
            blocker
            for blocker, ok in (
                (
                    "missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION",
                    env["BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]["ready"] or arena_results_ready,
                ),
                (
                    "missing_simulator_command_or_arena_results_dir",
                    commands["simulator"]["configured"] or arena_results_ready,
                ),
                (
                    "simulator_command_executable_not_found",
                    commands["simulator"]["ready"] or arena_results_ready,
                ),
            )
            if not ok
        ]
        sections: Dict[str, Dict[str, Any]] = {
            "local_deterministic_lane": _section(
                "ready" if not local_blockers else "blocked",
                local_blockers,
                artifact_inputs=arena_result_artifacts,
                package_audit=_package_audit_status(package_path),
            ),
            "real_arena_execution": _section(
                "ready" if simulator_path_ready else "ready_for_result_ingest"
                if arena_results_ready
                else "blocked",
                real_arena_blockers,
                command=commands["simulator"],
                arena_results=arena_result_artifacts,
                claim_boundary=dict(CONTROL_PLANE_NOT_PROOF),
                proof_boundary=(
                    "completed command or supplied result artifacts are still not generated-world rank fidelity "
                    "without accepted owner evidence"
                ),
            ),
            "rollout_vision_labeling": _section(
                "ready"
                if env["BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING"]["ready"]
                and commands["vision_labeling"]["ready"]
                else "blocked",
                [
                    blocker
                    for blocker, ok in (
                        (
                            "missing_env_BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING",
                            env["BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING"]["ready"],
                        ),
                        ("missing_vision_labeling_command", commands["vision_labeling"]["configured"]),
                        (
                            "vision_labeling_command_executable_not_found",
                            commands["vision_labeling"]["ready"],
                        ),
                    )
                    if not ok
                ],
                command=commands["vision_labeling"],
                gemini_env_present=bool(
                    secrets["GEMINI_API_KEY"]["present"]
                    or secrets["GOOGLE_GENAI_API_KEY"]["present"]
                ),
                proof_boundary="model labels remain review-required until accepted",
            ),
            "delivery_upload": _section(
                "ready"
                if env["BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"]["ready"]
                and commands["delivery_upload"]["ready"]
                else "blocked",
                [
                    blocker
                    for blocker, ok in (
                        (
                            "missing_env_BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD",
                            env["BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"]["ready"],
                        ),
                        ("missing_delivery_command", commands["delivery_upload"]["configured"]),
                        (
                            "delivery_command_executable_not_found",
                            commands["delivery_upload"]["ready"],
                        ),
                    )
                    if not ok
                ],
                command=commands["delivery_upload"],
            ),
            "live_agents_operator": _section(
                "ready"
                if env[LIVE_AGENTS_SDK_ENV]["ready"]
                and secrets["OPENAI_API_KEY"]["present"]
                and modules["openai_agents_sdk"]
                else "blocked",
                [
                    blocker
                    for blocker, ok in (
                        (f"missing_env_{LIVE_AGENTS_SDK_ENV}", env[LIVE_AGENTS_SDK_ENV]["ready"]),
                        ("missing_openai_api_key", secrets["OPENAI_API_KEY"]["present"]),
                        ("missing_openai_agents_sdk", modules["openai_agents_sdk"]),
                    )
                    if not ok
                ],
                auth_boundary=OPENAI_AUTH_FACTS,
            ),
            "live_codex_operator": _section(
                "ready"
                if env[LIVE_CODEX_SDK_ENV]["ready"]
                and (
                    modules["openai_codex_sdk"]
                    or (
                        commands["codex_cli"]["ready"]
                        and env[CODEX_CLI_HOST_OAUTH_ENV]["ready"]
                    )
                )
                else "blocked",
                [
                    blocker
                    for blocker, ok in (
                        (f"missing_env_{LIVE_CODEX_SDK_ENV}", env[LIVE_CODEX_SDK_ENV]["ready"]),
                        (
                            "missing_openai_codex_sdk_or_codex_cli_host_oauth",
                            modules["openai_codex_sdk"]
                            or (
                                commands["codex_cli"]["ready"]
                                and env[CODEX_CLI_HOST_OAUTH_ENV]["ready"]
                            ),
                        ),
                        (
                            f"missing_env_{CODEX_CLI_HOST_OAUTH_ENV}",
                            modules["openai_codex_sdk"]
                            or env[CODEX_CLI_HOST_OAUTH_ENV]["ready"],
                        ),
                        (
                            "missing_codex_cli",
                            modules["openai_codex_sdk"] or commands["codex_cli"]["ready"],
                        ),
                    )
                    if not ok
                ],
                auth_boundary=OPENAI_AUTH_FACTS,
                codex_cli_ready=commands["codex_cli"]["ready"],
                codex_cli_host_oauth_allowed=env[CODEX_CLI_HOST_OAUTH_ENV]["ready"],
            ),
            "webapp_upstream_truth": _capture_upstream_truth(capture_path),
            "digitalocean_control_plane": _digitalocean_read(
                allow_read=allow_digitalocean_read,
                token_env=digitalocean_token_env,
                droplet_name=digitalocean_droplet_name,
                droplet_ip=digitalocean_droplet_ip,
                timeout_seconds=timeout_seconds,
            ),
        }
        all_blockers: List[str] = []
        for name, section in sections.items():
            for blocker in section.get("blockers") or []:
                all_blockers.append(f"{name}:{blocker}")
        manifest = {
            "schema_version": LIVE_PIPELINE_SETUP_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": _overall_status(sections),
            "capture_root": str(capture_path) if capture_path else None,
            "package_dir": str(package_path) if package_path else None,
            "env_files": env_summary,
            "env_gates": env,
            "secrets": secrets,
            "modules": modules,
            "commands": commands,
            "sections": sections,
            "blockers": all_blockers,
            "next_inputs_needed": _next_inputs_needed(sections),
        }
        if output_path:
            path = Path(output_path)
        elif capture_path:
            path = capture_path / "pipeline" / "live_pipeline_setup" / "live_pipeline_setup_manifest.json"
        else:
            path = Path.cwd() / "live_pipeline_setup_manifest.json"
        ensure_dir(path.parent)
        write_json(path, manifest)
        return manifest
    finally:
        _restore_env(original_env)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit live Arena/package pipeline setup without printing secret values"
    )
    parser.add_argument("--capture-root")
    parser.add_argument("--package-dir")
    parser.add_argument("--arena-results-dir")
    parser.add_argument("--simulator-command")
    parser.add_argument("--vision-labeling-command")
    parser.add_argument("--delivery-command")
    parser.add_argument("--no-load-env-files", action="store_true")
    parser.add_argument("--allow-digitalocean-read", action="store_true")
    parser.add_argument("--digitalocean-token-env", default="DIGITALOCEAN_ACCESS_TOKEN")
    parser.add_argument("--digitalocean-droplet-name")
    parser.add_argument("--digitalocean-droplet-ip")
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    result = build_live_pipeline_setup_manifest(
        capture_root=args.capture_root,
        package_dir=args.package_dir,
        arena_results_dir=args.arena_results_dir,
        simulator_command=args.simulator_command,
        vision_labeling_command=args.vision_labeling_command,
        delivery_command=args.delivery_command,
        load_local_env=not args.no_load_env_files,
        allow_digitalocean_read=args.allow_digitalocean_read,
        digitalocean_token_env=args.digitalocean_token_env,
        digitalocean_droplet_name=args.digitalocean_droplet_name,
        digitalocean_droplet_ip=args.digitalocean_droplet_ip,
        timeout_seconds=args.timeout_seconds,
        output_path=args.output_path,
    )
    path = args.output_path
    if not path:
        path = (
            str(Path(args.capture_root).resolve() / "pipeline" / "live_pipeline_setup" / "live_pipeline_setup_manifest.json")
            if args.capture_root
            else str(Path.cwd() / "live_pipeline_setup_manifest.json")
        )
    print(f"[live-pipeline-setup] manifest={path}")
    print(f"[live-pipeline-setup] status={result['status']}")
    if result["blockers"]:
        print(f"[live-pipeline-setup] blockers={len(result['blockers'])}")
    return 0 if result["status"] != "blocked" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
