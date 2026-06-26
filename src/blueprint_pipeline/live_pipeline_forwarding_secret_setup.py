"""Create a local env file for WebApp-to-Pipeline live forwarding.

The generated env file contains the shared bearer token for both sides of the
handoff:

- ``ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`` for WebApp forwarding/probes.
- ``BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN`` for the Pipeline intake service.

The token is intentionally written only to the env file. Summaries and optional
manifests include paths and key names, not the raw token or a token hash.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, utc_now_iso, write_json

DEFAULT_ENV_FILE = Path.home() / ".blueprint-secrets" / "live_pipeline_forwarding.env"
DEFAULT_FORWARD_URL = "https://paperclip.tryblueprint.io/api/live-pipeline/job-requests"
TOKEN_KEYS = (
    "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN",
    "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN",
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _validate_forward_url(value: str) -> str:
    url = _string(value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("forward_url must be an absolute http(s) URL")
    if not parsed.path.endswith("/api/live-pipeline/job-requests"):
        raise ValueError("forward_url must end with /api/live-pipeline/job-requests")
    return url


def _parse_env_value(raw_value: str) -> str:
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    parts = list(lexer)
    return parts[0] if parts else ""


def parse_env_file_values(path: Path) -> dict[str, str]:
    """Parse simple shell-style KEY=value / export KEY=value files."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        values[key] = _parse_env_value(raw_value.strip())
    return values


def _export_line(key: str, value: str) -> str:
    return f"export {key}={shlex.quote(value)}"


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    """Normalize relative capture roots without rewriting remote absolute paths."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _staged_inputs_path(capture_root: Path | None) -> str | None:
    if capture_root is None:
        return None
    return str(capture_root / "pipeline" / "live_pipeline_staged_inputs.json")


def create_live_pipeline_forwarding_env(
    *,
    env_file: Path | None = None,
    forward_url: str = DEFAULT_FORWARD_URL,
    token: str | None = None,
    force: bool = False,
    capture_root: Path | None = None,
    site_slug: str | None = None,
    required: bool = True,
    timeout_ms: int | None = None,
    write_manifest: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write a reusable local env file for live forwarding and intake probes."""

    generated = generated_at or utc_now_iso()
    env_path = Path(env_file or DEFAULT_ENV_FILE).expanduser().resolve()
    forward_url = _validate_forward_url(forward_url)
    existing_values = parse_env_file_values(env_path)
    existing_token = next(
        (_string(existing_values.get(key)) for key in TOKEN_KEYS if _string(existing_values.get(key))),
        "",
    )
    selected_token = _string(token) or ("" if force else existing_token) or secrets.token_urlsafe(32)
    status = "created"
    if existing_token and not force and not token:
        status = "already_present"
    elif existing_token and (force or token):
        status = "rotated"

    capture_root_path = _absolute_path_without_symlink_resolution(capture_root) if capture_root else None
    env_values = {
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_URL": forward_url,
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN": selected_token,
        "ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED": "true" if required else "false",
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_URL": forward_url,
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN": selected_token,
    }
    if timeout_ms is not None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        env_values["ROBOT_EVAL_JOB_REQUEST_FORWARD_TIMEOUT_MS"] = str(timeout_ms)
    if capture_root_path is not None and _string(site_slug):
        env_values["ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON"] = json.dumps(
            {_string(site_slug): str(capture_root_path)},
            separators=(",", ":"),
        )
    elif capture_root_path is not None:
        env_values["ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT"] = str(capture_root_path)
    staged_inputs = _staged_inputs_path(capture_root_path)
    if staged_inputs:
        env_values["BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"] = staged_inputs

    ensure_dir(env_path.parent)
    lines = [
        "# Blueprint live WebApp-to-Pipeline forwarding env.",
        "# This file contains a bearer token. Keep it local and untracked.",
        *[_export_line(key, env_values[key]) for key in sorted(env_values)],
        "",
    ]
    env_path.write_text("\n".join(lines), encoding="utf-8")
    env_path.chmod(0o600)
    stat = env_path.stat()

    summary: dict[str, Any] = {
        "schema_version": "blueprint.live_pipeline_forwarding_secret_setup.v1",
        "generated_at": generated,
        "status": status,
        "env_file": str(env_path),
        "forward_url": forward_url,
        "file_mode_octal": oct(stat.st_mode & 0o777),
        "file_size_bytes": stat.st_size,
        "configured_keys": sorted(env_values),
        "token_env_keys": list(TOKEN_KEYS),
        "raw_token_written_to_stdout": False,
        "raw_token_written_to_manifest": False,
        "raw_token_hash_written_to_manifest": False,
        "raw_token_written_to_env_file": True,
        "webapp_preflight_args": ["--forwarding-env-file", str(env_path)],
        "shell_source_command": f"set -a; source {shlex.quote(str(env_path))}; set +a",
    }
    if capture_root_path is not None:
        summary["capture_root"] = str(capture_root_path)
    if _string(site_slug):
        summary["site_slug"] = _string(site_slug)
    if write_manifest is not None:
        write_json(Path(write_manifest), summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--forward-url", default=DEFAULT_FORWARD_URL)
    parser.add_argument("--token", help="Optional existing token to write. Omit to generate/reuse.")
    parser.add_argument("--force", action="store_true", help="Rotate an existing generated token.")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--site-slug")
    parser.add_argument("--optional", action="store_true", help="Set forwarding required to false.")
    parser.add_argument("--timeout-ms", type=int)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args(argv)

    summary = create_live_pipeline_forwarding_env(
        env_file=args.env_file,
        forward_url=args.forward_url,
        token=args.token,
        force=args.force,
        capture_root=args.capture_root,
        site_slug=args.site_slug,
        required=not args.optional,
        timeout_ms=args.timeout_ms,
        write_manifest=args.write_manifest,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
