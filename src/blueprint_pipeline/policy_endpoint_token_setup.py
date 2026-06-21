"""Create file-based bearer tokens for local policy endpoint smoke tests."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, utc_now_iso, write_json


DEFAULT_TOKEN_PATH = Path.home() / ".blueprint-secrets" / "team_policy_endpoint_token.txt"


def create_team_policy_endpoint_token(
    *,
    token_file: Path | None = None,
    force: bool = False,
    write_manifest: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    path = Path(token_file or DEFAULT_TOKEN_PATH).expanduser().resolve()
    ensure_dir(path.parent)
    existed = path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    if force or not existed:
        path.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    path.chmod(0o600)
    stat = path.stat()
    summary = {
        "schema_version": "team_policy_endpoint_token_setup.v1",
        "generated_at": generated,
        "status": "created" if force or not existed else "already_present",
        "token_file": str(path),
        "file_mode_octal": oct(stat.st_mode & 0o777),
        "file_size_bytes": stat.st_size,
        "raw_token_written_to_stdout": False,
        "raw_token_written_to_artifacts": False,
        "raw_token_hash_written_to_artifacts": False,
    }
    if write_manifest is not None:
        write_json(Path(write_manifest), summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args(argv)
    summary = create_team_policy_endpoint_token(
        token_file=args.token_file,
        force=args.force,
        write_manifest=args.write_manifest,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
