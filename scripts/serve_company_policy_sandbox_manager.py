#!/usr/bin/env python3
"""Run the trusted policy sandbox manager on a dedicated Blueprint worker."""
from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from blueprint_pipeline.company_policy_sandbox_manager import serve_manager


def _private_file(path: Path) -> Path:
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_uid != os.geteuid()
            or stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise ValueError("policy_sandbox_manager_private_file_invalid")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ack", required=True, choices=["trusted-policy-sandbox-manager"])
    args = parser.parse_args()
    settings = json.loads(_private_file(args.config).read_text())
    if not isinstance(settings, dict):
        raise ValueError("policy_sandbox_manager_config_invalid")
    token = _private_file(Path(str(settings["management_token_file"]))).read_text().strip()
    if not 32 <= len(token) <= 512 or any(char.isspace() for char in token):
        raise ValueError("policy_sandbox_manager_token_invalid")
    certificate = Path(str(settings["management_tls_certificate"]))
    private_key = _private_file(Path(str(settings["management_tls_private_key"])))
    if not certificate.is_absolute() or certificate.is_symlink() or not certificate.is_file():
        raise ValueError("policy_sandbox_manager_tls_certificate_invalid")
    serve_manager(settings=settings, token=token, certificate=certificate,
        private_key=private_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
