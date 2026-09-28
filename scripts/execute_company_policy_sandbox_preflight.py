#!/usr/bin/env python3
"""Execute one synthetic, no-scene company-policy sandbox preflight.

This command mutates only the selected dedicated worker's local container
runtime. It never allocates a provider, queues a launch, publishes a profile,
or sends a real observation. Customer credentials and image bytes are removed
before it returns.
"""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.company_policy_sandbox_executor import (
    HttpCredentialBroker,
    SubprocessCommandRunner,
    execute_company_policy_sandbox_preobservation,
)


def _mapping(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {resolved}")
    return value


def _private_file(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    resolved = Path(path).expanduser().resolve()
    if stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise ValueError(f"{label}_mode_must_be_0600")
    value = resolved.read_bytes().strip()
    if len(value) < 32:
        raise ValueError(f"{label}_too_short")
    return resolved, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--broker-base-url", required=True)
    parser.add_argument("--broker-token-file", required=True)
    parser.add_argument("--broker-client-id", default="blueprint-policy-sandbox-worker")
    parser.add_argument("--attestation-key-file", required=True)
    parser.add_argument("--attestation-key-id", required=True)
    parser.add_argument("--worker-boot-receipt", required=True)
    parser.add_argument("--apparmor-profiles-path", default="/sys/kernel/security/apparmor/profiles")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ack",
        required=True,
        choices=["synthetic-no-scene-local-runtime-mutation"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = _mapping(args.plan)
    contract = _mapping(args.contract)
    worker_boot_receipt = _mapping(args.worker_boot_receipt)
    token_path, _token = _private_file(
        args.broker_token_file, label="company_policy_broker_token"
    )
    _key_path, attestation_key = _private_file(
        args.attestation_key_file, label="company_policy_attestation_key"
    )
    visibility = str((contract.get("container") or {}).get("visibility") or "")
    broker = (
        HttpCredentialBroker(
            base_url=args.broker_base_url,
            token_file=token_path,
            client_id=args.broker_client_id,
        )
        if visibility == "private"
        else None
    )
    result = execute_company_policy_sandbox_preobservation(
        plan=plan,
        contract=contract,
        broker=broker,
        runner=SubprocessCommandRunner(),
        attestation_key=attestation_key,
        attestation_key_id=args.attestation_key_id,
        worker_boot_receipt=worker_boot_receipt,
        output_path=Path(args.output),
        apparmor_profiles_path=Path(args.apparmor_profiles_path),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).expanduser().resolve()),
                "executor_receipt_digest": (
                    (result.get("executor_receipt") or {}).get("receipt_digest")
                ),
                "terminal_receipt_digest": result["terminal_receipt"]["receipt_digest"],
                "real_observation_sent": False,
                "launch_authority_granted": False,
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "qualified_dry_run_no_real_observation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
