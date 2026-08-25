#!/usr/bin/env python3
"""Build an immutable no-spend company-policy sandbox plan.

This command validates inputs and writes command data only.  It does not
redeem credentials, pull images, start containers, create profiles, queue
launches, or contact a provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.common import write_json
from blueprint_pipeline.company_policy_sandbox_v2 import (
    SUPPORTED_RUNTIME_CLASSES,
    build_company_policy_sandbox_plan,
)


def _mapping(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {resolved}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-receipt", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--sandbox-attempt-id", required=True)
    parser.add_argument("--pipeline-release-sha", required=True)
    parser.add_argument("--worker-identity", required=True)
    parser.add_argument("--runtime-class", required=True, choices=sorted(SUPPORTED_RUNTIME_CLASSES))
    parser.add_argument("--proxy-image", required=True)
    parser.add_argument("--proxy-contract-digest", required=True)
    parser.add_argument("--seccomp-profile-id", required=True)
    parser.add_argument("--seccomp-profile-digest", required=True)
    parser.add_argument("--apparmor-profile-id", required=True)
    parser.add_argument("--apparmor-profile-digest", required=True)
    parser.add_argument("--registry-address", action="append", default=[], required=True)
    parser.add_argument("--allowed-registry-host", action="append", default=[], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ack", required=True, choices=["no-spend-plan-only"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    plan = build_company_policy_sandbox_plan(
        admission_receipt=_mapping(args.admission_receipt),
        contract=_mapping(args.contract),
        sandbox_attempt_id=args.sandbox_attempt_id,
        pipeline_release_sha=args.pipeline_release_sha,
        worker_identity=args.worker_identity,
        runtime_class=args.runtime_class,
        blueprint_proxy_image=args.proxy_image,
        blueprint_proxy_contract_digest=args.proxy_contract_digest,
        seccomp_profile_id=args.seccomp_profile_id,
        seccomp_profile_digest=args.seccomp_profile_digest,
        apparmor_profile_id=args.apparmor_profile_id,
        apparmor_profile_digest=args.apparmor_profile_digest,
        registry_addresses=args.registry_address,
        allowed_registry_hosts=args.allowed_registry_host,
    )
    write_json(output, plan)
    readback = json.loads(output.read_text(encoding="utf-8"))
    if readback != plan:
        raise RuntimeError("company_policy_sandbox_plan_atomic_readback_mismatch")
    print(
        json.dumps(
            {
                "status": "planned_no_spend",
                "output": str(output),
                "plan_digest": plan["plan_digest"],
                "launch_authority_granted": False,
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
