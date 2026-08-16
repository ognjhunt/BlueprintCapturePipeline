#!/usr/bin/env python3
"""Materialize one no-mutation admission for a bounded Newton controls canary.

The scientific builder and allocator already validate
``adp009d_newton_canary_admission.v1``.  This CLI is the missing production
bridge: it reopens the current spend lock and authenticated provider inventory,
binds them to the explicit human authority, and writes one immutable receipt.
It never calls a provider and cannot authorize a policy query.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.adp009d_physics_backend_comparison import (
    PhysicsBackendContractError,
    build_backend_profile,
    build_newton_canary_admission,
    validate_newton_canary_admission,
)

try:  # pytest imports scripts as a package; direct execution puts this dir on the path
    from scripts.build_adp009d_840313_launch_profile import (
        ProductionProfileBuildError,
        _read,
        _write_exact,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI invocation
    from build_adp009d_840313_launch_profile import (
        ProductionProfileBuildError,
        _read,
        _write_exact,
    )


MAX_SPEND_USD = 2.0


def materialize_newton_canary_admission(
    *,
    authorization_evidence_ref: str,
    spend_admission_lock_path: str | Path,
    provider_guard_path: str | Path,
    max_spend_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
    allowed_active_vast_instance_ids: Sequence[int] = (),
    issued_at: datetime | None = None,
) -> dict[str, Any]:
    """Write a validator-clean receipt without allocating or contacting a provider."""

    if (
        isinstance(max_spend_usd, bool)
        or not isinstance(max_spend_usd, (int, float))
        or not 0 < float(max_spend_usd) <= MAX_SPEND_USD
    ):
        raise ProductionProfileBuildError("newton_canary_admission_spend_cap_invalid")
    if (
        isinstance(hard_ttl_seconds, bool)
        or not isinstance(hard_ttl_seconds, int)
        or not 1800 <= hard_ttl_seconds <= 14_400
    ):
        raise ProductionProfileBuildError("newton_canary_admission_ttl_invalid")
    spend_lock = _read(Path(spend_admission_lock_path).expanduser())
    provider_guard = _read(Path(provider_guard_path).expanduser())
    admission = build_newton_canary_admission(
        authorization_evidence_ref=authorization_evidence_ref,
        spend_admission_lock=spend_lock,
        provider_zero_precheck=provider_guard,
        max_spend_usd=float(max_spend_usd),
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_vast_instance_ids=allowed_active_vast_instance_ids,
        issued_at=issued_at,
    )
    blockers = validate_newton_canary_admission(
        admission,
        profile=build_backend_profile("newton"),
        now=issued_at,
    )
    if blockers:
        raise ProductionProfileBuildError("newton_canary_admission_invalid:" + ",".join(blockers))
    unresolved_output = Path(output_path).expanduser()
    if unresolved_output.is_symlink():
        raise ProductionProfileBuildError("newton_canary_admission_output_symlink")
    output = unresolved_output.resolve()
    _write_exact(output, admission)
    return admission


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-evidence-ref", required=True)
    parser.add_argument("--spend-admission-lock", required=True)
    parser.add_argument("--provider-guard", required=True)
    parser.add_argument("--max-spend-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument(
        "--allowed-active-vast-instance-id",
        action="append",
        default=[],
        type=int,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_newton_canary_admission(
            authorization_evidence_ref=args.authorization_evidence_ref,
            spend_admission_lock_path=args.spend_admission_lock,
            provider_guard_path=args.provider_guard,
            max_spend_usd=args.max_spend_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            output_path=args.output,
            allowed_active_vast_instance_ids=args.allowed_active_vast_instance_id,
        )
    except (
        OSError,
        json.JSONDecodeError,
        PhysicsBackendContractError,
        ProductionProfileBuildError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "adp009d_newton_canary_admission_build.v1",
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
