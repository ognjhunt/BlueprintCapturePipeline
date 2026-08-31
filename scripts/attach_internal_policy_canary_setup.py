#!/usr/bin/env python3
"""Attach one verified internal canary setup before immutable profile publication."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_dispatcher import validate_launch_profile
from blueprint_pipeline.task_evaluation_policy_canary_setup import (
    validate_policy_canary_setup,
)


def attach_internal_policy_canary_setup(
    *,
    profile: Mapping[str, Any],
    setup: Mapping[str, Any],
    profile_validator: Callable[[Mapping[str, Any]], list[str]] = validate_launch_profile,
) -> dict[str, Any]:
    """Return a new digest-bound profile; never mutate published bytes in place."""

    source = deepcopy(dict(profile))
    blockers = profile_validator(source)
    if blockers:
        raise ValueError("policy_canary_source_profile_invalid:" + ",".join(blockers))
    canary = validate_policy_canary_setup(setup)
    configured_source_launch_id = source.get(
        "configured_source_launch_id", source.get("profile_id")
    )
    if canary["source_launch_id"] != configured_source_launch_id:
        raise ValueError("policy_canary_setup_profile_binding_mismatch")
    if "internal_policy_canary_setup" in source:
        if source["internal_policy_canary_setup"] != canary:
            raise ValueError("policy_canary_setup_immutable_conflict")
        return source
    output = deepcopy(source)
    output["internal_policy_canary_setup"] = canary
    output["profile_digest"] = canonical_digest(output, digest_field="profile_digest")
    blockers = profile_validator(output)
    if blockers:
        raise ValueError("policy_canary_profile_invalid:" + ",".join(blockers))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--canary-setup", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    setup = json.loads(Path(args.canary_setup).read_text(encoding="utf-8"))
    output = attach_internal_policy_canary_setup(profile=profile, setup=setup)
    destination = Path(args.output).expanduser().resolve()
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8", errors="strict")
    except OSError as exc:
        raise ValueError("policy_canary_profile_output_invalid") from exc
    print(json.dumps({"profile_id": output["profile_id"], "profile_digest": output["profile_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
