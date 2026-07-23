#!/usr/bin/env python3
"""Convert a validated G4 boot self-test into campaign preload evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-self-test", type=Path, required=True)
    parser.add_argument("--runtime-health", type=Path, required=True)
    parser.add_argument("--allocation-key", required=True)
    parser.add_argument("--host-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.allocation_key.strip() or not args.host_image_id.strip():
        raise ValueError("allocation key and host image id must be nonempty")
    host, host_sha = _load(args.host_self_test)
    health, health_sha = _load(args.runtime_health)
    if host.get("schema_version") != "blueprint_g4_host_self_test.v1":
        raise ValueError("host self-test schema is invalid")
    if host.get("status") != "passed":
        raise ValueError("host self-test did not pass")
    if health.get("schema_version") != "groot_oscar_closed_loop_image_healthcheck.v1":
        raise ValueError("runtime health schema is invalid")
    if health.get("status") != "passed" or health.get("blockers") not in (None, []):
        raise ValueError("sealed runtime health preflight did not pass")
    if health.get("build_time") is not False:
        raise ValueError("runtime health must come from a live runtime probe")
    if health.get("torch_cuda_available") is not True:
        raise ValueError("runtime health did not prove live CUDA availability")
    if health.get("oscar_dynamic_config_probe_mode") != "live_cuda_runtime":
        raise ValueError("runtime health did not use the live CUDA OSCAR probe")
    if health.get("oscar_dynamic_config_live_cuda_proven") is not True:
        raise ValueError("runtime health did not prove the live CUDA OSCAR config")

    source_sha = str(host.get("worker_source_sha") or "")
    image_digest = str(host.get("preloaded_worker_image_digest") or "")
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("host self-test source SHA is invalid")
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise ValueError("host self-test image digest is invalid")
    runtime_metadata = health.get("runtime_metadata")
    if (
        not isinstance(runtime_metadata, dict)
        or runtime_metadata.get("source_commit") != source_sha
    ):
        raise ValueError("runtime health source does not match the preloaded worker")
    if health.get("worker_image_digest") != image_digest:
        raise ValueError("runtime health digest does not match the preloaded worker")
    if host.get("image_present_before_allocation") is not True:
        raise ValueError("image was not proven present before allocation")
    if host.get("local_digest_inspect_passed") is not True:
        raise ValueError("local digest inspection did not pass")
    if host.get("cold_pull_required_during_campaign") is not False:
        raise ValueError("campaign would still require a cold pull")

    evidence = {
        "schema_version": "preloaded_worker_image.v1",
        "source_sha": source_sha,
        "image_digest": image_digest,
        "allocation_key": args.allocation_key.strip(),
        "host_image_id": args.host_image_id.strip(),
        "image_present_before_allocation": True,
        "local_digest_inspect_passed": True,
        "runtime_health_preflight_passed": True,
        "cold_pull_required_during_campaign": False,
        "host_self_test_sha256": host_sha,
        "runtime_health_sha256": health_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
