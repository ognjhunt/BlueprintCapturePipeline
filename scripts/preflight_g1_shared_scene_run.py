#!/usr/bin/env python3
"""Offline G1 scene and model identity check for local or container staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blueprint_pipeline.native_g1_run_preflight import preflight_g1_shared_scene_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-plan", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--policy-server-source", type=Path, required=True)
    parser.add_argument("--sonic-provider-source", type=Path, required=True)
    parser.add_argument("--sonic-encoder", type=Path, required=True)
    parser.add_argument("--sonic-encoder-sha256", required=True)
    parser.add_argument("--sonic-decoder", type=Path, required=True)
    parser.add_argument("--sonic-decoder-sha256", required=True)
    args = parser.parse_args()
    receipt = preflight_g1_shared_scene_run(
        scene_plan_path=args.scene_plan,
        bundle_root=args.bundle_root,
        inventory_path=args.inventory,
        candidate_id=args.candidate,
        checkpoint_root=args.checkpoint_root,
        policy_server_source=args.policy_server_source,
        sonic_provider_source=args.sonic_provider_source,
        sonic_encoder=args.sonic_encoder,
        sonic_encoder_sha256=args.sonic_encoder_sha256,
        sonic_decoder=args.sonic_decoder,
        sonic_decoder_sha256=args.sonic_decoder_sha256,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
