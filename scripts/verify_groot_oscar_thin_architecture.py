#!/usr/bin/env python3
"""Fail closed when the cached-foundation architecture regresses.

This is intentionally a cheap, network-free CI gate.  It catches packaging
drift before an operator allocates the large native Docker builder.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
FOUNDATION = IMAGE_ROOT / "Foundation.Dockerfile"
RELEASE = IMAGE_ROOT / "Release.Dockerfile"
ENTRYPOINT = IMAGE_ROOT / "thin_release_entrypoint.sh"
MODEL_CACHE = ROOT / "src/blueprint_pipeline/groot_oscar_model_cache.py"


def _assigned_literal(module: Path, name: str) -> object:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    assignments[target.id] = node.value

    def evaluate(node: ast.expr) -> object:
        if isinstance(node, ast.Name) and node.id in assignments:
            return evaluate(assignments[node.id])
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.List):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values)
                if key is not None
            }
        return ast.literal_eval(node)

    if name not in assignments:
        raise ValueError(f"missing_literal:{name}")
    return evaluate(assignments[name])


def verify() -> list[str]:
    blockers: list[str] = []
    foundation = FOUNDATION.read_text(encoding="utf-8")
    release = RELEASE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    final_stage = foundation.rsplit("FROM tensorrt-base", 1)[-1]

    required_foundation_fragments = (
        "AS robot-env-builder",
        "AS wbc-builder",
        "ca-certificates sudo",
        "sha256:a1bc93654f31669fd964ea3011a5e5e9676b9b6f8adcd762606e5140632ea72d",
        "sha256:b072f989d6315ac0e22dcb4771b083c5156d974a3496ac3504c77f4062eb248e",
        "test ! -d third_party/cppzmq/.git",
        "uv venv /opt/robot-venv --python 3.10 --seed",
        "install -m 0755 target/release/g1_deploy_onnx_ref",
        "cp -a /opt/onnxruntime/lib/libonnxruntime.so*",
        "test ! -d /opt/wbc/gear_sonic_deploy/build",
        "test ! -d /opt/onnxruntime/include",
        "! ldd /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
    )
    for fragment in required_foundation_fragments:
        if fragment not in foundation:
            blockers.append(f"foundation_runtime_contract_missing:{fragment}")
    forbidden_foundation_fragments = (
        "cp -a build target g1 scripts reference",
        "cp -a /tmp/wbc/gear_sonic /opt/wbc-runtime/gear_sonic",
        "COPY --from=wbc-builder /opt/onnxruntime /opt/onnxruntime",
        "snapshot_download",
        "/opt/blueprint/ckpts/",
    )
    for fragment in forbidden_foundation_fragments:
        if fragment in foundation:
            blockers.append(f"foundation_forbidden_payload_present:{fragment}")
    if foundation.count("uv venv /opt/robot-venv") != 1:
        blockers.append("foundation_robot_environment_not_consolidated")
    for build_package in (
        "build-essential",
        "clang",
        "cmake",
        "git-lfs",
        "ninja-build",
        "pkg-config",
        " sudo ",
    ):
        install_region = final_stage.split("installed_build_packages=", 1)[0]
        if build_package in install_region:
            blockers.append(f"foundation_final_stage_build_dependency:{build_package.strip()}")

    required_release_fragments = (
        "ARG FOUNDATION_IMAGE",
        "FROM ${FOUNDATION_IMAGE}",
        "BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-thin-release",
        "test ! -e /opt/blueprint/ckpts",
    )
    for fragment in required_release_fragments:
        if fragment not in release:
            blockers.append(f"thin_release_contract_missing:{fragment}")
    for fragment in ("snapshot_download", "hf_hub_download", "model.safetensors"):
        if fragment in release:
            blockers.append(f"thin_release_embeds_model_acquisition:{fragment}")
    if "groot_oscar_model_cache activate" not in entrypoint:
        blockers.append("thin_release_offline_model_activation_missing")

    try:
        schema = _assigned_literal(MODEL_CACHE, "SCHEMA_VERSION")
        verification_schema = _assigned_literal(
            MODEL_CACHE, "VERIFICATION_SCHEMA_VERSION"
        )
        pins = _assigned_literal(MODEL_CACHE, "MODEL_PINS")
        required = _assigned_literal(MODEL_CACHE, "REQUIRED_MODEL_FILES")
    except (SyntaxError, ValueError) as exc:
        blockers.append(f"model_cache_contract_unreadable:{exc}")
    else:
        if schema != "groot_oscar_external_model_cache.v2":
            blockers.append("model_cache_manifest_schema_not_strong_v2")
        if verification_schema != "groot_oscar_external_model_cache_verification.v2":
            blockers.append("model_cache_verification_schema_not_strong_v2")
        pin_rows = cast(tuple[object, ...], pins) if isinstance(pins, tuple) else ()
        required_map = (
            cast(dict[object, object], required) if isinstance(required, dict) else {}
        )
        pin_names = {
            row[0]
            for row in pin_rows
            if isinstance(row, tuple) and len(row) == 3 and isinstance(row[0], str)
        }
        required_names = {key for key in required_map if isinstance(key, str)}
        if pin_names != required_names:
            blockers.append("model_cache_required_files_do_not_cover_all_pins")
        if any(
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[2], str)
            or len(row[2]) != 40
            for row in pin_rows
        ):
            blockers.append("model_cache_contains_mutable_repository_pin")
        raw_gear_files = required_map.get("gear_sonic", ())
        gear_files = set(raw_gear_files) if isinstance(raw_gear_files, tuple) else set()
        if gear_files != {
            "model_encoder.onnx",
            "model_decoder.onnx",
            "observation_config.yaml",
            "planner_sonic.onnx",
        }:
            blockers.append("model_cache_gear_sonic_runtime_allowlist_incomplete")
    return sorted(set(blockers))


def main() -> int:
    blockers = verify()
    if blockers:
        for blocker in blockers:
            print(blocker, file=sys.stderr)
        return 2
    print("groot_oscar_thin_architecture_verification=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
