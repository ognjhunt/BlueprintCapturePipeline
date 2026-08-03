#!/usr/bin/env python3
"""Build the splatfacto (nerfstudio/gsplat) bakeoff arm packet — C5.

Adds the Linux open-trainer arms G1 (splatfacto, gsplat DefaultStrategy) and
G2 (splatfacto-MCMC, gsplat MCMCStrategy) to the MuSHRoom bakeoff under the
2026-08-02 v2 scorecard. The arms bind byte-identically to the same
point-seeded COLMAP input dataset as the Postshot P1/P2 arms by copying the
verified P1 ``input_dataset`` block from the frozen Postshot execution
packet — no re-derivation, no new digests, no provider pose estimation.

Discipline mirrors the existing packet builders: every input's self-digest
is verified first, hidden evaluator paths must not appear anywhere in the
emitted packet, the packet self-digests, and an existing packet with
different content is never overwritten. No paid runs are implied; the arms
ride the already-gated bakeoff budget on the ordinary Linux Vast lane.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402

SCHEMA_VERSION = "splatfacto_execution_packet.v1"
POSTSHOT_PACKET_RELATIVE = "provider_packets/postshot/postshot_execution_packet.v1.json"
OUTPUT_RELATIVE = "provider_packets/splatfacto/splatfacto_execution_packet.v1.json"

NERFSTUDIO_G1_PIN = "nerfstudio==1.1.5"
NERFSTUDIO_G2_PIN = (
    "nerfstudio @ git+https://github.com/nerfstudio-project/nerfstudio.git"
    "@50e0e3c70c775e89333256213363badbf074f29d"
)
GSPLAT_PIN = "gsplat==1.4.0"


def _fail(message: str) -> None:
    raise SystemExit(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"packet_input_unreadable:{path.name}:{type(exc).__name__}")
    if not isinstance(loaded, dict):
        _fail(f"packet_input_not_object:{path.name}")
    return loaded


def _verify_self_digest(payload: Mapping[str, Any], *, digest_field: str) -> None:
    recorded = str(payload.get(digest_field) or "")
    expected = canonical_digest(payload, digest_field=digest_field)
    if recorded != expected:
        _fail(f"{digest_field}_mismatch")


def _packet_digest_stable_view(packet: Mapping[str, Any]) -> str:
    view = {
        key: value
        for key, value in packet.items()
        if key not in {"timestamp", "splatfacto_execution_packet_digest"}
    }
    return canonical_digest(view, digest_field="splatfacto_execution_packet_digest")


def _arm(
    *,
    arm_id: str,
    strategy: str,
    input_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "method": "splatfacto",
        "strategy": strategy,
        "seed": 42,
        "max_iterations": 30000,
        "dataparser": "colmap",
    }
    if strategy == "mcmc":
        profile["mcmc_max_gs_num"] = 1_000_000
        display = "Splatfacto-MCMC (nerfstudio git pin, gsplat MCMCStrategy)"
        expected = (
            "gsplat MCMC relocation/densification capped at 1M Gaussians; the "
            "worker execution receipt records the exact argv, package "
            "versions, and durations"
        )
    else:
        display = "Splatfacto (nerfstudio 1.1.5, gsplat DefaultStrategy)"
        expected = (
            "gsplat default densify/cull strategy; the worker execution "
            "receipt records the exact argv, package versions, and durations"
        )
    return {
        "arm_id": arm_id,
        "display_name": display,
        "input_dataset": dict(input_dataset),
        "training_profile": profile,
        "pose_estimation_by_provider": False,
        "expected_behavior": expected,
    }


def build_splatfacto_execution_packet(
    *,
    proxy_root: str | Path,
    postshot_packet_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    proxy_root = Path(proxy_root)
    postshot_path = (
        Path(postshot_packet_path)
        if postshot_packet_path is not None
        else proxy_root / POSTSHOT_PACKET_RELATIVE
    )
    if not postshot_path.is_file():
        _fail(f"postshot_execution_packet_missing:{postshot_path}")
    postshot = _load_json(postshot_path)
    if postshot.get("schema_version") != "postshot_execution_packet.v1":
        _fail("postshot_execution_packet_schema_invalid")
    _verify_self_digest(postshot, digest_field="postshot_execution_packet_digest")

    p1 = next(
        (
            arm
            for arm in postshot.get("arms") or []
            if isinstance(arm, Mapping) and arm.get("arm_id") == "P1"
        ),
        None,
    )
    if p1 is None or not isinstance(p1.get("input_dataset"), Mapping):
        _fail("postshot_p1_input_dataset_missing")

    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_awaiting_worker",
        "scorecard": "docs/MUSHROOM_BAKEOFF_DECISION_SCORECARD_2026-08-02_v2.md",
        "source_capture_digest": str(postshot.get("source_capture_digest") or ""),
        "frozen_split_digest": str(postshot.get("frozen_split_digest") or ""),
        "input_dataset_source": {
            "packet": POSTSHOT_PACKET_RELATIVE,
            "packet_digest": str(postshot.get("postshot_execution_packet_digest")),
            "copied_arm_id": "P1",
        },
        "environment": {
            "g1": {
                "nerfstudio": NERFSTUDIO_G1_PIN,
                "gsplat": GSPLAT_PIN,
                "requirements_file": "requirements/splatfacto-arm-g1.txt",
                "venv_setup": "scripts/setup_splatfacto_venv.sh g1",
            },
            "g2": {
                "nerfstudio": NERFSTUDIO_G2_PIN,
                "gsplat": GSPLAT_PIN,
                "requirements_file": "requirements/splatfacto-arm-g2.txt",
                "venv_setup": "scripts/setup_splatfacto_venv.sh g2",
            },
        },
        "arms": [
            _arm(arm_id="G1", strategy="default", input_dataset=p1["input_dataset"]),
            _arm(arm_id="G2", strategy="mcmc", input_dataset=p1["input_dataset"]),
        ],
        "shared_training_intent": {
            "required_outputs": [
                "exported_splat_ply_or_spz",
                "camera_poses_export_if_supported",
                "training_log",
                "execution_receipt_with_versions_and_durations",
            ],
        },
        "worker_requirements": {
            "os": "linux",
            "gpu": "single NVIDIA GPU, >=16GB VRAM",
            "lane": "ordinary Linux Vast lane behind the canonical paid-resource seam",
        },
        "hidden_images_included": False,
        "provider_sees_hidden_views": False,
        "required_external_inputs": [],
        "post_run_obligations": [
            "evaluator_only_heldout_comparison_via_heldout_appearance_evaluation_v2",
            "preserve_unenhanced_outputs_and_training_logs",
            "no_paid_runs_outside_gated_bakeoff_budget",
        ],
        "proof_effect": "bakeoff_candidate_training_only",
        "claim_ceiling": "bakeoff_arm_result_pending_frozen_evaluator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    serialized = json.dumps(packet, sort_keys=True)
    if "evaluator_hidden" in serialized:
        _fail("hidden_path_leak:evaluator_hidden_reference_in_packet")

    packet["splatfacto_execution_packet_digest"] = canonical_digest(
        packet, digest_field="splatfacto_execution_packet_digest"
    )

    destination = (
        Path(output_path) if output_path is not None else proxy_root / OUTPUT_RELATIVE
    )
    if destination.is_file():
        existing = _load_json(destination)
        if _packet_digest_stable_view(existing) != _packet_digest_stable_view(packet):
            _fail(f"refusing to overwrite differing packet manifest:{destination}")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", required=True)
    parser.add_argument("--postshot-packet", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    packet = build_splatfacto_execution_packet(
        proxy_root=args.proxy_root,
        postshot_packet_path=args.postshot_packet,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": packet["status"],
                "arms": [arm["arm_id"] for arm in packet["arms"]],
                "splatfacto_execution_packet_digest": packet[
                    "splatfacto_execution_packet_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
