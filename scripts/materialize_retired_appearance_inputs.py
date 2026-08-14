#!/usr/bin/env python3
"""Materialize retained offline inputs for retired appearance methods.

This command exists only to replay historical evidence. It publishes no live
profile, performs no provider mutation, and must not be used to relaunch Aura.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.public_scene_broad_repair_aura_packet import (
    materialize_broad_repair_aura_packet,
)


STEPS: dict[str, Step] = {
    "aura-broad-repair-packet": Step(
        "Replay the historical offline Aura broad-repair input packet.",
        materialize_broad_repair_aura_packet,
        {
            "broad_support_packet_path": Param(
                "--broad-support-packet", required=True
            ),
            "backend_admission_path": Param("--backend-admission", required=True),
            "output_root": Param("--output-root", required=True),
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
