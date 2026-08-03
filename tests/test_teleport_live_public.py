"""Opt-in Teleport public-data lifecycle test.

This test is intentionally unreachable from normal CI.  It requires the exact
immutable packet/request paths, both explicit public-upload interlocks, a clean
published source commit, file-backed credentials, and the canonical allocator.
It must never be pointed at customer or confidential capture data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from blueprint_pipeline.paid_resource_allocator import main


@pytest.mark.slow
@pytest.mark.external_runtime
def test_opt_in_teleport_public_dataset_lifecycle() -> None:
    if os.environ.get("TELEPORT_PUBLIC_DATA_UPLOAD_AUTHORIZED", "").lower() != "true":
        pytest.skip("explicit Teleport public-data upload authorization not present")
    required = {
        "TELEPORT_PUBLIC_DATA_SPEND_CAP_USD": os.environ.get(
            "TELEPORT_PUBLIC_DATA_SPEND_CAP_USD"
        ),
        "TELEPORT_LIVE_UPLOAD_PACKET": os.environ.get("TELEPORT_LIVE_UPLOAD_PACKET"),
        "TELEPORT_LIVE_EXECUTION_REQUEST": os.environ.get(
            "TELEPORT_LIVE_EXECUTION_REQUEST"
        ),
        "TELEPORT_LIVE_CANDIDATE_OBSERVATIONS": os.environ.get(
            "TELEPORT_LIVE_CANDIDATE_OBSERVATIONS"
        ),
        "TELEPORT_LIVE_SEALED_EVALUATION_REQUEST": os.environ.get(
            "TELEPORT_LIVE_SEALED_EVALUATION_REQUEST"
        ),
        "TELEPORT_LIVE_OUTPUT_DIR": os.environ.get("TELEPORT_LIVE_OUTPUT_DIR"),
    }
    missing = sorted(key for key, value in required.items() if not str(value or "").strip())
    if missing:
        pytest.skip("missing opt-in Teleport live inputs: " + ",".join(missing))
    paths = {
        key: Path(str(value)).expanduser().resolve()
        for key, value in required.items()
        if key != "TELEPORT_PUBLIC_DATA_SPEND_CAP_USD"
    }
    assert all(
        path.is_file()
        for key, path in paths.items()
        if key != "TELEPORT_LIVE_OUTPUT_DIR"
    )
    exit_code = main(
        [
            "provider-reconstruction",
            "--provider",
            "teleport",
            "--upload-packet",
            str(paths["TELEPORT_LIVE_UPLOAD_PACKET"]),
            "--execution-request",
            str(paths["TELEPORT_LIVE_EXECUTION_REQUEST"]),
            "--candidate-observations",
            str(paths["TELEPORT_LIVE_CANDIDATE_OBSERVATIONS"]),
            "--sealed-evaluation-request",
            str(paths["TELEPORT_LIVE_SEALED_EVALUATION_REQUEST"]),
            "--output-dir",
            str(paths["TELEPORT_LIVE_OUTPUT_DIR"]),
            "--execute",
        ]
    )
    assert exit_code == 0
