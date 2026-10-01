"""Kernel mutation fencing cannot disable current foreign-process observation."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_sandbox.py
import os

import pytest


def test_ordinary_process_cannot_claim_historical_kernel_confinement(tmp_path):
    from blueprint_pipeline.control_plane_lane_historical_sandbox import HistoricalNativeSandbox
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match='native_unavailable'):
            HistoricalNativeSandbox(descriptor, descriptor, {}, tick=lambda: None)
    finally:
        os.close(descriptor)
