"""Pytest configuration and shared fixtures for Blueprint Capture Pipeline tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add source directory to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
FUNCTIONS_DIR = Path(__file__).parent.parent / "functions"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTIONS_DIR))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gcs-bucket",
        action="store",
        default=None,
        help="GCS bucket for integration tests (uses mock if not specified)",
    )
    parser.addoption(
        "--project-id",
        action="store",
        default="blueprint-8c1ca",
        help="GCP project ID for integration tests",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU-dependent tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless --run-gpu is specified."""
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def project_id(request):
    """Get GCP project ID from command line or environment."""
    return request.config.getoption("--project-id") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT", "blueprint-8c1ca"
    )


@pytest.fixture(scope="session")
def gcs_bucket(request):
    """Get GCS bucket from command line or environment."""
    return request.config.getoption("--gcs-bucket") or os.environ.get(
        "TEST_GCS_BUCKET"
    )
