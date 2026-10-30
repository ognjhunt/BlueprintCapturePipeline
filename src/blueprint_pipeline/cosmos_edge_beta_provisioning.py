"""Compatibility entrypoint for the pinned Cosmos server bootstrap."""
from .cosmos_droid_bundle import provisioning_script

DEPENDENCY_LOCK_SHA256 = "f5ec25154f3dc63f49680c91c0dee807b02ea6b846833d6f204bc5659f67e555"


def build_provisioning_script() -> str:
    return provisioning_script()
