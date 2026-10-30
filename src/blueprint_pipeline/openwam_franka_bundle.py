"""Exact Franka checkpoint plus shared pinned OpenWAM source/dependency artifact closure."""
from . import openwam_piper_bundle as shared
from .openwam_franka_assets import EXPECTED_FILES
from .openwam_franka_policy import INTERFACE_ID, MODEL_ID, MODEL_REVISION

RUNTIME_MODULES = (*shared.RUNTIME_MODULES, "openwam_franka_policy.py", "openwam_franka_assets.py",
    "openwam_franka_policy_server.py", "openwam_franka_runtime.py", "openwam_franka_bundle.py", "openwam_franka_provisioning.py")


def _profile():
    return {"schema_version": "openwam_franka_sealed_runtime.v1", "policy_interface_id": INTERFACE_ID,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "checkpoint_download_bytes": sum(size for size, _ in EXPECTED_FILES.values())}


def stage_runtime(runtime):
    return shared.stage_runtime(runtime, _binding=_profile())


def validate_runtime_contract(value):
    return shared.validate_runtime_contract(value, _binding=_profile())


def transfer_byte_budget(bundle):
    return shared.transfer_byte_budget(bundle, _binding=_profile())


def provisioning_script():
    return shared.provisioning_script(_module="blueprint_pipeline.openwam_franka_provisioning")
