"""Fixed single-FR3 artifact/bootstrap route inside the canonical paid worker only."""
from pathlib import Path
from types import SimpleNamespace

from . import openwam_piper_provisioning as shared
from .openwam_franka_assets import EXPECTED_FILES, verify_checkpoint_directory
from .openwam_franka_policy import MODEL_ID, MODEL_REVISION, OpenWAMFrankaPolicyClient
from .openwam_franka_runtime import policy_binding, serialized_binding, validate_execution_spec


def provision(runtime, output, spec, *, root=Path("/opt/blueprint-openwam-franka")):
    profile = SimpleNamespace(expected_files=EXPECTED_FILES, model_id=MODEL_ID, model_revision=MODEL_REVISION,
        verifier=verify_checkpoint_directory, validate_spec=validate_execution_spec, binding=policy_binding,
        serialized_binding=serialized_binding, client=OpenWAMFrankaPolicyClient,
        server_module="blueprint_pipeline.openwam_franka_policy_server")
    return shared.provision(runtime, output, spec, root=root, _profile=profile)


def main(argv=None):
    return shared._main(argv, provisioner=provision, receipt_name="openwam_franka_provisioning.v1.json")


if __name__ == "__main__":
    raise SystemExit(main())
