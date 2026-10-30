"""Exact single-FR3 OpenWAM server; two real views, upstream's configured null third."""
from .openwam_franka_assets import verify_checkpoint_directory
from .openwam_franka_policy import OpenWAMFrankaBinding, WIRE_CAMERAS
from .openwam_piper_policy_server import OpenWAMPiperServer, _build_server, _main
from .openwam_piper_policy_server import validate_wire_observation as _validate


def validate_wire_observation(value):
    return _validate(value, cameras=WIRE_CAMERAS, state_width=10, gripper_indices=(9,), gripper_bounds=(-1, 1),
                     image_size=(640, 480))


class OpenWAMFrankaServer(OpenWAMPiperServer):
    validate_observation = staticmethod(validate_wire_observation)


def build_server(*, checkpoint, source_root, binding, source_manifest=None):
    return _build_server(checkpoint=checkpoint, source_root=source_root, binding=binding, source_manifest=source_manifest,
        checkpoint_verifier=verify_checkpoint_directory, server_type=OpenWAMFrankaServer,
        checkpoint_name="checkpoint_step_9860.safetensors")


def main(argv=None):
    _main(argv, binding_type=OpenWAMFrankaBinding, builder=build_server, verifier=verify_checkpoint_directory, default_port=8848)


if __name__ == "__main__":
    main()
