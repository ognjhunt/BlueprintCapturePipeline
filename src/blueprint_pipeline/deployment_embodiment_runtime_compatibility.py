"""Read unchanged configurations from two exact corrected predecessor releases.

Only 9bd34559c (18 profiles) and b0ccd89e0 (those 18 plus the single FR3)
are accepted. Both follow the TRLC/Axol closure correction. Each complete derived
contract is reconstructed from the retained, verified model/source and current
matching profile before reuse. All source,
model, generated-file inventory and receipt digests remain checked by caller.
"""
from __future__ import annotations

from pathlib import Path

PREDECESSOR_GENERATOR = "sha256:85fc6a0fda052d9825ae32bd6a2722b94200e631469b6148d8c5c2ec1ecedf40"
PREDECESSOR_PROFILES = frozenset({
    "agilex_piper_bimanual", "almond_axol", "fanuc_crx_10ia", "fanuc_crx_10ia_l", "fanuc_m710ic_45m",
    "franka_research_3_bimanual", "i2rt_yam_bimanual", "kinova_gen3", "kinova_gen3_bimanual", "kuka_lbr_iiwa_14",
    "trlc_dk1_bimanual", "ufactory_xarm7", "ufactory_xarm7_bimanual", "universal_robots_ur10e",
    "universal_robots_ur5e", "universal_robots_ur5e_bimanual", "yaskawa_gp12", "yaskawa_gp20",
})

B0_GENERATOR = "sha256:38bc862ed4178d8a0d35be10b15451e8122cf5af4ff3b7acb1b74fece7c45d69"
B0_PROFILES = PREDECESSOR_PROFILES | {"franka_research_3_robotiq_2f85_sim"}


def is_exact_predecessor(record: dict) -> bool:
    profiles = {PREDECESSOR_GENERATOR: PREDECESSOR_PROFILES, B0_GENERATOR: B0_PROFILES}
    return record.get("embodiment_id") in profiles.get(record.get("generator_module_sha256"), ())


def verify_predecessor_metadata(root: Path, source: dict, record: dict) -> None:
    """No Xacro, SDK, temporary writes or network are needed for retained reuse."""
    from . import deployment_embodiment_runtime_configuration as runtime
    from .deployment_embodiment_assets import recipe_for, source_description

    runtime._require(is_exact_predecessor(record), "predecessor_not_supported")
    identity = source["embodiment_id"]
    recipe = recipe_for(identity)
    description = source_description(recipe, source)
    profiles = runtime._PROFILES[identity]
    if description["format"] == "mjcf":
        from . import deployment_embodiment_mjcf_configuration as mjcf
        entry = root / "source" / description["entry_path"]
        expected = mjcf.configuration_from_mjcf(entry, identity, profiles[0], source)
        expected["source_model"] = {"format": "mjcf", "path": entry.relative_to(root).as_posix(),
                                    "sha256": runtime._sha(entry)}
        expected["metadata_generator_module_sha256"] = runtime._sha(Path(mjcf.__file__))
    else:
        # These bytes and the complete expansion inventory have already been
        # verified. Do not rerun Xacro: the retained expansion includes its
        # original staging paths, and read-only model consumers need no Xacro.
        entry = (root / "expanded_source" / description["entry_path"]).with_suffix(".urdf") if (
            description["format"] == "xacro") else root / "source" / description["entry_path"]
        source_root, _ = runtime._tree(runtime.SafeXML.fromstring(entry.read_bytes()))
        arms = [runtime._prefix_arm(arm, arm.get("source_prefix", "")) for arm in profiles]
        parameters = None
        if recipe["arm_count"] == 2 and source["source_arm_count"] == 1:
            parameters = {"schema_version": "embodiment_station_parameters.v1", "embodiment_id": identity,
                "composition_mode": "two_instances", "station_root_link": "blueprint_station",
                "arms": [{"name": side, "prefix": side + "_", "source_base_link": source_root,
                          "source_mount_joint": None,
                          "station_from_base": {"translation_m": [0.0, y, 0.0], "quaternion_wxyz": [1, 0, 0, 0]}}
                         for side, y in (("left", .35), ("right", -.35))]}
            arms = [runtime._prefix_arm(arms[0], side + "_") for side in ("left", "right")]
        model = root / "robot.urdf"
        expected = runtime._configuration(runtime.SafeXML.fromstring(model.read_bytes()), identity, arms, source, parameters)
        expected["source_urdf_sha256"] = runtime._sha(model)
    # Check the complete derived payload, not selected manifest fields. This
    # also rejects a changed camera, reset, TCP, gain, dynamics warning, mount,
    # or reversed jaw mapping even if someone recomputes the outer digest.
    retained = {key: value for key, value in record.items()
                if key not in {"generator_module_sha256", "configuration_digest", "generated_files"}}
    runtime._require(retained == expected, "predecessor_metadata_changed")
