"""Read unchanged configurations from exact corrected predecessor releases.

The exact 9bd34559c (18), b0ccd89e0 (19), 2c2977ae4 (21) and later 23/28/30/31
profile generators are accepted for their original profiles only. All follow the TRLC/Axol closure correction. Each complete derived
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


CURRENT_21_GENERATOR = "sha256:d6c0d849fb7d932148480f38bc3a3015750d260128a8d74f586367cbae6f30c1"
CURRENT_21_PROFILES = B0_PROFILES | {"abb_gofa_crb15000_5_95", "abb_gofa_crb15000_12_127"}

CURRENT_23_GENERATOR = "sha256:4507e03e3d492453ba6e4b194bdffb1a95e9a2551a1bfe6ed37498c7911f5c73"
CURRENT_23_PROFILES = CURRENT_21_PROFILES | {"universal_robots_ur5e_robotiq_2f85_sim", "universal_robots_ur10e_robotiq_2f85_sim"}


CURRENT_28_GENERATOR = 'sha256:1a5c110a25b04daddfad1c280a95ba40a6877dc0eb250f29f3120f9728274f68'
CURRENT_28_PROFILES = CURRENT_23_PROFILES | {'fanuc_crx_10ia_l_robotiq_2f85_sim', 'fanuc_crx_10ia_robotiq_2f85_sim', 'fanuc_m710ic_45m_robotiq_2f85_sim', 'abb_gofa_crb15000_12_127_robotiq_2f85_sim', 'abb_gofa_crb15000_5_95_robotiq_2f85_sim'}


CURRENT_30_GENERATOR = "sha256:fb94386f26a8516680dd136b53884000d0420db47a520700bd920864bdef5f89"
CURRENT_30_PROFILES = CURRENT_28_PROFILES | {"yaskawa_gp12_robotiq_2f85_sim", "yaskawa_gp20_robotiq_2f85_sim"}


CURRENT_31_GENERATOR = "sha256:1322ed108a1a9ad969221f3d2f732397fde0571d7a9ac038b80582ac550946a8"
CURRENT_31_PROFILES = CURRENT_30_PROFILES | {"kuka_lbr_iiwa_14_robotiq_2f85_sim"}


def is_exact_predecessor(record: dict) -> bool:
    profiles = {PREDECESSOR_GENERATOR: PREDECESSOR_PROFILES, B0_GENERATOR: B0_PROFILES,
                CURRENT_21_GENERATOR: CURRENT_21_PROFILES, CURRENT_23_GENERATOR: CURRENT_23_PROFILES,
                CURRENT_28_GENERATOR: CURRENT_28_PROFILES, CURRENT_30_GENERATOR: CURRENT_30_PROFILES,
                CURRENT_31_GENERATOR: CURRENT_31_PROFILES}
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
        from .deployment_embodiment_simulation_models import for_identity
        gofa = for_identity(identity)
        if gofa is not None:
            # Reconstruct the original source limitations and complete declared
            # estimates. Retained source/generated bytes remain immutable.
            original_root = runtime.SafeXML.fromstring((root / gofa.SOURCE_MODEL).read_bytes())
            original = runtime._configuration(original_root, identity, arms, source, parameters)
            report = gofa.model_urdf(original_root, identity=identity, anchor=root)
            gofa.annotate_configuration(expected, original, report)
    # Check the complete derived payload, not selected manifest fields. This
    # also rejects a changed camera, reset, TCP, gain, dynamics warning, mount,
    # or reversed jaw mapping even if someone recomputes the outer digest.
    retained = {key: value for key, value in record.items()
                if key not in {"generator_module_sha256", "configuration_digest", "generated_files"}}
    runtime._require(retained == expected, "predecessor_metadata_changed")
