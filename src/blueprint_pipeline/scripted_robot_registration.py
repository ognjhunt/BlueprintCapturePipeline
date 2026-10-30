"""Control-plane construction of owned-script registrations; no install or spend."""
from collections.abc import Mapping
from typing import Any

from .scripted_robot_baseline_runtime import candidate_id, MINIMUM_GPU_MEMORY_GB
from .scripted_robot_runtime_dispatch import build_policy_spec
from .team_embodiment import validate_embodiment_manifest


def registration_from_prepared_inputs(*, prepared: Mapping[str, Any], model_path: Any, destination: Any,
                                      team_id: str, checkpoint_id: str, rights: Mapping[str, Any]) -> dict[str, Any]:
    """Build one registration from verified import bytes; never install or launch.

    The caller must supply the existing scoped rights record. This helper does
    not invent a model license, capture consent or budget authorization.
    """
    from .beta_policy_screening_team import registration_from_prepared_inputs as register

    manifest = validate_embodiment_manifest(prepared["robot_embodiment"])
    identity = candidate_id(manifest["embodiment_id"])
    registration = register(prepared=prepared, model_path=model_path, destination=destination,
        candidate_id=identity, team_id=team_id, checkpoint_id=checkpoint_id,
        policy_spec=build_policy_spec(manifest), rights=rights,
        display_name="Blueprint scripted pick/place — " + manifest["embodiment_id"])
    registration.update(minimum_gpu_memory_gb=MINIMUM_GPU_MEMORY_GB,
        deployment={"configuration_id": manifest["embodiment_id"], "policy_route_id": "blueprint_scripted_baseline",
            "companies": [], "relationship": "public_controller_integration",
            "evidence_url": "BlueprintCapturePipeline:docs/scripted_robot_baseline.md"})
    return registration

