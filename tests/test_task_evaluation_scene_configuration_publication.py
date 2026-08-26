from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from blueprint_pipeline.task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from blueprint_pipeline.task_evaluation_scene_configuration_publication import (
    publish_configured_scene_revision,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    test_configuration_request as configuration_request_fixture,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def test_control_plane_publishes_reads_back_and_seals_robot_neutral_revision(
    tmp_path: Path,
) -> None:
    request = configuration_request_fixture()
    artifacts = tmp_path / "provider-artifacts"
    artifacts.mkdir()
    roles = {
        "configured_appearance_without_source_object": "appearance.usdc",
        "appearance_removal_receipt": "appearance-receipt.json",
        "configured_collision_without_source_object": "collision.usda",
        "collision_excision_receipt": "collision-receipt.json",
        "statically_qualified_replacement_asset": "static-replacement.usda",
        "static_qualification_receipt": "static-receipt.json",
        "native_qualified_replacement_asset": "replacement.usda",
        "native_import_qualification_receipt": "native-receipt.json",
        "configured_scene_bundle_candidate_manifest": "bundle-candidate.json",
        "scene_assembly_receipt": "assembly-receipt.json",
    }
    rows = []
    for role, name in roles.items():
        path = artifacts / name
        path.write_bytes((role + "\n").encode())
        rows.append(_artifact(role, path))
    stage_results = [{"output_artifacts": rows}]
    envelope = {
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "request": request,
        "recipe": {
            "scene_identity": request["scene"]["identity"],
            "task_identity": request["task"]["identity"],
            "subject_identity": request["task"]["subject"]["identity"],
            "provider_disclosure": {
                "raw_source_bytes_to_external_provider": False,
            },
        },
    }
    output = tmp_path / "publication"
    output.mkdir()
    object_store = tmp_path / "object-store"
    object_store.mkdir()

    def publish(*, path: Path, object_name: str):
        destination = object_store / object_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {
            "uri": f"s3://blueprint-production-inputs/{object_name}",
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": _sha256(destination),
            "readback_size_bytes": destination.stat().st_size,
        }

    result = publish_configured_scene_revision(
        envelope=envelope,
        stage_results=stage_results,
        output_root=output,
        publisher=publish,
    )

    revision_path = Path(result["configured_scene_revision"]["path"])
    revision = validate_configured_scene_revision(
        json.loads(revision_path.read_text())
    )
    assert revision["robot_team_interface"][
        "episode_packet_compiled_by_production"
    ] is True
    assert revision["robot_team_interface"][
        "configuration_run_executed_episode"
    ] is False
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["configured_scene_revision_reference"]["uri"].startswith(
        "s3://blueprint-production-inputs/"
    )
    assert result["provider_mutation_performed"] is False
