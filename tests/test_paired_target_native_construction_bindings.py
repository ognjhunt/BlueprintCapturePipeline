from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_construction_bindings import (
    PairedTargetNativeConstructionBindingsError,
    materialize_paired_target_native_construction_bindings,
    validate_paired_target_native_construction_bindings,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict, *, digest_field: str) -> dict:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        digest_field: value[digest_field],
    }


def _fixture(root: Path, *, count: int) -> Path:
    preflight_tasks = []
    manipulation_tasks = []
    import_rows = []
    for index in range(count):
        task_id = f"task_{index}"
        asset_id = f"asset_{index}"
        task_record = _write(
            root / task_id / "freeze.json",
            {
                "schema_version": "dual_task_task_freeze.v1",
                "task_id": task_id,
                "task_freeze_digest": "",
            },
            digest_field="task_freeze_digest",
        )
        usd = root / task_id / "registered.usda"
        usd.write_text(f'#usda 1.0\ndef Xform "asset_{index}" {{}}\n')
        usd_record = {
            "path": str(usd),
            "size_bytes": usd.stat().st_size,
            "sha256": _sha(usd),
        }
        registered_record = _write(
            root / task_id / "registered.json",
            {
                "schema_version": "registered_replacement_asset.v1",
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_digest": task_record["task_freeze_digest"],
                "output_usd": usd_record,
                "receipt_digest": "",
            },
            digest_field="receipt_digest",
        )
        probe_record = _write(
            root / "native" / "probes" / f"{index}.json",
            {
                "schema_version": "simready_replacement_native_import_probe_result.v1",
                "status": "completed",
                "asset_id": asset_id,
                "native_simulator_import_qualified": True,
                "candidate_policy_queried": False,
                "result_digest": "",
            },
            digest_field="result_digest",
        )
        preflight_tasks.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "registered_replacement_asset_receipt": registered_record,
                "registered_replacement_usd": usd_record,
            }
        )
        manipulation_tasks.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze": task_record,
                "native_construction_binding_ready": True,
                "native_task_arena_request": None,
                "pending_requirements": [
                    "native_task_arena_packet_request_missing"
                ],
            }
        )
        import_rows.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "blockers": [],
                "native_simulator_import_qualified": True,
                "probe_result_path": f"probes/{index}.json",
                "probe_result_sha256": probe_record["sha256"],
                "probe_result_digest": probe_record["result_digest"],
            }
        )
    paired_record = _write(
        root / "paired.json",
        {
            "schema_version": "paired_target_native_preflight.v1",
            "scene_id": "scene_fixture",
            "replacement_object_count": count,
            "collision_scene": {
                "path": str(root / "collision.usda"),
                "size_bytes": 1,
                "sha256": "sha256:" + "a" * 64,
            },
            "tasks": preflight_tasks,
            "receipt_digest": "",
        },
        digest_field="receipt_digest",
    )
    import_record = _write(
        root / "native" / "result.json",
        {
            "schema_version": "paired_target_native_import_runtime_result.v1",
            "status": "completed",
            "scene_id": "scene_fixture",
            "replacement_count": count,
            "all_replacements_import_qualified": True,
            "candidate_policy_queried": False,
            "replacements": import_rows,
            "result_digest": "",
        },
        digest_field="result_digest",
    )
    manipulation_record = _write(
        root / "manipulation.json",
        {
            "schema_version": "paired_target_native_manipulation_preflight.v1",
            "status": "ready_for_native_construction_bindings",
            "preflight_phase": "pre_arena",
            "scene_id": "scene_fixture",
            "replacement_object_count": count,
            "task_freeze_set_digest": "sha256:" + "b" * 64,
            "paired_target_preflight": paired_record,
            "native_import_result": import_record,
            "tasks": manipulation_tasks,
            "native_import_qualified": True,
            "native_construction_bindings_ready": True,
            "native_reachability_executed": False,
            "controls_executed": False,
            "learned_policies_executed": False,
            "blockers": [],
            "pending_requirements": sorted(
                f"task_{index}:native_task_arena_packet_request_missing"
                for index in range(count)
            ),
            "receipt_digest": "",
        },
        digest_field="receipt_digest",
    )
    return Path(manipulation_record["path"])


@pytest.mark.parametrize("count", [1, 2, 5])
def test_materializes_one_to_five_path_backed_bindings(
    tmp_path: Path, count: int
) -> None:
    source = _fixture(tmp_path / "evidence", count=count)
    output = tmp_path / "binding.json"

    result = materialize_paired_target_native_construction_bindings(
        manipulation_preflight_path=source,
        output_path=output,
    )

    assert validate_paired_target_native_construction_bindings(result) == result
    assert result["replacement_object_count"] == count
    assert len(result["bindings"]) == count
    assert result["native_reachability_qualified"] is False
    assert result["controls_executed"] is False


def test_rejects_tampered_native_import_probe(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "evidence", count=2)
    probe = tmp_path / "evidence" / "native" / "probes" / "0.json"
    probe.write_text(probe.read_text() + "\n", encoding="utf-8")

    with pytest.raises(
        PairedTargetNativeConstructionBindingsError,
        match="native_import_probe_invalid",
    ):
        materialize_paired_target_native_construction_bindings(
            manipulation_preflight_path=source,
            output_path=tmp_path / "binding.json",
        )


def test_rejects_qualified_boundary_claim_tamper(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "evidence", count=1)
    result = materialize_paired_target_native_construction_bindings(
        manipulation_preflight_path=source,
        output_path=tmp_path / "binding.json",
    )
    result["native_reachability_qualified"] = True
    result["construction_digest"] = canonical_digest(
        result, digest_field="construction_digest"
    )

    with pytest.raises(
        PairedTargetNativeConstructionBindingsError,
        match="boundary_invalid:native_reachability_qualified",
    ):
        validate_paired_target_native_construction_bindings(result)
