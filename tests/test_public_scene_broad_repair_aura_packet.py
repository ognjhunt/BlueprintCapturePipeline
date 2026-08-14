import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_broad_repair_aura_packet import (
    materialize_broad_repair_aura_packet,
)
from blueprint_pipeline.public_scene_broad_repair_support import (
    materialize_broad_repair_support,
)
from tests.test_public_scene_broad_repair_support import _fixture, _write


def test_bridges_five_task_broad_support_to_strict_aura_packet(tmp_path: Path) -> None:
    candidate, result, relocation = _fixture(tmp_path, task_count=5)
    support_root = tmp_path / "support"
    support = materialize_broad_repair_support(
        candidate_set_path=candidate,
        renderer_result_path=result,
        output_relocation_receipt_path=relocation,
        output_root=support_root,
        repair_support_dilation_pixels=1,
    )
    backend: dict[str, object] = {
        "schema_version": "public_scene_released_code_inpainting_admission.v1",
        "status": "rights_admitted_for_private_derived_inpainting",
        "backend_id": "aurafusion360_exact_residual_multiview",
        "receipt_digest": "",
    }
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    backend_path = _write(tmp_path / "backend.json", backend)

    packet = materialize_broad_repair_aura_packet(
        broad_support_packet_path=(
            support_root / "public_scene_broad_repair_support_packet.v1.json"
        ),
        backend_admission_path=backend_path,
        output_root=tmp_path / "aura_packet",
    )

    assert packet["replacement_object_count"] == 5
    assert packet["maximum_replacement_objects"] == 5
    assert packet["broad_repair_support"]["receipt_digest"] == support["receipt_digest"]
    assert packet["claim_boundary"]["repair_support_includes_all_detectable_deleted_projection"]
    for lane_record in packet["lanes"]:
        lane = json.loads(Path(lane_record["path"]).read_text(encoding="utf-8"))
        assert lane["inpainting_execution_authorized"] is False
        assert lane["inpainting_result_qualified"] is False
        assert lane["exact_residual_masks"][0]["pixel_count"] == 4
