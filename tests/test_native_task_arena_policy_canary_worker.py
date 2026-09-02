from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_worker import (
    _construction_lineage_mode,
)


def _scene_plan() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "interiorgs-839873",
        "task_id": "scene-839873-mug-planar-push",
        "plan_digest": "",
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def _compiled_result(scene_revision_digest: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_episode_compilation_result.v1",
        "status": "compiled_for_production_launch",
        "blockers": [],
        "configured_scene_revision_digest": scene_revision_digest,
        "compiled_episode_packet_digest": "sha256:" + "e" * 64,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_controls_pending_canary_accepts_digest_bound_compiled_scene_lineage() -> None:
    revision = "sha256:" + "2" * 64
    mode = _construction_lineage_mode(
        inputs={"scene_revision_digest": revision},
        base_scene_plan=_scene_plan(),
        construction=_compiled_result(revision),
    )
    assert mode == "compiled_configured_scene_diagnostic"


def test_compiled_scene_lineage_cannot_change_the_scene_revision() -> None:
    with pytest.raises(
        RuntimeError, match="policy_canary_compiled_scene_lineage_invalid"
    ):
        _construction_lineage_mode(
            inputs={"scene_revision_digest": "sha256:" + "3" * 64},
            base_scene_plan=_scene_plan(),
            construction=_compiled_result("sha256:" + "2" * 64),
        )


def test_qualified_native_construction_path_remains_strict() -> None:
    plan = _scene_plan()
    construction: dict[str, object] = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "scene_plan_digest": plan["plan_digest"],
        "result_digest": "",
    }
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    assert _construction_lineage_mode(
        inputs={"scene_revision_digest": "sha256:" + "2" * 64},
        base_scene_plan=plan,
        construction=construction,
    ) == "qualified_native_construction_result"


def _plan_with_appearance(representation: str = "particlefield_3d_gaussian_splat") -> dict:
    return {
        "objects": [
            {"name": "task_object", "semantic_role": "task_object", "task_subject": True},
            {
                "name": "scene_appearance",
                "semantic_role": "scene_appearance",
                "sha256": "1bfd4438e057587c785b8211a70e26b896dd1ef90626e7923c541dbfd7c125cc",
            },
        ],
        "appearance_frame_alignment": {
            "status": "aligned",
            "representation": representation,
            "source_asset_sha256": (
                "sha256:9193a9de6bd81bd6348065b3cad46ad835b62dcfaa6212285a91bffd8a166445"
            ),
        },
    }


def test_appearance_backend_is_sealed_from_the_plan_not_a_default() -> None:
    from blueprint_pipeline.native_task_arena_policy_canary_worker import (
        appearance_render_backend_from_plan,
    )

    backend = appearance_render_backend_from_plan(_plan_with_appearance())
    assert backend["kind"] == "particlefield_blueprint_private_tensor_conversion"
    assert backend["development_only"] is True
    assert backend["launch_render_path"] == "particlefield_3d_gaussian_splat"
    assert backend["source_asset_digest"].startswith("sha256:9193a9de")
    assert backend["derived_asset_digest"].startswith("sha256:1bfd4438")
    assert backend["receipt_digest"].startswith("sha256:")

    official = _plan_with_appearance()
    official["appearance_frame_alignment"]["conversion_identity"] = (
        "threedgrut.export.scripts.transcode@a37ef721012dea0f29c0fcfff2d525023b4e854a"
    )
    sealed = appearance_render_backend_from_plan(official)
    assert sealed["kind"] == "particlefield_3dgrut_transcode"
    assert sealed["development_only"] is False

    packet_request = {
        "appearance_variant": {
            "representation": "particlefield_3d_gaussian_splat",
            "source_gaussian_sha256": (
                "sha256:9193a9de6bd81bd6348065b3cad46ad835b62dcfaa6212285a91bffd8a166445"
            ),
            "particlefield_authoring_implementation": "nvidia_3dgrut_direct_nurec_transcode",
            "upstream_converter": {
                "source_revision": "a37ef721012dea0f29c0fcfff2d525023b4e854a",
                "source_identity_verified": True,
            },
        }
    }
    plan_without_alignment_identity = _plan_with_appearance()
    plan_without_alignment_identity["appearance_frame_alignment"].pop("source_asset_sha256")
    from_packet = appearance_render_backend_from_plan(
        plan_without_alignment_identity, packet_request=packet_request
    )
    assert from_packet["kind"] == "particlefield_3dgrut_transcode"
    assert from_packet["conversion_identity"] == (
        "threedgrut.export.scripts.transcode@a37ef721012dea0f29c0fcfff2d525023b4e854a"
    )
    legacy_request = {
        "appearance_variant": {
            **packet_request["appearance_variant"],
            "particlefield_authoring_implementation": "nvidia_usd_convert_gsplat",
            "upstream_converter": {"version": "0.1.15"},
        }
    }
    legacy = appearance_render_backend_from_plan(
        plan_without_alignment_identity, packet_request=legacy_request
    )
    assert legacy["kind"] == "particlefield_blueprint_private_tensor_conversion"
    assert legacy["development_only"] is True
    assert legacy["conversion_identity"] == "nvidia_usd_convert_gsplat@0.1.15"

    native = _plan_with_appearance("nurec_volume")
    nurec = appearance_render_backend_from_plan(native)
    assert nurec["kind"] == "isaac_native_nurec"
    assert nurec["launch_render_path"] == "plain_nurec_volume"
    assert nurec["derived_asset_digest"] is None


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda plan: plan.pop("appearance_frame_alignment"), "policy_canary_appearance_render_path_unresolved"),
        (lambda plan: plan["objects"].pop(1), "policy_canary_scene_appearance_asset_not_exact"),
        (
            lambda plan: plan["appearance_frame_alignment"].pop("source_asset_sha256"),
            "policy_canary_appearance_source_digest_missing",
        ),
    ],
)
def test_backend_sealing_fails_closed_on_missing_identity(mutate, expected: str) -> None:
    from blueprint_pipeline.native_task_arena_policy_canary_worker import (
        appearance_render_backend_from_plan,
    )

    plan = _plan_with_appearance()
    mutate(plan)
    with pytest.raises(RuntimeError, match=expected):
        appearance_render_backend_from_plan(plan)


def test_preload_gate_requires_authority_bound_to_this_backend() -> None:
    from blueprint_pipeline.native_task_arena_policy_canary_worker import (
        appearance_render_backend_from_plan,
        preload_observation_integrity_gate,
    )
    from blueprint_pipeline.native_task_camera_observability import (
        build_policy_observation_integrity_authority,
    )

    backend = appearance_render_backend_from_plan(_plan_with_appearance())

    def authority(**overrides):
        kwargs = dict(
            appearance_render_backend_receipt_digest=backend["receipt_digest"],
            reference_renderer_identity="nvcr.io/nvidia/nre/nre@sha256:pinned",
            reference_source_sha256=backend["source_asset_digest"],
            views={
                view: {
                    "reference_png_sha256": "sha256:" + "1" * 64,
                    "candidate_png_sha256": "sha256:" + "2" * 64,
                }
                for view in ("external", "wrist", "overview")
            },
            parity_passed=True,
            human_review_status="approved",
            reviewer="reviewer",
            contact_sheet_sha256="sha256:" + "3" * 64,
        )
        kwargs.update(overrides)
        return build_policy_observation_integrity_authority(**kwargs)

    missing = preload_observation_integrity_gate(None, appearance_render_backend=backend)
    assert missing["policy_observation_integrity_passed"] is False
    assert missing["candidate_policy_loaded"] is False
    assert missing["blockers"] == [
        "native_task_appearance_reference_parity_missing",
        "native_task_human_visual_review_not_approved",
    ]
    passing = preload_observation_integrity_gate(authority(), appearance_render_backend=backend)
    assert passing["policy_observation_integrity_passed"] is True
    assert passing["blockers"] == []
    other = preload_observation_integrity_gate(
        authority(appearance_render_backend_receipt_digest="sha256:" + "0" * 64),
        appearance_render_backend=backend,
    )
    assert other["blockers"] == ["native_task_appearance_reference_parity_backend_mismatch"]
    unreviewed = preload_observation_integrity_gate(
        authority(human_review_status="pending"), appearance_render_backend=backend
    )
    assert unreviewed["blockers"] == ["native_task_human_visual_review_not_approved"]
    failed = preload_observation_integrity_gate(
        authority(parity_passed=False), appearance_render_backend=backend
    )
    assert failed["blockers"] == ["native_task_appearance_reference_parity_failed"]
    invalid = preload_observation_integrity_gate({"schema_version": "x"}, appearance_render_backend=backend)
    assert invalid["policy_observation_integrity_passed"] is False
    assert invalid["blockers"]
