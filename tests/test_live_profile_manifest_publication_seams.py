"""Keep generated-manifest publication complete across the live catalog."""

from __future__ import annotations

from pathlib import Path
import json
import re

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.robot_eval_provider_input_setup import (
    LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    bind_live_profile_manifest_publication,
    file_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_REACHABILITY = (
    REPO_ROOT / "docs" / "arm_decision_proof_v1" / "LIVE_LANE_REACHABILITY.md"
)


def test_every_website_reachable_builder_has_digest_safe_publication() -> None:
    documented = set(
        re.findall(
            r"\| `[^`]+` \| `(build_[^`]+_live_profile\.py)` \|",
            LIVE_REACHABILITY.read_text(encoding="utf-8"),
        )
    )
    assert len(documented) == 15
    assert documented == set(LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS)
    assert {
        builder
        for builder, seam in LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS.items()
        if seam == "exact_commit_raw_github"
    } == {"build_adp009d_840313_live_profile.py"}
    assert set(LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS.values()) == {
        "exact_commit_raw_github",
        "content_addressed_full_readback",
    }
    for builder in documented:
        source = REPO_ROOT / "scripts" / builder
        assert source.is_file()
        source_text = source.read_text(encoding="utf-8")
        if LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS[builder] == "exact_commit_raw_github":
            assert "RAW_GITHUB_ROOT" in source_text
            assert 'source_uri = f"{RAW_GITHUB_ROOT}/{source_commit}/' in source_text
        else:
            # Shared builders bind this filename in LaneLiveProfileSpec; the
            # two custom builders pass it directly to the same helper.
            assert f'profile_builder="{builder}"' in source_text
            assert "Local digest-bound content-addressed publication receipt" in source_text


@pytest.mark.parametrize(
    "profile_builder",
    sorted(
        builder
        for builder, seam in LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS.items()
        if seam == "content_addressed_full_readback"
    ),
)
def test_each_generated_builder_executes_the_strict_publication_binder(
    profile_builder: str, tmp_path: Path
) -> None:
    commit = "3" * 40
    source = tmp_path / f"{profile_builder}.json"
    source.write_text('{"schema_version":"request.v1"}\n', encoding="utf-8")
    digest = file_digest(source)
    identity = digest.removeprefix("sha256:")
    receipt = {
        "schema_version": "task_evaluation_immutable_manifest_publication.v1",
        "status": "published",
        "source": {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256": digest,
        },
        "profile_builder": profile_builder,
        "publication_seam": "content_addressed_full_readback",
        "published_uri": f"gs://fixture/sha256/{identity[:2]}/{identity}.json",
        "storage_scheme": "gs",
        "remote_size_bytes": source.stat().st_size,
        "remote_sha256": digest,
        "provider_full_byte_readback_verified": True,
        "content_addressed_key": True,
        "exclusive_create": True,
        "upload_receipt_digest": "sha256:" + "a" * 64,
        "provider_compute_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = tmp_path / f"{profile_builder}.publication.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    inputs = [
        {"name": "source_bundle_manifest", "path": str(source), "digest": digest}
    ]

    assert bind_live_profile_manifest_publication(
        reference=str(receipt_path),
        source_commit=commit,
        run_spec_digest=digest,
        profile_builder=profile_builder,
        immutable_inputs=inputs,
    )[0] == receipt["published_uri"]
    for bypass in (
        f"https://raw.githubusercontent.com/example/repo/{commit}/request.json",
        "gs://fixture/bare.json",
    ):
        with pytest.raises(
            TaskEvaluationLaunchError, match="manifest_publication_receipt_required"
        ):
            bind_live_profile_manifest_publication(
                reference=bypass,
                source_commit=commit,
                run_spec_digest=digest,
                profile_builder=profile_builder,
                immutable_inputs=inputs,
            )


def test_generated_and_exact_commit_publication_paths_are_fail_closed(
    tmp_path: Path,
) -> None:
    commit = "1" * 40
    source = tmp_path / "request.json"
    source.write_text('{"schema_version":"request.v1"}\n', encoding="utf-8")
    digest = file_digest(source)
    receipt = {
        "schema_version": "task_evaluation_immutable_manifest_publication.v1",
        "status": "published",
        "source": {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256": digest,
        },
        "profile_builder": "build_retained_scene_render_live_profile.py",
        "publication_seam": "content_addressed_full_readback",
        "published_uri": (
            "gs://fixture/sha256/"
            f"{digest.removeprefix('sha256:')[:2]}/"
            f"{digest.removeprefix('sha256:')}.json"
        ),
        "storage_scheme": "gs",
        "remote_size_bytes": source.stat().st_size,
        "remote_sha256": digest,
        "provider_full_byte_readback_verified": True,
        "content_addressed_key": True,
        "exclusive_create": True,
        "upload_receipt_digest": "sha256:" + "a" * 64,
        "provider_compute_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = tmp_path / "publication.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    inputs = [{"name": "source_bundle_manifest", "path": str(source), "digest": digest}]

    uri, binding, bound_inputs = bind_live_profile_manifest_publication(
        reference=str(receipt_path),
        source_commit=commit,
        run_spec_digest=digest,
        profile_builder="build_retained_scene_render_live_profile.py",
        immutable_inputs=inputs,
    )
    assert uri == receipt["published_uri"]
    assert binding is not None
    assert bound_inputs[-1]["name"] == "manifest_publication_receipt"

    for scheme, seam in (
        ("s3", "content_addressed_full_readback"),
        ("r2", "content_addressed_full_readback"),
        ("gs", "gcs_content_addressed_full_readback"),
    ):
        compatible = json.loads(json.dumps(receipt))
        compatible["publication_seam"] = seam
        compatible["published_uri"] = (
            f"{scheme}://fixture/sha256/"
            f"{digest.removeprefix('sha256:')[:2]}/"
            f"{digest.removeprefix('sha256:')}.json"
        )
        compatible["storage_scheme"] = scheme
        compatible["receipt_digest"] = canonical_digest(
            compatible, digest_field="receipt_digest"
        )
        compatible_path = tmp_path / f"publication-{scheme}-{seam}.json"
        compatible_path.write_text(json.dumps(compatible), encoding="utf-8")
        assert bind_live_profile_manifest_publication(
            reference=str(compatible_path),
            source_commit=commit,
            run_spec_digest=digest,
            profile_builder="build_retained_scene_render_live_profile.py",
            immutable_inputs=inputs,
        )[0] == compatible["published_uri"]

    for name, mutate in (
        (
            "wrong-key",
            lambda value: value.update(
                published_uri=(
                    "gs://fixture/sha256/ff/" + "f" * 64 + ".json"
                )
            ),
        ),
        (
            "wrong-size",
            lambda value: value["source"].update(
                size_bytes=source.stat().st_size + 1
            ),
        ),
        (
            "provider-mutation",
            lambda value: value.update(provider_compute_mutation_performed=True),
        ),
    ):
        tampered = json.loads(json.dumps(receipt))
        mutate(tampered)
        tampered["receipt_digest"] = canonical_digest(
            tampered, digest_field="receipt_digest"
        )
        tampered_path = tmp_path / f"publication-{name}.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(
            TaskEvaluationLaunchError, match="manifest_publication_receipt_invalid"
        ):
            bind_live_profile_manifest_publication(
                reference=str(tampered_path),
                source_commit=commit,
                run_spec_digest=digest,
                profile_builder="build_retained_scene_render_live_profile.py",
                immutable_inputs=inputs,
            )

    different_source = tmp_path / "different-request.json"
    different_source.write_text('{"schema_version":"request.v2"}\n', encoding="utf-8")
    different_digest = file_digest(different_source)
    different_identity = different_digest.removeprefix("sha256:")
    different = json.loads(json.dumps(receipt))
    different["source"] = {
        "path": str(different_source.resolve()),
        "size_bytes": different_source.stat().st_size,
        "sha256": different_digest,
    }
    different["published_uri"] = (
        f"gs://fixture/sha256/{different_identity[:2]}/{different_identity}.json"
    )
    different["remote_size_bytes"] = different_source.stat().st_size
    different["remote_sha256"] = different_digest
    different["receipt_digest"] = canonical_digest(
        different, digest_field="receipt_digest"
    )
    different_path = tmp_path / "publication-different-source.json"
    different_path.write_text(json.dumps(different), encoding="utf-8")
    with pytest.raises(
        TaskEvaluationLaunchError, match="manifest_publication_receipt_invalid"
    ):
        bind_live_profile_manifest_publication(
            reference=str(different_path),
            source_commit=commit,
            run_spec_digest=digest,
            profile_builder="build_retained_scene_render_live_profile.py",
            immutable_inputs=inputs,
        )

    with pytest.raises(
        TaskEvaluationLaunchError, match="manifest_publication_source_not_immutable_input"
    ):
        bind_live_profile_manifest_publication(
            reference=str(receipt_path),
            source_commit=commit,
            run_spec_digest=digest,
            profile_builder="build_retained_scene_render_live_profile.py",
            immutable_inputs=[*inputs, dict(inputs[0])],
        )
    for bypass in (
        f"https://raw.githubusercontent.com/example/repo/{commit}/request.json",
        "gs://fixture/stale/request.json",
    ):
        with pytest.raises(TaskEvaluationLaunchError, match="publication_receipt_required"):
            bind_live_profile_manifest_publication(
                reference=bypass,
                source_commit=commit,
                run_spec_digest=digest,
                profile_builder="build_retained_scene_render_live_profile.py",
                immutable_inputs=inputs,
            )

    exact = f"https://raw.githubusercontent.com/example/repo/{commit}/request.json"
    assert bind_live_profile_manifest_publication(
        reference=exact,
        source_commit=commit,
        run_spec_digest=digest,
        profile_builder="build_adp009d_840313_live_profile.py",
        immutable_inputs=inputs,
    )[0] == exact
    with pytest.raises(TaskEvaluationLaunchError, match="exact_commit_raw_github"):
        bind_live_profile_manifest_publication(
            reference=exact.replace(commit, "2" * 40),
            source_commit=commit,
            run_spec_digest=digest,
            profile_builder="build_adp009d_840313_live_profile.py",
            immutable_inputs=inputs,
        )
