from blueprint_pipeline.production_gpu_release_candidate import build_release_candidate_manifest


def _manifest(**overrides):
    values = dict(
        source_sha="a" * 40,
        clean_worktree=True,
        build_context_digest="sha256:" + "b" * 64,
        dockerfile_digest="sha256:" + "c" * 64,
        base_image_ref="registry/base@sha256:" + "d" * 64,
        dependency_digests={"uv.lock": "sha256:" + "e" * 64},
        worker_source_digests={"worker.py": "sha256:" + "f" * 64},
        model_asset_revisions={"groot": "repo@immutable-revision", "kitchen": "sha256:asset"},
        runtime_contract={
            "runtime_user": "blueprint",
            "uid": 10001,
            "gid": 10001,
            "supplementary_groups": ["video", "render"],
            "entrypoint": ["/entrypoint"],
            "command": ["serve"],
            "required_environment_names": ["OUTPUT_URI"],
            "oci_runtime": "nvidia",
            "gpu_access": "all",
        },
        build_command=["scripts/build_push.sh", "--sealed"],
        image_tag="registry/worker:rc-a",
        pushed_image_ref="registry/worker@sha256:" + "1" * 64,
        build_timestamp="2026-07-14T00:00:00Z",
        builder_identity="github:run:1",
        sbom_ref="oci://sbom",
        provenance_ref="oci://provenance",
    )
    values.update(overrides)
    return build_release_candidate_manifest(**values)


def test_release_candidate_seals_complete_closure_deterministically():
    first = _manifest(build_timestamp="one")
    second = _manifest(build_timestamp="two")
    assert first["status"] == "sealed"
    assert first["build_input_fingerprint"] == second["build_input_fingerprint"]
    assert first["pushed_image_ref"].endswith("1" * 64)


def test_release_candidate_rejects_dirty_or_mutable_inputs():
    result = _manifest(
        clean_worktree=False, base_image_ref="ubuntu:latest", image_tag="registry/worker:latest"
    )
    assert result["status"] == "blocked"
    assert "build_worktree_not_clean" in result["blockers"]
    assert "base_image_not_digest_pinned" in result["blockers"]
    assert "immutable_versioned_image_tag_required" in result["blockers"]
