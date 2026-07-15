from blueprint_pipeline.thin_release_image_contract import build_thin_release_contract


def _diagnostic(ref: str, layers: list[tuple[str, int]]) -> dict:
    return {
        "status": "completed",
        "resolved_digest_ref": ref,
        "layers": [
            {"digest": digest, "size_bytes": size} for digest, size in layers
        ],
    }


def test_measures_only_layers_above_cached_foundation() -> None:
    foundation = _diagnostic("foundation@sha256:" + "a" * 64, [("sha256:base", 20_000)])
    release = _diagnostic(
        "release@sha256:" + "b" * 64,
        [("sha256:base", 20_000), ("sha256:code", 500)],
    )
    result = build_thin_release_contract(release, foundation, max_release_bytes=1_000)
    assert result["status"] == "passed"
    assert result["release_delta_compressed_size_bytes"] == 500


def test_reads_native_registry_diagnostic_layer_size_field() -> None:
    foundation = {
        **_diagnostic("foundation@sha256:" + "a" * 64, []),
        "layers": [{"digest": "sha256:base", "size": 20_000}],
    }
    release = {
        **_diagnostic("release@sha256:" + "b" * 64, []),
        "layers": [
            {"digest": "sha256:base", "size": 20_000},
            {"digest": "sha256:code", "size": 500},
        ],
    }
    result = build_thin_release_contract(release, foundation, max_release_bytes=1_000)
    assert result["status"] == "passed"
    assert result["release_delta_compressed_size_bytes"] == 500


def test_rejects_release_that_does_not_extend_exact_foundation() -> None:
    foundation = _diagnostic("foundation@sha256:" + "a" * 64, [("sha256:base", 20_000)])
    release = _diagnostic("release@sha256:" + "b" * 64, [("sha256:other", 500)])
    result = build_thin_release_contract(release, foundation)
    assert result["status"] == "blocked"
    assert "release_does_not_extend_exact_foundation_layers" in result["blockers"]


def test_rejects_release_delta_over_budget() -> None:
    foundation = _diagnostic("foundation@sha256:" + "a" * 64, [("sha256:base", 20_000)])
    release = _diagnostic(
        "release@sha256:" + "b" * 64,
        [("sha256:base", 20_000), ("sha256:code", 2_000)],
    )
    result = build_thin_release_contract(release, foundation, max_release_bytes=1_000)
    assert result["status"] == "blocked"
    assert "thin_release_compressed_delta_exceeds_budget" in result["blockers"]


def test_rejects_mutable_registry_refs_even_when_layers_match() -> None:
    foundation = _diagnostic("foundation:versioned", [("sha256:base", 20_000)])
    release = _diagnostic(
        "release:versioned", [("sha256:base", 20_000), ("sha256:code", 500)]
    )
    result = build_thin_release_contract(release, foundation)
    assert result["status"] == "blocked"
    assert "foundation_registry_ref_not_digest_pinned" in result["blockers"]
    assert "thin_release_registry_ref_not_digest_pinned" in result["blockers"]


def test_rejects_foundation_layers_out_of_order() -> None:
    foundation = _diagnostic(
        "foundation@sha256:" + "a" * 64,
        [("sha256:base-a", 20_000), ("sha256:base-b", 10_000)],
    )
    release = _diagnostic(
        "release@sha256:" + "b" * 64,
        [
            ("sha256:base-b", 10_000),
            ("sha256:base-a", 20_000),
            ("sha256:code", 500),
        ],
    )
    result = build_thin_release_contract(release, foundation)
    assert result["status"] == "blocked"
    assert "release_does_not_extend_exact_foundation_layers" in result["blockers"]
