from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.standard_splat_conversion import (
    StandardSplatConversionError,
    build_standard_splat_conversion_request,
    materialize_standard_splat_conversion,
)


def _request(source: Path, data: Path) -> dict:
    import hashlib

    value = {
        "schema_version": "standard_splat_conversion_request.v1",
        "program_id": "arm-decision-proof-v1",
        "frozen_before_conversion": True,
        "learned_policy_outcomes_observed": False,
        "source": {
            "relative_path": source.relative_to(data).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
            "dataset": "fixture-dataset",
            "revision": "a" * 40,
            "license": "fixture-license",
        },
        "rights": {
            "terms_digest": "sha256:" + "b" * 64,
            "conversion_execution_location": "local_only",
            "raw_private_upload_authorized": False,
            "training_authorized": False,
        },
        "output_filename": "scene_standard.ply",
    }
    return build_standard_splat_conversion_request(value)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    source = data / "source" / "compressed.ply"
    source.parent.mkdir(parents=True)
    count = 18
    write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
            opacity=np.ones(count, dtype=np.float32),
            f_dc=np.zeros((count, 3), dtype=np.float32),
            scales=np.zeros((count, 3), dtype=np.float32),
            quats=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (count, 1),
            ),
            properties=(),
        ),
        source,
    )
    cli = (
        repo
        / "tools/splat_render/node_modules/@playcanvas/splat-transform/bin/cli.mjs"
    )
    cli.parent.mkdir(parents=True)
    cli.write_text("// fixture\n", encoding="utf-8")
    package = cli.parents[1] / "package.json"
    package.write_text(json.dumps({"version": "3.2.0"}), encoding="utf-8")
    request = repo / "request.json"
    request.write_text(json.dumps(_request(source, data)), encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True
    )
    return {"repo": repo, "data": data, "source": source, "request": request}


def test_local_conversion_materializer_binds_tool_rights_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)

    def copy_conversion(source, destination, **_kwargs):
        shutil.copy2(source, destination)
        return {
            "status": "completed",
            "decoder": "playcanvas_splat_transform",
            "output_bytes": destination.stat().st_size,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.standard_splat_conversion.convert_to_standard_ply",
        copy_conversion,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.standard_splat_conversion.read_compressed_ply_chunk_bounds",
        lambda _path: SimpleNamespace(vertex_count=18),
    )
    receipt = materialize_standard_splat_conversion(
        request_path=paths["request"],
        repo_root=paths["repo"],
        data_root=paths["data"],
        output_root=paths["data"] / "conversion",
        receipt_output=paths["repo"] / "retained" / "receipt.json",
    )

    assert receipt["source"]["source_gaussian_count"] == 18
    assert receipt["output"]["gaussian_count_preserved"] is True
    assert receipt["decoder"]["version"] == "3.2.0"
    assert receipt["raw_source_uploaded"] is False
    assert receipt["gaussian_ownership_claimed"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_request_rejects_provider_conversion_or_raw_upload() -> None:
    value = {
        "schema_version": "standard_splat_conversion_request.v1",
        "program_id": "arm-decision-proof-v1",
        "frozen_before_conversion": True,
        "learned_policy_outcomes_observed": False,
        "source": {
            "relative_path": "scene/source.ply",
            "size_bytes": 1,
            "sha256": "sha256:" + "a" * 64,
            "dataset": "fixture",
            "revision": "b" * 40,
            "license": "fixture",
        },
        "rights": {
            "terms_digest": "sha256:" + "c" * 64,
            "conversion_execution_location": "provider",
            "raw_private_upload_authorized": True,
            "training_authorized": False,
        },
        "output_filename": "scene.ply",
    }
    with pytest.raises(StandardSplatConversionError) as caught:
        build_standard_splat_conversion_request(value)
    assert "standard_splat_rights_invalid" in caught.value.codes


def test_request_rejects_malformed_source_and_terms_digests() -> None:
    value = {
        "schema_version": "standard_splat_conversion_request.v1",
        "program_id": "arm-decision-proof-v1",
        "frozen_before_conversion": True,
        "learned_policy_outcomes_observed": False,
        "source": {
            "relative_path": "scene/source.ply",
            "size_bytes": 1,
            "sha256": "sha256:not-a-digest",
            "dataset": "fixture",
            "revision": "b" * 40,
            "license": "fixture",
        },
        "rights": {
            "terms_digest": "sha256:also-invalid",
            "conversion_execution_location": "local_only",
            "raw_private_upload_authorized": False,
            "training_authorized": False,
        },
        "output_filename": "scene.ply",
    }
    with pytest.raises(StandardSplatConversionError) as caught:
        build_standard_splat_conversion_request(value)
    assert set(caught.value.codes) == {
        "standard_splat_rights_invalid",
        "standard_splat_source_sha256_invalid",
    }
