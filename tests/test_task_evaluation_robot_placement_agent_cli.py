from __future__ import annotations

import base64
import hashlib

from blueprint_pipeline.task_evaluation_robot_placement_agent_cli import (
    _persist_images,
    _read_mapping,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_cli_persists_digest_bound_preview_without_embedding_data_url(tmp_path) -> None:
    digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
    records = _persist_images(
        [
            {
                "label": "top_down",
                "digest": digest,
                "image_url": "data:image/png;base64,"
                + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
                "detail": "high",
            }
        ],
        output_dir=tmp_path,
        prefix="candidate-00",
    )

    assert records[0]["digest"] == digest
    assert records[0]["size_bytes"] == len(_ONE_PIXEL_PNG)
    assert records[0]["path"].endswith("candidate-00-00-top_down.png")


def test_cli_reads_only_mapping_bindings(tmp_path) -> None:
    path = tmp_path / "binding.json"
    path.write_text('{"schema_version":"fixture.v1"}\n', encoding="utf-8")

    assert _read_mapping(path, label="scene_binding") == {
        "schema_version": "fixture.v1"
    }
