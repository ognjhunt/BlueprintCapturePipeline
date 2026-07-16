import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_runpod_serverless_campaign_io as campaign_io
from blueprint_pipeline import groot_oscar_runpod_serverless_campaign_worker as worker


SOURCE = "c" * 40
IMAGE = "docker.io/example/worker@sha256:" + "d" * 64
MODEL = "sha256:" + "e" * 64
PREFIX = ".blueprint-campaigns/campaign-test"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {
        "local_path": str(path),
        "size_bytes": len(value),
        "sha256": _sha(value),
    }


def _io_evidence(tmp_path: Path) -> Path:
    rows = []
    bundle = _write(tmp_path / "bundle.zip", b"sealed-bundle")
    bundle["relative_path"] = f"{PREFIX}/input/payload.zip"
    rows.append(bundle)
    attempts = []
    for attempt_id, kind, seed, timeout in worker.EXPECTED_ATTEMPTS:
        raw = json.dumps(
            {
                "attempt_id": attempt_id,
                "source_commit": SOURCE,
                "image_digest": IMAGE.rsplit("@", 1)[-1],
            },
            sort_keys=True,
        ).encode()
        row = _write(tmp_path / f"{attempt_id}.json", raw)
        row["relative_path"] = f"{PREFIX}/input/{attempt_id}.json"
        rows.append(row)
        attempts.append(
            {
                "attempt_id": attempt_id,
                "kind": kind,
                "seed": seed,
                "timeout_seconds": timeout,
                "attempt_manifest": {
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                },
            }
        )
    campaign_payload = {
        "schema_version": worker.INPUT_SCHEMA_VERSION,
        "source_commit": SOURCE,
        "worker_image_ref": IMAGE,
        "model_manifest_digest": MODEL,
        "runtime": {
            "dynamic_episode_termination": True,
            "stop_immediately_on_declared_completion": True,
            "fixed_frame_count": None,
            "review_width": 640,
            "review_height": 480,
        },
        "payload_bundle": {
            "relative_path": bundle["relative_path"],
            "sha256": bundle["sha256"],
        },
        "attempts": attempts,
    }
    raw_campaign = json.dumps(campaign_payload, sort_keys=True).encode()
    campaign_row = _write(tmp_path / "campaign.json", raw_campaign)
    campaign_row["relative_path"] = f"{PREFIX}/input/campaign.json"
    rows.append(campaign_row)
    evidence = tmp_path / "campaign_io.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": campaign_io.EVIDENCE_SCHEMA_VERSION,
                "source_commit": SOURCE,
                "worker_image_ref": IMAGE,
                "model_manifest_digest": MODEL,
                "network_volume_id": "volume-1",
                "data_center_id": "EUR-IS-1",
                "campaign_prefix": PREFIX,
                "output_relative_path": f"{PREFIX}/output/results",
                "campaign_manifest": {
                    "relative_path": campaign_row["relative_path"],
                    "sha256": campaign_row["sha256"],
                },
                "files": rows,
            }
        ),
        encoding="utf-8",
    )
    return evidence


def test_campaign_io_evidence_binds_all_six_inputs(tmp_path: Path) -> None:
    result = campaign_io.validate_campaign_io_evidence(
        _io_evidence(tmp_path),
        source_commit=SOURCE,
        image_ref=IMAGE,
        model_manifest_digest=MODEL,
        volume_id="volume-1",
        data_center_id="EUR-IS-1",
    )

    assert result["status"] == "passed"
    assert len(result["files"]) == 6
    assert result["campaign_manifest_relative_path"].startswith(f"{PREFIX}/input/")
    assert result["output_relative_path"].startswith(f"{PREFIX}/output/")


def test_campaign_io_invalid_key_blocks_without_escaping_validation(
    tmp_path: Path,
) -> None:
    evidence = _io_evidence(tmp_path)
    value = json.loads(evidence.read_text(encoding="utf-8"))
    value["campaign_prefix"] = "../model-cache"
    evidence.write_text(json.dumps(value), encoding="utf-8")

    result = campaign_io.validate_campaign_io_evidence(
        evidence,
        source_commit=SOURCE,
        image_ref=IMAGE,
        model_manifest_digest=MODEL,
        volume_id="volume-1",
        data_center_id="EUR-IS-1",
    )

    assert result["status"] == "blocked"
    assert "campaign_io_prefix_invalid" in result["blockers"]


class _FakeClient:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.deleted = []

    def download_file(self, _bucket: str, key: str, destination: str) -> None:
        Path(destination).write_bytes(self.objects[key])

    def head_object(self, *, Bucket: str, Key: str):
        del Bucket
        return {"ContentLength": len(self.objects[Key])}

    def delete_objects(self, *, Bucket: str, Delete: dict) -> None:
        del Bucket
        for row in Delete["Objects"]:
            self.deleted.append(row["Key"])
            self.objects.pop(row["Key"], None)


def test_retrieve_verifies_hashes_core_schemas_and_complete_file_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_prefix = f"{PREFIX}/output/results"
    result = {
        "schema_version": worker.SCHEMA_VERSION,
        "status": "completed",
        "smoke_passed": True,
        "all_dynamic_episodes_completed": True,
        "runs": [{"attempt_id": row[0]} for row in worker.EXPECTED_ATTEMPTS],
    }
    local_files = {
        "campaign_result.json": json.dumps(result, sort_keys=True).encode(),
    }
    for attempt_id, _kind, _seed, _timeout in worker.EXPECTED_ATTEMPTS:
        local_files[f"{attempt_id}/attempt_result.json"] = json.dumps(
            {
                "schema_version": worker.ATTEMPT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "status": "completed",
            },
            sort_keys=True,
        ).encode()
    manifest = {
        "schema_version": worker.ARTIFACT_SCHEMA_VERSION,
        "status": "completed",
        "file_count": len(local_files),
        "total_size_bytes": sum(len(value) for value in local_files.values()),
        "files": [
            {
                "relative_path": name,
                "size_bytes": len(value),
                "sha256": _sha(value),
            }
            for name, value in sorted(local_files.items())
        ],
    }
    local_files["campaign_artifact_manifest.json"] = json.dumps(
        manifest, sort_keys=True
    ).encode()
    remote = {f"{output_prefix}/{name}": value for name, value in local_files.items()}
    client = _FakeClient(remote)
    monkeypatch.setattr(
        campaign_io,
        "_credentials",
        lambda *_args, **_kwargs: ("access", "secret", {"status": "passed"}),
    )
    monkeypatch.setattr(campaign_io, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(
        campaign_io,
        "_remote_keys",
        lambda _client, *, volume_id, prefix: sorted(
            key for key in remote if key.startswith(prefix)
        ),
    )
    contract = {
        "network_volume_id": "volume-1",
        "data_center_id": "EUR-IS-1",
        "output_relative_path": output_prefix,
    }

    retrieved = campaign_io.retrieve_campaign_outputs(
        contract,
        destination=tmp_path / "retrieved",
        access_key_file="unused",
        secret_key_file="unused",
    )

    assert retrieved["status"] == "completed"
    assert retrieved["transfer_status"] == "completed"
    assert retrieved["downloaded_file_count"] == 6
    assert retrieved["json_decode_failure_count"] == 0
