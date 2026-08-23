from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.postshot_license_transport import (
    PostshotLicenseTransportError,
    close_postshot_license,
    stage_postshot_license,
)


class _Missing(Exception):
    response = {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}


class _Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, **kwargs):
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]

    def generate_presigned_url(self, operation, **kwargs):
        return f"https://objects.example/{operation}/{kwargs['Params']['Key']}?signed=1"

    def delete_object(self, **kwargs):
        self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)

    def head_object(self, **kwargs):
        if (kwargs["Bucket"], kwargs["Key"]) not in self.objects:
            raise _Missing()
        return {"ContentLength": len(self.objects[(kwargs["Bucket"], kwargs["Key"])])}


def _license(path: Path) -> None:
    path.write_text(
        "POSTSHOT_LOGIN_EMAIL=operator@example.com\nPOSTSHOT_LOGIN_PASSWORD=not-recorded\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_license_transport_is_single_use_and_receipts_never_record_urls_or_values(tmp_path: Path) -> None:
    source = tmp_path / "license.env"
    _license(source)
    client = _Client()
    staged = stage_postshot_license(
        job_dir=tmp_path / "run",
        license_file=source,
        expiration_seconds=600,
        client=client,
        bucket="private-bucket",
    )
    receipt_text = (tmp_path / "run/postshot_license_transport.json").read_text()
    assert "operator@example.com" not in receipt_text
    assert "signed=1" not in receipt_text
    closed = close_postshot_license(staged=staged, job_dir=tmp_path / "run")
    assert closed["status"] == "closed"
    assert closed["object_absence_confirmed"] is True


def test_license_transport_refuses_group_readable_or_injected_files(tmp_path: Path) -> None:
    source = tmp_path / "license.env"
    _license(source)
    source.chmod(0o640)
    with pytest.raises(PostshotLicenseTransportError, match="permissions_invalid"):
        stage_postshot_license(
            job_dir=tmp_path / "run",
            license_file=source,
            expiration_seconds=600,
            client=_Client(),
            bucket="private-bucket",
        )
    source.chmod(0o600)
    source.write_text(source.read_text() + "PATH=/evil\n", encoding="utf-8")
    with pytest.raises(PostshotLicenseTransportError, match="schema_invalid"):
        stage_postshot_license(
            job_dir=tmp_path / "run2",
            license_file=source,
            expiration_seconds=600,
            client=_Client(),
            bucket="private-bucket",
        )
