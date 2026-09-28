# Covers (for impacted-test selection):
#   tests/remote_cpu_fakes.py
"""ADP-009D/day-28, plan 14 PR 1: the hermetic fakes behave like B2, GCS and Cloud Run where it matters."""

from __future__ import annotations

import hashlib

import pytest
from botocore.exceptions import ClientError

from tests.remote_cpu_fakes import (
    FakeArtifactStore,
    FakeClock,
    FakeCloudRunError,
    FakeCloudRunJobs,
    FakeGcsError,
    FakeTransportBucket,
    env_value,
)

BUCKET = "b2-bucket"
JOB = "projects/blueprint-8c1ca/locations/us-central1/jobs/blueprint-remote-cpu-episode-compilation"
ATTEMPT_ENV = "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID"


def _status(error: ClientError) -> tuple[int, str]:
    return error.response["ResponseMetadata"]["HTTPStatusCode"], error.response["Error"]["Code"]


def test_fake_presigned_urls_expire_and_are_scoped_to_one_object() -> None:
    clock = FakeClock()
    store = FakeArtifactStore(clock=clock, bucket=BUCKET)
    store.put_object(Bucket=BUCKET, Key="cas/a.bin", Body=b"alpha")
    store.put_object(Bucket=BUCKET, Key="cas/b.bin", Body=b"bravo")

    get_url = store.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": "cas/a.bin"}, ExpiresIn=60, HttpMethod="GET"
    )
    assert "X-Amz-Signature=" in get_url and "X-Amz-Expires=60" in get_url
    response = store.request("GET", get_url)
    assert (response.status, response.body) == (200, b"alpha")
    assert store.request("GET", get_url.replace("cas/a.bin", "cas/b.bin")).status == 403
    assert store.request("PUT", get_url, body=b"overwrite").status == 403
    assert store.request("GET", get_url.replace("X-Amz-Expires=60", "X-Amz-Expires=6000")).status == 403
    clock.advance(59)
    assert store.request("GET", get_url).status == 200
    clock.advance(1)
    expired = store.request("GET", get_url)
    assert expired.status == 403 and b"Request has expired" in expired.body

    put_url = store.generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": "staging/rcj/a1/blobs.tar", "ContentLength": 5}, ExpiresIn=120
    )
    assert store.request("PUT", put_url, body=b"123456").status == 403
    assert store.request("PUT", put_url.replace("blobs.tar", "index.json"), body=b"12345").status == 403
    assert store.request("GET", put_url).status == 403
    stored = store.request("PUT", put_url, body=b"12345")
    assert stored.status == 200 and stored.headers["ETag"] == '"' + hashlib.md5(b"12345").hexdigest() + '"'
    assert store.head_object(Bucket=BUCKET, Key="staging/rcj/a1/blobs.tar")["ContentLength"] == 5
    assert store.bytes_received_from_client == len(b"alpha") + len(b"bravo") + 5

    with pytest.raises(ValueError):
        store.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": "cas/a.bin"}, ExpiresIn=604801)
    with pytest.raises(ValueError):
        store.generate_presigned_url("delete_object", Params={"Bucket": BUCKET, "Key": "cas/a.bin"}, ExpiresIn=60)

    bucket = FakeTransportBucket(clock=clock)
    name = "transport/rcj-ec-1/rcj-ec-1-a1-0-" + "a" * 32 + ".json"
    generation = bucket.create(name, b"{}", if_generation_match=0)
    with pytest.raises(FakeGcsError) as exists:
        bucket.create(name, b"{}", if_generation_match=0)
    assert exists.value.code == 412
    assert bucket.reader().get(name, generation=generation) == b"{}"
    replaced = bucket.create(name, b'{"x":1}', if_generation_match=generation)
    with pytest.raises(FakeGcsError) as old:
        bucket.get(name, generation=generation)
    assert old.value.code == 404 and replaced > generation
    assert not hasattr(bucket, "list_blobs") and [attr for attr in dir(bucket.reader()) if not attr.startswith("_")] == ["get"]
    bucket.delete(name, generation=replaced)
    assert bucket.exists(name, generation=replaced) is False


def test_fake_b2_delete_hides_until_versions_are_removed() -> None:
    store = FakeArtifactStore(bucket=BUCKET, min_part_bytes=4, max_copy_bytes=8)
    key = "staging/rcj-ec-1/rcj-ec-1-a1-0/blobs.tar"
    first = store.put_object(Bucket=BUCKET, Key=key, Body=b"v1", Metadata={"sha256": "1" * 64})
    second = store.put_object(Bucket=BUCKET, Key=key, Body=b"v2")
    assert first["VersionId"] != second["VersionId"]

    hidden = store.delete_object(Bucket=BUCKET, Key=key)
    assert hidden["DeleteMarker"] is True
    for call in (lambda: store.head_object(Bucket=BUCKET, Key=key), lambda: store.get_object(Bucket=BUCKET, Key=key)):
        with pytest.raises(ClientError) as missing:
            call()
        assert _status(missing.value)[0] == 404
    assert store.list_objects_v2(Bucket=BUCKET, Prefix="staging/")["KeyCount"] == 0
    listing = store.list_object_versions(Bucket=BUCKET, Prefix="staging/rcj-ec-1/")
    assert len(listing["Versions"]) == 2 and len(listing["DeleteMarkers"]) == 1
    assert store.get_object(Bucket=BUCKET, Key=key, VersionId=first["VersionId"])["Body"].read() == b"v1"

    pages, entries, markers = 0, [], {}
    while True:
        page = store.list_object_versions(Bucket=BUCKET, Prefix="staging/rcj-ec-1/", MaxKeys=1, **markers)
        pages += 1
        entries.extend(page.get("Versions", []) + page.get("DeleteMarkers", []))
        if not page["IsTruncated"]:
            break
        markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
    assert pages == 3 and len({entry["VersionId"] for entry in entries}) == 3
    for entry in entries:
        store.delete_object(Bucket=BUCKET, Key=entry["Key"], VersionId=entry["VersionId"])
    emptied = store.list_object_versions(Bucket=BUCKET, Prefix="staging/rcj-ec-1/")
    assert emptied.get("Versions", []) == [] and emptied.get("DeleteMarkers", []) == []

    source = store.put_object(Bucket=BUCKET, Key="staging/src.tar", Body=b"archive")
    received, sent = store.bytes_received_from_client, store.bytes_sent_to_client
    with pytest.raises(ClientError) as precondition:
        store.copy_object(Bucket=BUCKET, Key="cas/out.tar", CopySource={"Bucket": BUCKET, "Key": "staging/src.tar"},
                          CopySourceIfMatch='"stale"', MetadataDirective="REPLACE", Metadata={"sha256": "2" * 64})
    assert _status(precondition.value) == (412, "PreconditionFailed")
    store.copy_object(Bucket=BUCKET, Key="cas/out.tar", CopySource={"Bucket": BUCKET, "Key": "staging/src.tar"},
                      CopySourceIfMatch=source["ETag"], MetadataDirective="REPLACE", Metadata={"sha256": "2" * 64})
    copied = store.head_object(Bucket=BUCKET, Key="cas/out.tar")
    assert copied["Metadata"] == {"sha256": "2" * 64} and copied["ContentLength"] == len(b"archive")
    assert (store.bytes_received_from_client, store.bytes_sent_to_client) == (received, sent)

    big = store.put_object(Bucket=BUCKET, Key="staging/big.tar", Body=b"0123456789ab")
    with pytest.raises(ClientError) as too_large:
        store.copy_object(Bucket=BUCKET, Key="cas/big.tar", CopySource=f"{BUCKET}/staging/big.tar")
    assert _status(too_large.value)[0] == 400
    upload = store.create_multipart_upload(Bucket=BUCKET, Key="cas/big.tar", Metadata={"sha256": "3" * 64})["UploadId"]
    parts = [
        {"PartNumber": number, "ETag": store.upload_part_copy(
            Bucket=BUCKET, Key="cas/big.tar", UploadId=upload, PartNumber=number,
            CopySource={"Bucket": BUCKET, "Key": "staging/big.tar"}, CopySourceRange=span,
            CopySourceIfMatch=big["ETag"])["CopyPartResult"]["ETag"]}
        for number, span in ((1, "bytes=0-5"), (2, "bytes=6-11"))
    ]
    done = store.complete_multipart_upload(Bucket=BUCKET, Key="cas/big.tar", UploadId=upload,
                                           MultipartUpload={"Parts": parts})
    assert done["ETag"].endswith('-2"')
    assert store.get_object(Bucket=BUCKET, Key="cas/big.tar", Range="bytes=6-11")["Body"].read() == b"6789ab"
    assert store.head_object(Bucket=BUCKET, Key="cas/big.tar")["Metadata"] == {"sha256": "3" * 64}


def test_fake_cloud_run_listing_paginates_and_scripts_every_terminal_state() -> None:
    clock = FakeClock()
    jobs = FakeCloudRunJobs(clock=clock, max_page_size=2)
    jobs.add_job(JOB, image="gcr.io/blueprint-8c1ca/pipeline@sha256:" + "d" * 64,
                 command=["python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap"])
    job = jobs.get_job(JOB)
    assert (job["template"]["taskCount"], job["template"]["template"]["maxRetries"]) == (1, 0)
    assert job["template"]["template"]["containers"][0]["args"] == []

    def run(attempt: str, *, etag: str, validate_only: bool = False) -> dict:
        overrides = {"containerOverrides": [{"env": [{"name": ATTEMPT_ENV, "value": attempt}]}],
                     "taskCount": 1, "timeout": "1800s"}
        return jobs.run_job(JOB, etag=etag, overrides=overrides, validate_only=validate_only)

    assert run("probe", etag=job["etag"], validate_only=True)["metadata"] is None
    behaviours = ("succeed", "crash", "hang", "timeout", "lost_response", "duplicate")
    jobs.script(*behaviours)
    for index, behaviour in enumerate(behaviours):
        clock.advance(1)
        if behaviour == "lost_response":
            with pytest.raises(FakeCloudRunError) as lost:
                run(f"attempt-{index}", etag=job["etag"])
            assert lost.value.status is None
        else:
            assert run(f"attempt-{index}", etag=job["etag"])["metadata"]["name"].startswith(JOB + "/executions/")
    changed = jobs.update_job(JOB, timeout_seconds=900)
    with pytest.raises(FakeCloudRunError) as stale:
        run("stale-etag", etag=job["etag"])
    assert stale.value.code == "ABORTED" and changed["etag"] != job["etag"]

    def listing() -> tuple[list[dict], int]:
        rows, token, pages = [], None, 0
        while True:
            page = jobs.list_executions(JOB, page_size=100, page_token=token)
            pages += 1
            assert len(page["executions"]) <= 2
            rows.extend(page["executions"])
            token = page.get("nextPageToken")
            if not token:
                return rows, pages

    clock.advance(10)
    rows, pages = listing()
    assert (len(rows), pages) == (7, 4)
    assert all(row["runningCount"] == 1 and row.get("completionTime") is None for row in rows)
    clock.advance(1800)
    rows, _ = listing()
    create_times = [row["createTime"] for row in rows]
    assert create_times == sorted(create_times, reverse=True) and len({row["name"] for row in rows}) == 7
    by_attempt: dict[str, list[dict]] = {}
    for row in rows:
        by_attempt.setdefault(env_value(row, ATTEMPT_ENV), []).append(row)
    counts = {attempt: [(row["succeededCount"], row["failedCount"], row["cancelledCount"], row["runningCount"])
                        for row in executions] for attempt, executions in by_attempt.items()}
    assert counts == {
        "attempt-0": [(1, 0, 0, 0)], "attempt-1": [(0, 1, 0, 0)], "attempt-2": [(0, 0, 0, 1)],
        "attempt-3": [(0, 1, 0, 0)], "attempt-4": [(1, 0, 0, 0)], "attempt-5": [(1, 0, 0, 0), (1, 0, 0, 0)],
    }
    assert by_attempt["attempt-2"][0].get("completionTime") is None
    assert by_attempt["attempt-3"][0]["conditions"][0]["reason"] == "DeadlineExceeded"
    assert all(row.get("completionTime") for attempt, group in by_attempt.items() if attempt != "attempt-2" for row in group)

    hung = by_attempt["attempt-2"][0]["name"]
    cancelled = jobs.cancel_execution(hung)
    assert (cancelled["cancelledCount"], cancelled["runningCount"]) == (1, 0) and cancelled["completionTime"]
    with pytest.raises(FakeCloudRunError) as again:
        jobs.cancel_execution(hung)
    assert again.value.code == "FAILED_PRECONDITION"
    with pytest.raises(FakeCloudRunError) as bad_token:
        jobs.list_executions(JOB, page_token="not-a-token")
    assert bad_token.value.code == "INVALID_ARGUMENT"
    with pytest.raises(FakeCloudRunError) as missing:
        jobs.get_execution(JOB + "/executions/unknown-00000")
    assert missing.value.status == 404
