# PR 3 — The listener ends expired and revoked scenes (design doc 1c, listener half)

> Read `00-index.md` first.

**Goal:** stop re-staging and re-failing scenes whose website authority ended.
- `consent_expired`, `source_revoked` and similar authority endings finish the handoff job as
  terminal. The message is acknowledged with a terminal receipt, and a redelivery is
  acknowledged without staging anything.
- The listener records what it acknowledged, and what it staged from Firebase Storage. PR 4's
  retirement relies on both records.

**Branch / base / worktree:** `claude/disk-1c-listener-terminal-authority` from `origin/main`,
`/Users/nijelhunt_1/workspace/BlueprintCapturePipeline-disk-1c1-20260926`.

## Facts (verified; line numbers in `src/blueprint_pipeline/pubsub_handoff_listener.py`)

- **Where authority endings come from.**
  - The WebApp answers `409 {code: "consent_expired" | "source_revoked" | …}` from
    `server/routes/internal-capture-worlds.ts`.
  - `website_task_context.website_webapp_request` (69-84) raises
    `ValueError(f"website_control_{operation}_http_{status}:{code}")`, and the codes match
    `[a-z][a-z0-9_]{0,99}`.
  - The exception escapes `run_e2e` wrapped in `PipelineError` / `StageError`.
- **Why they retry forever.**
  - `process_handoff_payload`'s exception handler (1373-1397) finishes the lease as
    `failed_retryable` and re-raises.
  - `pull_and_process` (1696-1717) then calls `heartbeat.defer_retry()` (600 s) and never
    acknowledges. Every redelivery re-claims the lease and `stage_handoff_capture` re-downloads
    the whole capture prefix (159-192).
- **Lease and staging order.** `_claim_job_lease` (627-690) runs before staging. It returns
  `completed | active | corrupt | claimed`, and a completed claim returns early (1178-1213).
- **Acknowledgement.** Acks are batched in one `subscriber.acknowledge` call at the end of
  `pull_and_process` (1746-1747), and nothing records them per capture. The comment at 1651-1653
  ("explicitly nacked") is stale: the code never nacks.
- **Only terminal codes end a job.** The WebApp's own forwarder treats `consent_expired` and
  `source_revoked` as terminal (`server/utils/taskEvaluationSceneIntake.ts`). Other 409 codes
  (`task_brief_missing`, `provider_terms_not_configured_or_changed`, `idempotency_conflict`) can be
  fixed by a person, so they stay retryable.
- **Dead-letter IAM is missing.** `deploy/terraform/main.tf:904-922` configures a dead-letter
  policy (5 attempts), but nothing grants the Pub/Sub service agent publisher on
  `pipeline-trigger-dlq` or subscriber on the subscription, so dead-lettering cannot happen.
  `data "google_project" "current"` exists at `main.tf:552`.

## Design

```python
AUTHORITY_ENDING_CODES = frozenset({"consent_expired", "source_revoked"})
_AUTHORITY_ENDING_RE = re.compile(
    r"website_control_[a-z0-9-]+_http_409:(consent_expired|source_revoked)(?![a-z0-9_])")
TERMINAL_AUTHORITY_STATUS = "terminal_authority_ended"
JOB_TERMINAL_RECEIPT_FILENAME = "pipeline_job_terminal_receipt.json"
JOB_TERMINAL_RECEIPT_SCHEMA_VERSION = "pipeline_job_terminal_receipt.v1"
JOB_ACK_RECEIPT_FILENAME = "pipeline_job_ack_receipt.json"
JOB_ACK_RECEIPT_SCHEMA_VERSION = "pubsub_handoff_ack_receipt.v1"
STAGING_MANIFEST_FILENAME = "pipeline_staging_manifest.json"
STAGING_MANIFEST_SCHEMA_VERSION = "pipeline_handoff_staging_manifest.v1"


def authority_ending_code(exc: BaseException) -> str | None:
    """The WebApp authority code that permanently ended this job, if any.

    Walks the exception chain (__cause__, then __context__), at most 16 links, and
    guards against cycles. Only exact typed WebApp 409 codes qualify.
    """


def payload_sha256(payload: bytes | str | Mapping[str, Any]) -> str:
    """Hex sha256 of the message bytes, as `_write_delivery_evidence` already records it.

    str -> utf-8 bytes; Mapping -> json.dumps(sort_keys=True, separators=(",", ":")).
    """
```

**Process path**
- `process_handoff_payload(..., payload_digest=None)` computes `payload_sha256(payload)` when not
  given.
- `_claim_job_lease(..., payload_sha256=...)` returns `("terminal", ledger)` when the ledger status
  is `terminal_authority_ended` and the ledger's `terminal_payload_sha256` equals the incoming
  digest. A different payload reopens the job: the lease is claimed as today, and a history row
  `{"status": "reopened_after_terminal_authority", ...}` is appended.
- On `"terminal"`, return without staging:

```python
{"schema_version": "v1", "status": "skipped_terminal_authority_ended",
 "queue_disposition": TERMINAL_AUTHORITY_STATUS, "bucket": ..., "scene_id": ..., "capture_id": ...,
 "capture_root": str(capture_root), "blockers": [ledger["terminal_code"]], "job_ledger": ledger}
```

- In the exception handler, compute `code = authority_ending_code(exc)` before writing
  `failed_retryable`. When `code` is not None:
  - Call `_finish_job_lease(... update={...})` with:
    - `status: TERMINAL_AUTHORITY_STATUS`, `terminal_code: code`, `terminal_at`, `updated_at`;
    - `terminal_payload_sha256`, `last_error_type`, `last_error` (at most 500 characters);
    - `queue_disposition: TERMINAL_AUTHORITY_STATUS`;
    - `attempt_history` plus `{attempt_number, status: TERMINAL_AUTHORITY_STATUS, stage, started_at, ended_at, code}`.
  - Write `pipeline_job_terminal_receipt.json` through `write_json`:
    - `schema_version`, `status: "authority_ended"`, `code`, `bucket`, `scene_id`, `capture_id`;
    - `attempt_count`, `payload_sha256`, `ended_at`, `error` (at most 500 characters);
    - `receipt_digest`: `"sha256:" + sha256(canonical JSON without the digest field)`.
  - **Return** the terminal result shape above with status `terminal_authority_ended`. Do not
    re-raise.
- `pull_and_process` needs no change for the terminal case. It acknowledges any result that is not
  retryable. Add the disposition to the `acked` bookkeeping below.

**Ack receipts.**
- `pull_and_process` collects `(capture_root, message_id, payload_sha256, delivery_attempt,
  disposition)` for every message it acknowledges.
- After `subscriber.acknowledge(...)` **returns**, it writes (or replaces) `pipeline_job_ack_receipt.json`
  in each capture root:

```json
{"schema_version": "pubsub_handoff_ack_receipt.v1", "subscription": "<resource>",
 "message_id": "...", "payload_sha256": "...", "delivery_attempt": 1,
 "disposition": "terminal_success" | "terminal_authority_ended",
 "acknowledged_at": "<utc iso>", "acknowledgement_count": 1}
```

- `acknowledgement_count` increments across redeliveries.
- If `acknowledge` raises, write no receipt (let the exception propagate as today).
- Permanent-invalid payloads have no capture root and keep only their delivery evidence.

**Staging manifest.** `stage_handoff_capture` records every listed blob and writes
`pipeline_staging_manifest.json` after all downloads succeed:

```json
{"schema_version": "pipeline_handoff_staging_manifest.v1", "bucket": "...",
 "prefix": "scenes/<s>/captures/<c>/", "staged_at": "<utc iso>",
 "objects": [{"name": "scenes/<s>/captures/<c>/raw/video.mov", "relative_path": "raw/video.mov",
              "size": 123, "generation": "1790...", "md5_hash": "<base64>", "crc32c": "<base64>"}]}
```

- Values come from `getattr(blob, attr, None)`. `generation` is stored as a string.
- Before downloading a blob, skip it when the previous manifest has the same `name` with non-null
  `generation` and `size` equal to the blob's, and the local file exists with that size. Skipped
  blobs keep their manifest row. A blob whose generation or size is unknown always downloads.

**Status reader.** `read_handoff_job_status` reports:
- `terminal_code`;
- `terminal_receipt_present`;
- `ack_receipt` (the parsed receipt, or null);
- `retry_expected_on_redelivery: False` for the terminal status.

**Infra.** Add the service-agent grants to `deploy/terraform/main.tf`:

```hcl
locals {
  pubsub_service_agent = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_topic_iam_member" "pipeline_dlq_pubsub_agent_publisher" {
  topic  = google_pubsub_topic.pipeline_dlq.name
  role   = "roles/pubsub.publisher"
  member = local.pubsub_service_agent
}

resource "google_pubsub_subscription_iam_member" "pipeline_handoff_listener_pubsub_agent_subscriber" {
  subscription = google_pubsub_subscription.pipeline_handoff_listener.name
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_service_agent
}
```

Add the same two bindings to `deploy/scripts/deploy.sh`, beside the subscription creation at
857-866, using `gcloud pubsub topics add-iam-policy-binding` /
`gcloud pubsub subscriptions add-iam-policy-binding`. Resolve the project number with
`gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)'`.

## Tasks

### Task 3.1 — classify authority endings

- [ ] **Step 1: failing tests** in `tests/test_pubsub_handoff_listener.py`:

```python
@pytest.mark.parametrize("code", ["consent_expired", "source_revoked"])
def test_authority_ending_code_walks_the_exception_chain(code):
    try:
        try:
            raise ValueError(f"website_control_scene-sponsorship_http_409:{code}")
        except ValueError as inner:
            raise listener_module.PipelineError("website_task_context failed") from inner
    except listener_module.PipelineError as outer:
        assert listener_module.authority_ending_code(outer) == code


@pytest.mark.parametrize("message", [
    "website_control_scene-sponsorship_http_409:task_brief_missing",
    "website_control_scene-sponsorship_http_503:consent_expired",
    "website_control_scene-sponsorship_http_409:consent_expired_soon",
    "consent_expired",
])
def test_other_failures_are_not_authority_endings(message):
    assert listener_module.authority_ending_code(ValueError(message)) is None
```

- [ ] **Steps 2–5:** implement `authority_ending_code` and `payload_sha256`; commit "Recognize when the website has permanently ended a scene's authority".

### Task 3.2 — terminal job, terminal receipt, no restaging

- [ ] **Step 1: failing tests.** Follow the file's patterns: `FakeStorageClient`, `FakeSubscriber`,
  `SimpleNamespace` received messages, `_ios_bundle_blobs`, and the website fixtures around
  1252-1340.

```python
def _expired(*_args, **_kwargs):
    try:
        raise ValueError("website_control_scene-sponsorship_http_409:consent_expired")
    except ValueError as exc:
        raise listener_module.PipelineError("website scene failed") from exc


def test_consent_expired_finishes_the_job_as_terminal_and_is_acknowledged(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES, delivery_attempt=1)])
    _install_fake_pubsub(monkeypatch, subscriber)
    acknowledged = pull_and_process(subscription="sub", storage_root=tmp_path, provider="openai",
        storage_client=FakeStorageClient(_ios_bundle_blobs()), run_e2e=_expired)
    assert acknowledged == 1 and subscriber.acknowledged == ["a1"]
    ledger = json.loads((_capture_root(tmp_path) / "pipeline_job_ledger.json").read_text())
    assert ledger["status"] == "terminal_authority_ended"
    assert ledger["terminal_code"] == "consent_expired"
    receipt = json.loads((_capture_root(tmp_path) / "pipeline_job_terminal_receipt.json").read_text())
    assert receipt["status"] == "authority_ended" and receipt["receipt_digest"].startswith("sha256:")
    ack = json.loads((_capture_root(tmp_path) / "pipeline_job_ack_receipt.json").read_text())
    assert ack["disposition"] == "terminal_authority_ended"


def test_redelivered_terminal_capture_is_acknowledged_without_staging(tmp_path, monkeypatch):
    ...  # first pull as above; second pull with a storage client whose list_blobs fails the test
    assert second_result_status == "skipped_terminal_authority_ended"


def test_a_new_payload_reopens_a_terminal_capture(tmp_path, monkeypatch):
    ...  # terminal first, then a payload with a different pipeline_handoff_uri or robot_eval field
    assert run_e2e_calls == 1   # the reopened job runs


def test_non_terminal_409_stays_retryable(tmp_path, monkeypatch):
    ...  # run_e2e raises ...http_409:task_brief_missing -> not acknowledged, ledger failed_retryable
```

Reuse or add helpers `_received`, `_install_fake_pubsub`, `_capture_root` and `PAYLOAD_BYTES` from
the file's existing tests. Do not duplicate one that already exists under another name.

- [ ] **Steps 2–5:** implement; run the whole listener test file; commit "Acknowledge scenes whose authority ended instead of retrying them forever".

### Task 3.3 — ack receipts

- [ ] **Step 1: failing tests**

```python
def test_ack_receipt_is_written_after_acknowledge_returns(tmp_path, monkeypatch): ...
def test_no_ack_receipt_when_acknowledge_fails(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([...])
    subscriber.acknowledge = lambda **_k: (_ for _ in ()).throw(RuntimeError("pubsub down"))
    with pytest.raises(RuntimeError):
        pull_and_process(...)
    assert not (_capture_root(tmp_path) / "pipeline_job_ack_receipt.json").exists()
```

- [ ] **Steps 2–5:** implement; fix the stale "explicitly nacked" comment; commit "Record every acknowledged handoff in its capture".

### Task 3.4 — staging manifest and unchanged-object skip

Extend the fakes. `FakeBlob` gains optional `size`, `generation`, `md5_hash` and `crc32c`, and
counts `download_to_filename` calls.

- [ ] **Step 1: failing tests**

```python
def test_staging_manifest_records_cloud_identity(tmp_path): ...   # rows carry size/generation/md5/crc32c
def test_unchanged_objects_are_not_downloaded_again(tmp_path):
    blob = FakeBlob("scenes/s/captures/c/raw/capture_upload_complete.json", b"{}", size=2, generation=7)
    ...
    stage_handoff_capture(handoff, storage_root=tmp_path, client=client)
    stage_handoff_capture(handoff, storage_root=tmp_path, client=client)
    assert blob.download_count == 1
def test_changed_generation_downloads_again(tmp_path): ...
```

- [ ] **Steps 2–5:** implement; commit "Remember what was staged from Firebase Storage and skip unchanged objects".

### Task 3.5 — dead-letter IAM and status reader

- [ ] Terraform and `deploy.sh` bindings as above.
- [ ] Extend `scripts/validate_pubsub_handoff_infra.py` so the static validator requires both
  resources in `main.tf`, with a test in `tests/test_validate_pubsub_handoff_infra.py` and a
  fixture main.tf without them that fails.
- [ ] `read_handoff_job_status` terminal fields, with a test.
- [ ] Commit "Let the Pub/Sub service agent dead-letter exhausted handoffs".
- [ ] Record in the PR description: **owner action** — `terraform apply` (or the `deploy.sh`
  bindings) is needed for dead-lettering to take effect. The listener change alone ends the
  retry loop for authority endings.

## PR verification

- `tests/test_pubsub_handoff_listener.py` (entire file);
- `tests/test_validate_pubsub_handoff_infra.py`;
- `tests/test_runtime_security_controls.py -k handoff`;
- `tests/test_deploy_systemd_contract.py`.
