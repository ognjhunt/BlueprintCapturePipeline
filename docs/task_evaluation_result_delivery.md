# Task Evaluation Result delivery

Task Evaluation Results are delivered by a five-stage, fail-closed path. The
Pipeline remains the scientific and byte-integrity authority; the WebApp is a
tenant-scoped projection and authenticated transport.

1. **Validate** — verify the episode evidence index digest and independently
   re-hash every indexed receipt, multicamera manifest, exact lossless policy
   input, lossless camera frame, and external/wrist/overview review video.
2. **Seal** — bind the terminal Decision Envelope, evidence-index digest, run
   identity, and every customer-visible artifact into one delivery digest.
3. **Project** — produce a small secret-clean result projection for Firestore.
   Large media and evidence bytes remain Pipeline-owned.
4. **Package** — stream a deterministic ZIP64 review pack and full-evidence
   pack. The review pack contains human-review media and receipts; the full pack
   also contains the exact lossless policy inputs and camera frames. Packaging
   checks free disk first and never reconstructs missing evidence.
5. **Publish** — send the signed `task_evaluation_run_publication.v2` projection
   to the WebApp. The WebApp derives access from the authoritative capture owner
   and verified Firebase tenant; Pipeline cannot choose its audience.

## Customer and operations views

Each owner or verified organization sees only its own result cards, bounded
decision, five delivery stages, episode outcomes, external/wrist videos,
review-only overview, exact receipts/manifests, and review/full ZIP downloads.
Blueprint operations may inspect all tenant records for delivery health. There
is no public or cross-team leaderboard unless a separately authorized campaign
proves identical task, testbed, robot, candidates, seeds, scoring, and disclosure
rights.

## Storage and transport

Firestore stores the small immutable projection and access index. Evidence bytes
remain under the Pipeline run root and are served only by exact artifact ID from
the sealed registry. Each request is re-hashed by Pipeline, then streamed through
an authenticated WebApp proxy. Email, Google Drive, and ad hoc shared links are
not systems of record.

The current ADP rehearsal is `development_only`. Its videos are derived review
evidence, simulator results are not physical success, and successful execution
does not establish policy superiority, deployment approval, or safety.

## Private result artifact offload

ADP-009D's day-28 replay gate requires retained lossless policy inputs and derived
review media to remain readable after the control-plane hot window. The existing
storage GC now offloads registered bulk result payloads through the dedicated
artifact-store credentials (Backblaze B2 on the control plane). It keeps the
immutable registry, delivery, closure receipts, reports, and artifact IDs local.
Whole-directory cold archival excludes roots with a result registry.

Each eviction follows source SHA-256 verification, content-addressed upload,
complete remote byte readback, an atomic and fsynced registry-bound reference,
and final source re-verification. Active processes, storage pins, live queue
references, hot results, unsealed closure, unregistered bytes, and aliases with a
protected role retain their local payloads. Bulk uploads use four workers and
reserve space for the references. The existing offload opt-in and hot-window
settings govern the timer; no additional allocation or policy execution occurs.

The authenticated artifact endpoint retrieves only the requested object, verifies
its whole-file checksum before returning bytes, and retains the existing HTTP
range and content headers. A shared disk reservation bounds the temporary file.
Response completion, errors, and cancellation release it; a later request reaps
only abandoned download directories owned by dead service processes. No public
bucket ACL or external redirect is used. The `offloaded_bytes` receipt measures
logical payload bytes; compare filesystem free space separately because retained
hardlinks may still own the underlying blocks.

For an explicitly selected, sealed run, first inspect the dry run, then apply the
same command with `--apply --ack offload-sealed-result-artifacts`. Supply the
canonical pins and all queue roots; use `--hot-window-seconds 0` only for an
explicitly authorized immediate migration:

```bash
python -m blueprint_pipeline.task_evaluation_result_artifact_store \
  --run-root /var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries/RUN_ID \
  --pins-root /var/lib/blueprint/pipeline-control-plane/storage-pins \
  --queue-root /var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canary-dispatches \
  --report-out /var/lib/blueprint/pipeline-control-plane/storage-gc/result-artifacts.json
```

Validation covers signed full and range downloads after eviction, remote
corruption, run/registry/bucket substitution, alias handling, source changes,
interrupted responses, disk refusal, active-run protection, and the GC timer
integration. These are storage and transport claims; they do not change episode
validity, qualification, or physical-evidence claims.
