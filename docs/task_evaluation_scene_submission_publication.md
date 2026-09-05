# Publish an assembled scene-construction submission

Run the protected-main module on the production host as the configured service
account, after exact-main deployment and successful assembler validation:

```bash
python -m blueprint_pipeline.task_evaluation_scene_configuration_submission_publication \
  --manifest <immutable-staging-root>/bundle_manifest.v1.json \
  --receipt-out <production-receipts>/scene-submission-publication.v1.json \
  --expected-source-commit <exact-deployed-protected-main-SHA> \
  --service-account blueprint
```

The receipt must be outside the staging directory. The immutable manifest
inventories every staged file except itself; the publisher verifies that exact
inventory, the request digest and references, execution commit, and all local
hashes before performing any upload.

Publisher source rows must retain their HTTPS publisher URIs and
`publication_allowed=false`. Every publisher artifact must be present in this
host-only inventory. Source directory rows and exact raw-source digests cannot
be relabeled as uploadable artifacts. The source-binding resolver handles their
separate production-host consumption.

Only admitted files and the bundle manifest are uploaded to their exact
`s3://blueprint/task-evaluation/production-inputs/<namespace>/` locations.
Credentials come from the existing production object-store secret-file client.
Small objects stream from a file; larger objects use bounded multipart chunks.
A production-host namespace lock serializes cooperating publishers. Existing
objects receive exact full-byte readback and differing bytes block the run.
The operator must exclusively own this fresh namespace: this is not global
object-store compare-and-swap and does not prevent arbitrary external writers.
No undocumented provider conditional-write headers are required. Every object
receives a full streaming service-account readback.

Identical existing objects are reverified and reused. Different existing bytes
block publication. Repeating an exact successful publication revalidates remote
bytes and returns the original self-digesting receipt without duplicate uploads.
Interrupted publication can resume by verifying its existing partial inventory;
failed multipart attempts are aborted. Do not mistake a partially populated
object prefix for a successful publication receipt.

This command does not submit a request or allocate a provider. After its
`published_and_read_back` receipt exists, use the existing
`scripts/submit_task_evaluation_preparation_via_webapp.py` command for the exact
validated request and an exclusive Website submission receipt.
