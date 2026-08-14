# SAM 3.1 source-track canary

## Purpose and claim ceiling

This lane runs one authorized Meta SAM 3.1 Object Multiplex worker over exact,
hash-bound source frames and returns persistent 2D binary-mask tracks or an
explicit abstention. It does not establish object identity as observed fact,
metric location, 3D extent, collision geometry, physics, task success,
deployment readiness, safety, or comparative policy ranking. The frozen ranking
verdict remains `thesis_not_supported`.

The image is separate from `deploy/docker/sam3`, which remains the privacy
service. Do not replace or modify that service to run this canary.

## Required immutable inputs

The source-track request must truthfully provide:

- exact retained-video, capture, camera-solution, frame-registry, and source
  frame digests;
- decoded presentation timestamps and hash-bound sync-map rows;
- proof that each analyzed JPEG corresponds to an available retained source
  frame;
- exact camera records for later lifting;
- the permitted `semantic_analysis` use only;
- license-use, privacy-use, trade-control, checkpoint-access, and execution
  authorization digests;
- the exact versioned worker image digest and source commit.

MuSHRoom `koivu` is useful processed indoor walkthrough evidence, but its public
archive contains processed RGB/depth/poses rather than the original retained
video and encoder-attempt/retention ledger. Do not populate the fields above
with archive or derived-video digests. Until a reduced-authority processed
sequence contract is implemented, MuSHRoom may exercise hermetic reconstruction,
testbed, Router, and abstention paths but not this retained-video import path.

## Image and checkpoint

The worker recipe is `deploy/docker/sam31_source_tracks/Dockerfile`. Publication
must use an immutable versioned tag and then resolve and record its registry
digest. Never use `latest`.

The image pins the official SAM code but does not contain the gated checkpoint.
At canary startup the adapter downloads `facebook/sam3.1` file
`sam3.1_multiplex.pt` at repository revision
`daa63191845a41281374e725f4c9e51c7a824460`, verifies SHA-256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`,
unsets the Hugging Face token, enables offline model loading, and only then runs
inference. The token and signed URLs must be stored in owner-only (`0600`)
files and must not appear in persisted artifacts.

## Admission and execution

Use only the canonical paid-resource command:

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary \
  --provider vast \
  --probe-kind semantic-sam31-source-tracks \
  --provider-launch-request /path/to/sam31_canary_request.json \
  --release-evidence /path/to/release_evidence.json \
  --model-cache-evidence /path/to/model_evidence.json \
  --preflight-bundle /path/to/sam31_vast_preflight.json \
  --admission-out /path/to/sam31_admission.json \
  --bound-request-out /path/to/sam31_bound_request.json \
  --adapter-output /path/to/sam31_execution.json \
  --pod-name sam31-source-track-canary \
  --expected-source-commit <exact-40-character-commit> \
  --sam31-input-bundle /host/immutable/sam31-input.zip \
  --sam31-input-bundle-receipt /host/immutable/sam31-input-receipt.json \
  --sam31-attempt-authority /host/authorities/sam31-paid-attempt-authority.json \
  --sam31-hf-token-file /private/hf-token.txt \
  --sam31-max-hourly-rate-usd <explicit-rate-ceiling> \
  --sam31-max-spend-usd <explicit-cap> \
  --sam31-hard-ttl-seconds <at-most-3600> \
  --sam31-retry-cap 0 \
  --sam31-authority-id <authority-id> \
  --execute
```

Before `--execute`, require a clean exact protected-main commit or a clean,
pushed `codex/` diagnostic branch with `--experimental-branch-diagnostic`; a
digest-bound one-shot authority file; current Vast capacity; and API-confirmed
zero global Vast inventory. A configured provider or a free-form authority ID
is not enough. The allocator consumes the authority exactly once before it
stages any bytes, creates the signed object-store transport itself, and arms its
run-owned independent watchdog before provider create.

The lifecycle opens a paid-lane lease and pending teardown before create,
records the exact instance ID, polls only for the signed output, validates all
bindings and claim ceilings, terminates the instance in `finally`, deletes and
absence-proves both staged objects, removes signed URL files, closes the
watchdog against API inventory, and requires scoped and global provider zero.
Any ambiguity, stale input, runtime mismatch, secret-file permission failure,
timeout, disagreement, cleanup failure, or teardown failure is terminal
failure or abstention, never a pass.
